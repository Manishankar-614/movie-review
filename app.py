from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from flask_caching import Cache
from flask_migrate import Migrate
from dotenv import load_dotenv
import joblib
import numpy as np
import requests
import os
import math
from functools import wraps

load_dotenv() 

# --- 1. APP & DB CONFIGURATION ---
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
OMDB_API_KEY = os.getenv('OMDB_API_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///' + os.path.join(basedir, 'instance', 'site.db'))
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
migrate = Migrate(app, db) 

login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

model = joblib.load('sentiment_model.joblib')

# --- 2. DATABASE MODELS ---
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    is_admin = db.Column(db.Boolean, nullable=False, default=False)
    reviews = db.relationship('Review', backref='author', lazy=True)

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    imdb_id = db.Column(db.String(20), nullable=True)

# --- 3. ADMIN DECORATOR ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin:
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('app_index'))
        return f(*args, **kwargs)
    return decorated_function

# --- 4. AUTHENTICATION & PUBLIC ROUTES ---
@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for('app_index'))
    return render_template('home.html')

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('app_index'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        is_admin_user = True if email == 'admin@admin.com' else False
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password=hashed_password, is_admin=is_admin_user)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('app_index'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user, remember=True)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('app_index'))
        else:
            flash('Login unsuccessful. Please check email and password.', 'danger')
    return render_template('login.html')

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))

# --- 5. INTERNAL APP ROUTES ---
@app.route("/app")
@login_required
def app_index():
    return render_template('index.html')

@app.route("/app/profile")
@login_required
def profile():
    reviews = Review.query.filter_by(user_id=current_user.id).order_by(Review.id.desc()).all()
    
    total_reviews = len(reviews)
    positive_count = sum(1 for r in reviews if r.sentiment == 'Positive')
    negative_count = total_reviews - positive_count
    
    if total_reviews > 0:
        positive_percent = (positive_count / total_reviews) * 100
        negative_percent = (negative_count / total_reviews) * 100
    else:
        positive_percent = 0
        negative_percent = 0
        
    stats = {
        "total": total_reviews,
        "positive": positive_count,
        "negative": negative_count,
        "positive_percent": positive_percent,
        "negative_percent": negative_percent
    }
    
    return render_template('profile.html', reviews=reviews, stats=stats)

@app.route("/app/movie", methods=['GET', 'POST'])
@login_required
def movie():
    search_results = None
    search_term = None
    total_pages = 0
    page = request.args.get('page', 1, type=int)

    if request.method == 'GET' and request.args.get('s'):
        search_term = request.args.get('s')
    elif request.method == 'POST':
        search_term = request.form.get('movie_title')
    
    if search_term:
        # --- THIS IS THE FIX ---
        api_url = f'http://www.omdbapi.com/?s={search_term}&apikey={OMDB_API_KEY}&page={page}'
        try:
            response = requests.get(api_url)
            data = response.json()
            if data.get('Response') == 'True':
                search_results = data.get('Search')
                total_results = int(data.get('totalResults', 0))
                total_pages = math.ceil(total_results / 10)
            else:
                flash(f"Error: {data.get('Error')}", 'danger')
        except requests.exceptions.RequestException:
            flash('Error connecting to the movie database.', 'danger')
            
    return render_template('movie.html', 
                           search_results=search_results, 
                           search_term=search_term,
                           current_page=page,
                           total_pages=total_pages)

@app.route("/app/movie/<string:imdb_id>")
@login_required
@cache.cached(timeout=86400)
def movie_details(imdb_id):
    movie_data = None
    reviews = []
    
    api_url = f'http://www.omdbapi.com/?i={imdb_id}&plot=full&apikey={OMDB_API_KEY}'
    try:
        response = requests.get(api_url)
        movie_data = response.json()
        if movie_data.get('Response') == 'False':
            movie_data = None
    except requests.exceptions.RequestException:
        flash('Error connecting to the movie database.', 'danger')
        
    if movie_data:
        reviews = db.session.query(Review, User).join(User).filter(Review.imdb_id == imdb_id).order_by(Review.id.desc()).all()
        
    return render_template('movie_details.html', movie_data=movie_data, reviews_with_users=reviews)

@app.route('/app/predict', methods=['POST'])
@login_required
def predict():
    review_text = request.form['review_text']
    imdb_id = request.form.get('imdb_id')
    
    data_to_predict = [review_text]
    probabilities = model.predict_proba(data_to_predict)
    confidence = np.max(probabilities)
    prediction = model.predict(data_to_predict)[0]
    prediction_text = 'Positive' if prediction == 1 else 'Negative'
        
    review = Review(
        content=review_text, 
        sentiment=prediction_text, 
        confidence=float(confidence),  # <-- This fix is already in your file
        user_id=current_user.id,
        imdb_id=imdb_id
    )
    db.session.add(review)
    db.session.commit()
    
    if imdb_id:
        cache.delete_memoized(movie_details, imdb_id)
    
    if request.form.get('is_ajax') == 'true':
        return jsonify({
            'status': 'success',
            'prediction_text': prediction_text,
            'prediction': int(prediction),
            'confidence': f'{float(confidence)*100:.1f}' # <-- This fix is already in your file
        })
    
    flash('Your review has been analyzed and saved!', 'success')
    return redirect(url_for('movie_details', imdb_id=imdb_id))

# --- 6. ADMIN ROUTES ---
@app.route("/admin")
@login_required
@admin_required
def admin():
    reviews = db.session.query(Review, User).join(User).order_by(Review.id.desc()).all()
    return render_template('admin.html', reviews_with_users=reviews)

@app.route("/admin/delete/<int:review_id>", methods=['POST'])
@login_required
@admin_required
def delete_review(review_id):
    review_to_delete = Review.query.get_or_404(review_id)
    
    if review_to_delete.imdb_id:
        cache.delete_memoized(movie_details, review_to_delete.imdb_id)
        
    db.session.delete(review_to_delete)
    db.session.commit()
    flash('Review has been deleted.', 'success')
    return redirect(url_for('admin'))

# --- 7. RUN THE APP ---
if __name__ == '__main__':
    with app.app_context():
        if not os.path.exists(os.path.join(basedir, 'instance')):
            os.makedirs(os.path.join(basedir, 'instance'))
        db.create_all()
    app.run(debug=True)