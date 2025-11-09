import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier # <-- Import the Neural Network
from sklearn.pipeline import Pipeline
import joblib

# 1. Load Data
print("Loading data...")
df = pd.read_csv('IMDB Dataset.csv')
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# 2. Split Data
X = df['review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create a model pipeline
print("Training model (this may take a few minutes)...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
    ('model', MLPClassifier(
        hidden_layer_sizes=(100,),  # One hidden layer with 100 neurons
        max_iter=200,               # Max training cycles
        early_stopping=True,        # Stop if the model isn't improving
        random_state=42,
        verbose=True                # Print training progress
    ))
])

# 4. Train the model
pipeline.fit(X_train, y_train)

# 5. Evaluate the model
accuracy = pipeline.score(X_test, y_test)
print(f"Model trained with accuracy: {accuracy * 100:.2f}%")

# 6. Save the model pipeline
joblib.dump(pipeline, 'sentiment_model.joblib')

print("New Neural Net model saved as 'sentiment_model.joblib'")