import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pickle

# Load the dataset
data = pd.read_csv('sample_sentiment_data.csv')

# Check data
print("Data Sample:")
print(data.head())
print("\nSentiment Distribution:")
print(data['sentiment'].value_counts())

# Split the data into features and target
X = data['text']
y = data['sentiment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that combines TF-IDF Vectorization and Logistic Regression
pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'logisticregression__C': [0.01, 0.1, 1, 10],
    'tfidfvectorizer__max_features': [None, 1000, 5000]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print best parameters
print(f'Best parameters: {grid_search.best_params_}')

# Train the model with the best parameters
model = grid_search.best_estimator_
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred, average="weighted")}')
print(f'Recall: {recall_score(y_test, y_pred, average="weighted")}')
print(f'F1 Score: {f1_score(y_test, y_pred, average="weighted")}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
