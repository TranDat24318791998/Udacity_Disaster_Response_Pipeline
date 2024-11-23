import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import joblib  # To save the model

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def load_data(database_filepath):
    """
    Load data from SQLite database.
    
    Args:
    database_filepath: str. Filepath of the database.
    
    Returns:
    X: DataFrame. Feature data (messages).
    Y: DataFrame. Target data (categories).
    category_names: list. List of category names for the classification.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterMessages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]  # Assuming categories start from the 5th column
    category_names = Y.columns.tolist()
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize and preprocess text data.
    
    Args:
    text: str. Text data to tokenize.
    
    Returns:
    tokens: list. Processed list of tokens.
    """
    # Normalize text and tokenize
    text = text.lower()
    tokens = word_tokenize(text)
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    
    return tokens

def build_model():
    """
    Build a machine learning pipeline with an LGBMClassifier.
    
    Returns:
    model: Pipeline. Machine learning pipeline.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)))
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model on the test set and print classification reports.
    
    Args:
    model: Pipeline. Trained machine learning model.
    X_test: DataFrame. Test feature data.
    Y_test: DataFrame. Test target data.
    category_names: list. List of category names.
    """
    Y_pred = model.predict(X_test)
    for i, column in enumerate(category_names):
        print(f'Category: {column}')
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.
    
    Args:
    model: Pipeline. Trained machine learning model.
    model_filepath: str. Filepath to save the pickle file.
    """
    joblib.dump(model, model_filepath)

def main():
    """
    Main function to execute the training pipeline.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)
        
        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
