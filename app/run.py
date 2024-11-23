import sys
sys.path.append('../data')  
from process_data import clean_text  

import json
import plotly
import pandas as pd


from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
# from data.process_data import clean_text  # Import clean_text from process_data.py

import logging

logging.basicConfig(level=logging.DEBUG)  

app = Flask(__name__)

def tokenize(text):
    """
    Tokenize and preprocess input text using the clean_text function.
    
    Args:
    text: str. The raw text to process.
    
    Returns:
    tokens: list. A list of processed tokens.
    """
    # Preprocess the text using clean_text to ensure consistency
    cleaned_text = clean_text(text)
    tokens = word_tokenize(cleaned_text)
    return tokens

# Load data from the SQLite database
engine = create_engine('sqlite:///../data/DisasterResponse.db')  # Path to the database
df = pd.read_sql_table('DisasterMessages', engine)  # Load the table into a DataFrame

# Load the trained model
model = joblib.load("../models/classifier.pkl")  # Path to the saved model

# Index webpage displays visuals and receives user input text for the model
@app.route('/')
@app.route('/index')

def index():
    """
    Render the main page with visualizations.
    """

    # Extract data for visualizations
    # Visualization 1: Distribution of Message Genres
    genre_counts = df.groupby('genre').count()['message']  # Count messages by genre
    genre_names = list(genre_counts.index)  # Get the genre names

    # Visualization 2: Distribution of Categories
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)  # Sum occurrences of each category
    category_names = list(category_counts.index)  # Get category names

    # Visualization 3: Number of Messages with Multiple Categories
    num_categories_per_message = df.iloc[:, 4:].sum(axis=1)  # Count how many categories each message has
    multi_label_counts = num_categories_per_message.value_counts().sort_index()  # Count frequency of messages by number of labels
    multi_label_numbers = list(multi_label_counts.index)  # Unique label counts
    multi_label_values = list(multi_label_counts.values)  # Their frequencies

    # Create visuals
    graphs = [
        # Bar chart for message genres
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
        # Bar chart for category distributions
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Category", 'tickangle': -45}
            }
        },
        # Bar chart for messages with multiple categories
        {
            'data': [
                Bar(
                    x=multi_label_numbers,
                    y=multi_label_values
                )
            ],
            'layout': {
                'title': 'Number of Messages with Multiple Categories',
                'yaxis': {'title': "Number of Messages"},
                'xaxis': {'title': "Number of Categories"}
            }
        }
    ]

    # Encode Plotly graphs in JSON format
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render the master.html template with the visualizations
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Web page that handles user queries and displays model results
@app.route('/go')
def go():
    """
    Handle user input query and display classification results.
    """
    # Get the user's query from the URL
    query = request.args.get('query', '')

    # Predict classifications using the trained model pipeline
    classification_labels = model.predict([query])[0]  # The pipeline preprocesses and predicts
    classification_results = dict(zip(df.columns[4:], classification_labels))  # Map labels to results

    # Render the go.html template with the query and classification results
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    """
    Run the Flask application on the specified host and port.
    """
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
