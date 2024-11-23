import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge them.
    
    Args:
    messages_filepath: str. Filepath for the messages CSV file.
    categories_filepath: str. Filepath for the categories CSV file.
    
    Returns:
    df: DataFrame. Merged dataset containing messages and their categories.
    """
    # Load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets on 'id'
    df = messages.merge(categories, on='id')
    
    return df

def clean_text(text):
    """
    Clean text data by applying NLP techniques.
    
    Args:
    text: str. Raw text data.
    
    Returns:
    str. Cleaned text data.
    """
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text into words
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join words back into a single string
    return ' '.join(words)

def clean_data(df):
    """
    Clean the merged DataFrame by splitting categories into separate columns,
    converting values to binary, removing duplicates, and cleaning the text column.
    
    Args:
    df: DataFrame. Merged dataset containing messages and their categories.
    
    Returns:
    df: DataFrame. Cleaned dataset.
    """
    # Split categories into separate columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    
    # Convert category values to binary (0 or 1)
    for column in categories:
        categories[column] = categories[column].str.split('-').str[1].astype(int)
    
    # Replace 'categories' column in the original dataframe
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Clean the message column
    df['message'] = df['message'].apply(clean_text)
    
    return df

def save_data(df, database_filename):
    """
    Save the cleaned dataset into an SQLite database.
    
    Args:
    df: DataFrame. Cleaned dataset.
    database_filename: str. Filepath for the SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterMessages', engine, index=False, if_exists='replace')

def main():
    """
    Main function to execute the ETL pipeline: load data, clean data, and save data.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        
        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as well as '
              'the filepath of the database to save the cleaned data to as the third argument. '
              '\n\nExample: python process_data.py disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
