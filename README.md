# Disaster Response Pipeline Project
This is a Udacity project in the Data Scientist Nanodegree.
You can check repository here:
https://github.com/TranDat24318791998/Udacity_Disaster_Response_Pipeline

## **Project Structure**

### **Folders and Files**
- **app/**
  - **templates/**
    - `master.html`: Template for the main app interface (visualizations and input form).
    - `go.html`: Template for displaying classification results.
  - `run.py`: Python script to launch the web application.
  
- **data/**
  - `disaster_messages.csv`: Contains raw disaster messages.
  - `disaster_categories.csv`: Contains labels for the messages.
  - `DisasterResponse.db`: SQLite database with the cleaned dataset.
  - `process_data.py`: Python script for ETL pipeline (Extract, Transform, Load).

- **models/**
  - `train_classifier.py`: Python script for machine learning pipeline (training the model).
  - `classifier.pkl`: Pickle file with the trained machine learning model.

---

## Data Processing
1. **Load Data**:
   - Merge `disaster_messages.csv` and `disaster_categories.csv` on `id`.

2. **Clean Data**:
   - Process text: lowercase, remove URLs/punctuation, tokenize, remove stopwords, and lemmatize.
   - Split `categories` into binary columns.
   - Remove duplicates.

3. **Save Data**:
   - Store cleaned data into an SQLite database (`DisasterResponse.db`).

---

## Model Training
1. **Load Data**:
   - Load cleaned data from the SQLite database.
   - Use `message` as features (`X`) and categories as targets (`Y`).

2. **Build Pipeline**:
   - Preprocess text using `CountVectorizer` and `TfidfTransformer`.
   - Train a `MultiOutputClassifier` with `LGBMClassifier`.

3. **Train and Evaluate**:
   - Split data into training (80%) and testing (20%).
   - Evaluate model using precision, recall, and F1-score for each category.

4. **Save Model**:
   - Save trained model as a pickle file (`classifier.pkl`).


## **Requirements**

Make sure to install the required Python packages before running the project. The dependencies are listed in the `requirements.txt` file. Install them with the following command:
`pip install -r requirements.txt`

## **Instructions**:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
