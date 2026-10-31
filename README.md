# fake-job-detection system 

## Project Description
"An AI/ML-based application that detects fraudulent job postings by analyzing job descriptions and classifying them as genuine or fake, helping users avoid job scams."

## Technologies Used
- Python, Machine learning, Pandas, Scikit-learn, Streamlit, TF-IDF Vectorizer, Logistic Regression.

## Project Files
* app.py – Runs the web application.
* train.py – Trains the machine learning model.
* model.pkl – Saved trained model.
* vectorizer.pkl – Saved TF-IDF vectorizer.

## How It Works
- Enter or paste a job description.
- Click the Predict button.
- The application analyses the text using the trained ML model.
- Displays the result as:
    * Fraudulent Job Posting
    * Legitimate Job Posting

## How to Run
- Install the required libraries:
  ```bash
  pip install streamlit pandas scikit-learn joblib
  ```
- Run the application:
  ```bash
  streamlit run app.py
  ```
## Project Goal
- To help job seekers identify potentially fraudulent job postings using Machine Learning.

