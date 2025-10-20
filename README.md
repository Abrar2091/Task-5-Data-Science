# Task 5: Consumer Complaint Text Classification

This project performs multi-class text classification on consumer complaint data using Machine Learning and Deep Learning techniques.
It follows a complete end-to-end data science workflow, from exploratory analysis to final predictions.

# Dataset

The dataset used is from the Consumer Complaint Database
Each complaint is categorized into one of the following:
| Label | Category                           |
| ----- | ---------------------------------- |
| 0     | Credit reporting, repair, or other |
| 1     | Debt collection                    |
| 2     | Consumer loan                      |
| 3     | Mortgage                           |

# Project Workflow
# Explanatory Data Analysis (EDA) and Feature Engineering

Objective: Understand the structure, patterns, and relationships in the complaint data.

Steps:
Checked missing values and class distribution
Analyzed complaint text length and frequency
Extracted features such as word count, character count, stopword ratio, etc.

Visualized:
01_class_distribution.png
02_text_length_analysis.png
03_top_words_by_category.png
04_wordclouds_by_category.png
06_correlation_matrix.png
11_statistical_measures.png

Key Outputs:
Feature-rich dataset ready for model training.

# Text Pre-Processing
Objective: Clean and standardize text for effective model input.

Steps:
Lowercasing
Removing punctuation and special symbols
Stopword removal
Tokenization
Lemmatization using TextBlob
TF-IDF vectorization
Files Generated:
vectorizer.pkl
preprocessor.pkl
preprocessing_statistics.csv
Visualization: 07_preprocessing_comparison.png

# Selection of Multi-Classification Model
Objective: Train and select the best model for multi-class classification.

Models Implemented:
Logistic Regression
Random Forest
Support Vector Machine (SVM)
Artificial Neural Network (ANN)

Files Generated:
best_model.pkl
classification_report_Logistic_Regression.txt

Notebook: ANN.ipynb

# Comparison of Model Performance
Objective: Compare models based on key evaluation metrics.

Metrics Used:
Accuracy
Precision
Recall
F1-score
Training Time

Visualizations:
08_model_metrics_comparison.png
09_performance_radar_chart.png
10_time_vs_performance.png

CSV Outputs:
model_comparison_results.csv
per_class_metrics.csv

# Model Evaluation
Objective: Evaluate the final model performance using visual and statistical metrics.

Evaluation Methods:
Confusion Matrix
Cross Validation
F1-Score Heatmap
Per-Class Analysis

Visual Outputs:
12_cross_validation_results.png
13_confusion_matrices.png
15_f1_score_heatmap.png

Files:
confusion_matrix.csv
per_class_metrics.csv

# Architecture Overview

```
FEX
TASK 5 DATA SCIENCE EXAMPLE/
│
├── complaint_classification_outputs/
│   ├── 01_class_distribution.png
│   ├── 02_text_length_analysis.png
│   ├── 03_top_words_by_category.png
│   ├── 04_wordclouds_by_category.png
│   ├── 05_pattern_features.png
│   ├── 06_correlation_matrix.png
│   ├── 07_preprocessing_comparison.png
│   ├── 08_model_metrics_comparison.png
│   ├── 09_performance_radar_chart.png
│   ├── 10_time_vs_performance.png
│   ├── 11_statistical_measures.png
│   ├── 12_cross_validation_results.png
│   ├── 13_confusion_matrices.png
│   ├── 14_per_class_performance.png
│   ├── 15_f1_score_heatmap.png
│   ├── classification_report_Logistic_Regression.txt
│   ├── model_comparison_results.csv
│   ├── per_class_metrics.csv
│   ├── confusion_matrix.csv
│   ├── preprocessing_statistics.csv
│   ├── sample_predictions.csv
│   ├── preprocessor.pkl
│   ├── vectorizer.pkl
│   └── best_model.pkl
│
├── ANN.ipynb
├── complaints.csv.zip
└── README.md

```

# Setup & Execution
Requirements
Python 3.8 or higher
Jupyter Notebook or VS Code

Installation
Install required packages:
pip install pandas numpy matplotlib seaborn scikit-learn textblob tensorflow plotly

Running the Notebook
jupyter notebook ANN.ipynb
Or open ANN.ipynb in VS Code and click Run All Cells

# Deployment (Optional)

For deployment as an API:
Save model as .pkl files (already done).

Create a Flask app:

from flask import Flask, request, jsonify
import joblib
model = joblib.load('best_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return jsonify({'prediction': int(pred)})

if __name__ == '__main__':
    app.run(debug=True)


Run with:
python app.py


Test using Postman or curl:
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"text\":\"My mortgage account has an issue\"}"

