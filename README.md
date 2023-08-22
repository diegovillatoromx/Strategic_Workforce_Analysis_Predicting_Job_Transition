# Predictive Employee Intent Analysis: Identifying Future Job Seekers and Company Devotees using Demographics and Experience Data

## Business Objective
In the realm of Big Data and Data Science, a company specializes in recruiting data scientists from those who successfully complete their training courses. With a large pool of enrolled individuals, the company aims to differentiate candidates who genuinely intend to join their workforce post-training, from those who are actively seeking new job opportunities. This distinction holds the key to reducing costs, enhancing training quality, and optimizing course planning. Leveraging demographic, educational, and experiential data gathered during candidate enrollment, the task at hand is to develop predictive models that ascertain the likelihood of a candidate either seeking alternative employment or committing to the company. This analysis not only informs strategic human resource decisions, but also provides insights into the factors influencing employee decisions concerning their future career paths.

## Architecture Diagram
<img src="architecture_diagram.png" >

## Aim
This project is designed to understand the factors that lead a person to leave their current job for HR research too. By model(s) that uses the current credentials,demographics,experience data you will predict the probability of a candidate to look for a new job or will work for the company, as well as interpreting affected factors on employee decision.

## Approach
- Importing the required libraries and reading the dataset.
- Inspecting and cleaning up the data
- Perform data encoding on categorical variables
- Exploratory Data Analysis (EDA)
  - Data Visualization
- Feature Engineering
  - Dropping of unwanted columns
- Model Building
  - Using the statsmodel library
- Model Building
  - Performing train test split
  - Logistic Regression Model
- Model Validation (predictions)
  - Accuracy score
  - Confusion matrix
  - ROC and AUC
  - Recall score
  - Precision score
  - F1-score
- Handling the unbalanced data
  - With balanced weights
  - Random weights
  - Adjusting imbalanced data
  - Using SMOTE
- Feature Selection
  - Barrier threshold selection
  - RFE method
- Save the model in the form of a pickle file

## Tech Stack
- Language
   - Python
- Libraries
  - numpy, pandas, matplotlib, seaborn, sklearn, pickle, imblearn,
statsmodel 

## Data Description
The CSV consists of around 2000 rows and 16 columns in the [dataset](https://github.com/diegovillatoromx/Strategic_Workforce_Analysis_Predicting_Job_Transition/blob/main/input/DS_Job_Change_Data.csv).
### Features:
- enrollee_id : Unique ID for candidate
- city: City code
- city_ development _index : Developement index of the city (scaled)
- gender: Gender of candidate
- relevent_experience: Relevant experience of candidate
- enrolled_university: Type of University course enrolled if any
- education_level: Education level of candidate
- major_discipline :Education major discipline of candidate
- experience: Candidate total experience in years
- company_size: No of employees in current employer's company
- company_type : Type of current employer
- last_new_job: Difference in years between previous job and current job
- training_hours: training hours completed
- target: 0 – Not looking for job change, 1 – Looking for a job change
    
## Modular Code Overview

```
  Data
    |_data_regression.csv

  src
    |_Engine.py
    |_ML_pipeline
              |_encoding.py
              |_evaluate_metrics.py
              |_feature_engg.py
              |_imbalanced_data.py
              |_ml_model.py
              |_stats_model.py
              |_rescale_variables.py
              |_scaler.py
              |_train_model.py
              |_utils.py

  
  hiring_data_scientists.ipynb

  output
    |_adjusted_model.pkl
    |_balanced_model1.pkl
    |_balanced_model2.pkl
    |_log_ROC.pkl
    |_model_rfe_feat.pkl
    |_model_stats.pkl
    |_model_var_feat.pkl
    |_model1.pkl
    |_smote_model.pkl
```
