# Customer Churn Prediction Pipeline

## Objective

Build a reusable and production-ready machine learning pipeline to predict customer churn using the Telco Customer Churn Dataset.

## Methodology / Approach

1. Loaded and cleaned the dataset.
2. Converted the target variable (Churn) into binary values (Yes = 1, No = 0).
3. Split the data into training and testing sets.
4. Applied preprocessing using Scikit-learn Pipeline and ColumnTransformer:

   * StandardScaler for numerical features
   * OneHotEncoder for categorical features
5. Trained Logistic Regression and Random Forest models.
6. Used GridSearchCV with 5-fold cross-validation for hyperparameter tuning.
7. Selected the best-performing model and exported the complete pipeline using Joblib.

## Key Results / Observations

* Successfully built an end-to-end ML pipeline for churn prediction.
* Automated preprocessing reduced manual feature engineering.
* GridSearchCV helped identify optimal model parameters.
* The complete pipeline was saved as a `.pkl` file for future deployment and inference.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib
