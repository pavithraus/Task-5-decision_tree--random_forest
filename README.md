# Task-5-Decision Trees and Random Forests(Heart Disease Prediction using Tree-Based Models)

This project applies **Decision Tree** and **Random Forest** models for **Classification** and **Regression** tasks using the **Heart Disease dataset**. The goal is to explore and evaluate tree-based models to predict heart-related outcomes and understand key feature contributions.

---

##  Tools & Libraries

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  
- Graphviz

---

##  Dataset Used

**Dataset: [Heart_Disease.csv] **https://github.com/pavithraus/Task-5-decision_tree--random_forest/blob/main/Heart_Disease.csv

**Key Features:**

- age, sex, cp, chol, thalach, oldpeak, etc.
- target: Indicates presence (1) or absence (0) of heart disease.

---

##  Objectives

- Load and explore the Heart Disease dataset.
- Train tree-based models for:
  - **Classification**: Predict whether a patient has heart disease.
  - **Regression**: Predict cholesterol levels.
- Perform model evaluation and tuning using cross-validation.
- Visualize feature importance and model structure.
- Interpret and compare performance.

---

##  Key Tasks

### 1. Data Preprocessing

- Load and clean data using pandas.
- Normalize features with StandardScaler.
- Encode if needed (no categorical encoding required in this dataset).

---

### 2. Classification

**File:** heart_classification.py  
**Goal:** Predict heart disease (target)  
**Models:**  
- DecisionTreeClassifier  
- RandomForestClassifier

**Steps:**

- Split data into train/test sets
- Tune max_depth, n_estimators using GridSearchCV
- Evaluate using:
  - Accuracy
  - Classification report (Precision, Recall, F1)
- Visualize:
  - Tree (dt_class.png)
  - Feature importances

---

### 3. Regression

**File:** heart_regression.py  
**Goal:** Predict cholesterol level (chol)  
**Models:**  
- DecisionTreeRegressor  
- RandomForestRegressor

**Steps:**

- Use same split and scaling
- Tune hyperparameters via GridSearchCV
- Evaluate using:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score
- Visualize:
  - Tree (decision_tree)
  - Feature importances (bar plot)

---

##  Model Evaluation

### Classification (Random Forest Classifier)

| Metric              | Value                          |
|---------------------|-------------------------------|
| Accuracy            | High (~90% depending on split) |
| Precision/Recall/F1 | Provided in classification_report() |

### Regression (Random Forest Regressor)

| Metric  | Value (Example)         |
|---------|-------------------------|
| MAE     | ~8.5                    |
| MSE     | ~110                    |
| R²      | ~0.72                   |

> *Note: Actual values may vary slightly based on the random state and train/test split.*

---

##  Visual Outputs

- dt_class.png: Tree structure for classification.
- ![dt_class](https://github.com/user-attachments/assets/12613666-00c8-48a7-868c-7050e1420b04)

- Feature importance charts for both classifier.
- ![Random_forest(classifier)_feature Importance](https://github.com/user-attachments/assets/c7dd5a44-3c42-4308-b7be-1bd48d1a844e)


- desicion_tree.png: Tree structure for Regression.
- ![decision_tree](https://github.com/user-attachments/assets/77794986-eb5b-4444-8645-136a4be60daf)

- Feature importance charts for both regressor.
- ![Random_forest(regressor)_Feature_importance](https://github.com/user-attachments/assets/1a7b16e5-5585-4b56-8edc-7f10033b4c98)


---

##  Predicting on New Data

Models can be adapted to make predictions on new patient data by formatting it to match training features and applying the same preprocessing pipeline.

---

##  Learning Outcome

- Learned how tree-based models handle both classification and regression.
- Understood how to tune hyperparameters using GridSearchCV.
- Interpreted feature importances to understand model behavior.
- Compared simple (Decision Tree) vs ensemble (Random Forest) models.

---

##  About

This project was completed by **Pavithra** as part of an **Elevate Labs internship**.  
**Built using:** Google Colab  
**Learning references:** GeeksForGeeks, W3Schools, AI. 
Thank you **Elevate Labs**

---

