**Name:** DORNIPADU VENKATA SRI SAI

**Company:** CODTECH IT SOLUTIONS

**Id:** CT08EGV

**Domain:** Data science

**Duration:** December to January 2025

**Mentor:** Muzammil Ahmed


## Overview of the project
### Project:PREDICTIVE MODELING WITH LINEAR REGRESSION
## Objective
The project aims to:

Predict diabetes progression based on BMI using a simple linear regression model.
Evaluate and visualize the model’s performance.

## Key Steps in the Project:
Load the Dataset:
The load_diabetes() function from Scikit-learn provides a preloaded dataset about diabetes progression.
X is reduced to a single feature (BMI), and y is the target variable representing a quantitative measure of diabetes progression.
Prepare Data:
A pandas DataFrame is created to store the BMI feature and target values for easier inspection and manipulation.
The first few rows are printed to inspect the data structure.
Split Data:
The dataset is split into training and testing subsets using the train_test_split() function (80% training, 20% testing).
Random state ensures reproducibility.
Train the Model:
A LinearRegression model from Scikit-learn is used.
The model is trained on the training data (X_train, y_train).
Model Evaluation:
The model’s coefficient and intercept are printed to understand the regression equation.
Predictions (y_pred) are made on the testing data (X_test).
Performance metrics are calculated:
Mean Squared Error (MSE): Measures the average squared difference between actual and predicted values.
R-squared (R²): Indicates how well the model explains the variability in the data (closer to 1 is better).
Visualization:
A regression line is plotted to show the relationship between BMI and the diabetes progression target.
An "Actual vs. Predicted Values" scatter plot is provided to evaluate model performance visually.


### Technologies and tools used:

### **1. Programming Language:**
   - **Python:**
     - A powerful, widely-used programming language for machine learning, data analysis, and visualization.


### **2. Libraries and Frameworks:**
   - **NumPy:**
     - Provides support for efficient numerical computations and operations on arrays and matrices.
     - Used for slicing the data to select the BMI feature and reshaping the dataset.
   - **Pandas:**
     - A versatile library for data manipulation and analysis.
     - Used to create a DataFrame for easier inspection of the feature (`BMI`) and target variable (`y`).
   - **Matplotlib:**
     - A visualization library for creating static, animated, and interactive plots.
     - Used to plot scatter plots, regression lines, and "Actual vs Predicted" values.
   - **Scikit-learn:**
     - A machine learning library for Python.
     - Key components used:
       - `datasets.load_diabetes`: Loads the diabetes dataset.
       - `model_selection.train_test_split`: Splits the dataset into training and testing sets.
       - `linear_model.LinearRegression`: Implements the linear regression model.
       - `metrics.mean_squared_error` and `metrics.r2_score`: Compute evaluation metrics for model performance.

### **3. Data Source:**
   - **Diabetes Dataset:**
     A built-in dataset in Scikit-learn that contains ten baseline variables and a target variable indicating diabetes progression.

### **4. Development Environment:**
   - **Jupyter Notebook (or Python IDEs):**
     - Likely used for running Python code interactively and visualizing results inline.
   - **Anaconda (optional):**
     - Could be used to manage Python environments and libraries.

### **5. Statistical and Machine Learning Techniques:**
   - Linear regression modeling for predictive analysis.
 
