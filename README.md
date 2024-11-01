# Geonomic-Predictors-OF-Drug-Sensivity-In-Cancer-With-Help-Of-Machine-Learning
## Overview

This project aims to explore and compare the performance of different regression algorithms for predictive analysis on a given dataset. The models examined include **Linear Regression**, **Ridge Regression**, and **Stochastic Gradient Descent (SGD) Regression**. The performance of each model was evaluated using multiple metrics, such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² Score, to determine their suitability for this task. This analysis provides insight into the effectiveness of regularization, the impact of hyperparameters, and the importance of model selection.

## Dataset

The dataset used for this project includes several features that correlate with the target variable, enabling us to assess the models' ability to capture these relationships and make accurate predictions. The dataset was divided into training and test sets to evaluate the generalization capability of each model on unseen data.

## Methodology

### Step 1: Import Libraries and Initialize Tools
The first step involved importing essential Python libraries required for the project, including:
- **Scikit-Learn** for machine learning model creation and evaluation.
- **NumPy** for numerical operations.
- **Pipelines** to streamline the process of scaling and model training.
  
### Step 2: Data Preprocessing
Data preprocessing was crucial to ensure the data was ready for modeling. This included:
- **Scaling**: We used `StandardScaler` to standardize the feature values. Standardization transforms data to have a mean of 0 and a standard deviation of 1, which is essential for models like Ridge and SGD, as it helps them converge more quickly and avoid issues due to large feature magnitudes.

### Step 3: Model Initialization and Training

We implemented three models: **Linear Regression**, **Ridge Regression**, and **SGD Regression**.

1. **Linear Regression**: We initialized and trained a simple Linear Regression model to serve as a baseline.
   - Using `make_pipeline`, we combined `StandardScaler` with `LinearRegression`, creating a streamlined pipeline to preprocess the data and fit the model simultaneously.
   - The model was trained using the `fit` method with `X_train` and `y_train`, where it learned relationships between features and the target variable.

2. **Ridge Regression**: Next, we tested Ridge Regression, which adds L2 regularization to Linear Regression to reduce overfitting.
   - Using `Ridge` within a `StandardScaler` pipeline, the model adjusted for potentially overfitting relationships by penalizing large coefficients.
   - Ridge Regression allows the model to generalize better, especially when dealing with multicollinearity or redundant features.

3. **SGD Regression**: Lastly, we tested Stochastic Gradient Descent (SGD) Regression, which uses an iterative optimization algorithm suitable for large datasets.
   - The model was initialized with `SGDRegressor`, configured to use 1000 maximum iterations and a tolerance of `1e-3` for convergence.
   - `SGDRegressor` learns in a stochastic, or batch-based, manner, updating model weights on each subset of data, which should make it faster for large datasets. However, we noticed convergence issues, as reflected in the poor performance metrics.

### Step 4: Model Evaluation

For each model, we calculated the following metrics to assess performance:

- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values, reflecting prediction quality.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, RMSE provides an error measure in the same units as the target variable.
- **Mean Absolute Error (MAE)**: The average absolute difference between predicted and actual values, offering insight into the typical error magnitude.
- **R² Score**: Indicates the proportion of variance explained by the model. An R² close to 1 indicates that the model captures most of the variance in the data.

Each model’s performance metrics were printed to compare results. Below are the outcomes:

### Results

1. **Linear Regression**:
   - **MSE**: 0.0138
   - **RMSE**: 0.1176
   - **MAE**: 0.0764
   - **R² Score**: 0.9862
   - *Description*: Linear Regression performed well, with an R² Score of 98.6%, indicating that it captures a high proportion of the variance in the target variable. Both MSE and MAE are relatively low, suggesting the model's predictions are close to actual values.

2. **Ridge Regression**:
   - **MSE**: 0.0137
   - **RMSE**: 0.1172
   - **MAE**: 0.0759
   - **R² Score**: 0.9863
   - *Description*: Ridge Regression slightly outperformed Linear Regression, with a marginally better R² Score and lower MSE and MAE values. The regularization in Ridge Regression helped reduce overfitting risk while preserving accuracy, making it a strong choice for generalization.

3. **SGD Regression**:
   - **MSE**: 1.47e+21
   - **RMSE**: 3.84e+10
   - **MAE**: 4.77e+8
   - **R² Score**: -1.47e+21
   - *Description*: The SGD Regression model exhibited extremely poor performance, with extremely high error values and a negative R² Score, indicating that it failed to converge. These results suggest that SGD Regression either requires more tuning or is less suitable for this particular dataset and setup.

### Conclusion

Both **Linear Regression** and **Ridge Regression** models delivered excellent results with high R² Scores and low error values, making them effective options for this predictive task. **Ridge Regression** slightly outperformed Linear Regression, benefiting from regularization to handle potential overfitting. On the other hand, **SGD Regression** faced convergence issues, resulting in a large error and a negative R² Score, which may require further tuning or a different configuration to be viable for this dataset.
