# Linear Regression Salary Prediction

## Overview
This repository contains Python code for creating a synthetic dataset of salaries based on age and implementing a simple linear regression model to predict salaries. The code includes data generation, model training using gradient descent, and visualization of the model's performance.

## Files
1. **data_generation.py**: Generates a synthetic dataset of salaries based on randomly generated ages. The dataset is saved in a CSV file named 'salary_data.csv'.
2. **linear_regression_prediction.py**: Reads the generated dataset, trains a linear regression model, and predicts salaries on a test set. It includes functions for visualizing data, implementing gradient descent, and plotting the model's fit.
3. **salary_data.csv**: The generated dataset used for training and testing the linear regression model.

## How to Use
1. Run **data_generation.py** to create the synthetic dataset.
2. Execute **linear_regression_prediction.py** to train the model and make predictions on a test set.
3. The code includes visualizations such as scatter plots, epoch vs. cost graphs, and a curve plot to show the model's fit on the test data.

## Dependencies
- Python 3.10
- Required libraries: NumPy, Pandas, Matplotlib, scikit-learn

## Notes
- The linear regression model is trained using gradient descent, and the learning rate and number of epochs can be adjusted for optimization.
- The code provides insights into the model's performance by visualizing the data and plotting the cost vs. epoch graph.

Feel free to explore, modify, and use the code for your own projects!

---

You can customize this description based on additional details about your project, such as specific use cases, performance metrics, or any unique features you've implemented.
