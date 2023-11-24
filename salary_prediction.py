"""
Author: Sahil Sanjay Gunjal.
Python Version: 3.10
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def read_data():
    """
    This is used to Read the data.
    :return: dataframe
    """
    df = pd.read_csv('salary_data.csv')
    return df


def visualize_data(df):
    """
    This is used to visualize the data.
    :param df: Dataframe
    :return: None
    """
    plt.scatter(df['Age'], df['Salary'])
    plt.xlabel('Age(years)')
    plt.ylabel('Salary($)')
    plt.show()


def predict_values(x_values, w, b):
    """
    This function predicts the values and return it.
    :param x_values: All X_values: Ages
    :param w: Parameter 1 for training the model(weights)
    :param b: Parameter 2 for training the model.(bias)
    :return: Predicted values
    """
    return np.dot(w, x_values) + b


def calculate_cost(predicted_salary, y_values):
    """
    This function is used to calculate the cost using mean square error (MSE).
    :param predicted_salary: predicted salary
    :param y_values: Actual salary
    :return: Cost
    """
    m = len(predicted_salary)
    return 1/(2*m) * np.sum((predicted_salary - y_values)**2)


def gradient_descent(x_values, y_values, w, b, learning_alpha, epochs):
    """
    Perform gradient descent to optimize the parameters (weights and bias) of a linear regression model.

    :param x_values: Input feature values.
    :param y_values: Actual output values.
    :param w: Initial weight.
    :param b: Initial bias.
    :param learning_alpha: Learning rate, determining the step size in the optimization process.
    :param epochs: Number of iterations for gradient descent.
    :return: Updated weights (w), bias (b), list of epochs, and corresponding cost values.
    """
    total_epochs = []
    cost = []
    for i in range(epochs):
        total_epochs.append(i+1)

        # Predict salary using the current weights and bias
        predicted_salary = predict_values(x_values, w, b)

        # Calculate the current cost using mean squared error
        curr_cost = calculate_cost(predicted_salary, y_values)
        cost.append(curr_cost)
        m = len(y_values)

        # Compute the gradients with respect to weights (dw) and bias (db)
        dw = (1/m) * np.sum((predicted_salary - y_values) * x_values)
        db = (1/m) * np.sum(predicted_salary - y_values)

        # Update weights and bias using the gradients and the learning rate
        w -= learning_alpha * dw
        b -= learning_alpha * db

    return w, b, total_epochs, cost


def train_model(w, b, learning_alpha, epochs ,train_df):
    """
    Train a linear regression model using gradient descent.

    :param w: Initial weight.
    :param b: Initial bias.
    :param learning_alpha: Learning rate, determining the step size in the optimization process.
    :param epochs: Number of iterations for gradient descent.
    :param train_df: DataFrame containing training data with 'Age' and 'Salary' columns.
    :return: Updated weights (final_w), bias (final_b), list of epochs, and corresponding cost values.
    """
    x_values = train_df['Age']
    y_values = train_df['Salary']

    # Use gradient descent to optimize weights and bias
    final_w, final_b, total_epochs, cost = gradient_descent(x_values, y_values, w, b, learning_alpha, epochs)

    return final_w, final_b, total_epochs, cost


def find_parameters(train_df):
    """
    Find optimal parameters for a linear regression model using training data.

    :param train_df: Training DataFrame with 'Age' and 'Salary' columns.
    :return: Updated weights (final_w), bias (final_b), learning rate, list of epochs, and corresponding cost values.
    """

    # let's initialize the parameters for w and b
    w = 0.001
    b = 0.01
    learning_alpha = 0.001
    epochs = 10000

    # Feature scaling (normalize age and salary)
    train_df['Age'] = (train_df['Age'] - train_df['Age'].mean()) / train_df['Age'].std()
    train_df['Salary'] = (train_df['Salary'] - train_df['Salary'].mean()) / train_df['Salary'].std()

    final_w, final_b, total_epochs, cost = train_model(w, b, learning_alpha, epochs, train_df)
    return final_w, final_b, learning_alpha, total_epochs, cost


def predict_salary_on_test(test_df, final_w, final_b):
    """
    Predict salaries on the test data and visualize the model's performance.

    :param test_df: Test DataFrame with 'Age' and 'Salary' columns.
    :param final_w: Optimized weights obtained from training.
    :param final_b: Optimized bias obtained from training.
    :return: None
    """
    # Feature scaling for test data
    test_df['Age'] = (test_df['Age'] - test_df['Age'].mean()) / test_df['Age'].std()

    # Reset the index to avoid index-related issues
    test_df.reset_index(drop=True, inplace=True)

    x_values = test_df['Age']
    y_values = test_df['Salary']

    # Make predictions
    test_predicted = predict_values(x_values, final_w, final_b)

    # Rescale the predictions to the original salary range
    test_predicted_rescaled = test_predicted * test_df['Salary'].std() + test_df['Salary'].mean()

    # Print predicted and actual values
    for i in range(len(test_predicted_rescaled)):
        print(f'predicted: {test_predicted_rescaled[i]}, Actual: {y_values[i]}')

    # Calculate and print the average difference between predicted and actual values
    total_sum = 0
    for i in range(len(test_predicted_rescaled)):
        total_sum += abs(y_values[i] - test_predicted_rescaled[i])

    print("Average Difference:", total_sum/len(test_predicted_rescaled))

    # Visualize the model's fit on the test data
    fit_curve_plot(test_df,test_predicted_rescaled)


def epochs_vs_cost(epochs, cost):
    """
    This method plot the epoch vs cost graph.
    :param epochs: Total epochs
    :param cost: Cost per epoch
    :return: None
    """
    plt.plot(epochs, cost)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Cost vs Epoch')
    plt.show()


def fit_curve_plot(test_df,test_predicted_rescaled):
    """
    This method is used to plot how well model fits the testing data.
    :param test_df: Test data
    :param test_predicted_rescaled: Predicted values
    :return:None
    """
    plt.scatter(test_df['Age'], test_df['Salary'], label='Actual Data', color='blue')
    plt.plot(test_df['Age'], test_predicted_rescaled, label='Model Predictions', color='red')
    plt.xlabel('Age')
    plt.ylabel('Salary')
    plt.title('Linear Regression Model')
    plt.legend()
    plt.show()


def main():
    """
    This method is used to call the different functions to read the data, train model and predict salary.
    :return: None
    """
    df = read_data()
    # visualizing the data
    # visualize_data(df)

    # Separate the data into training and testing
    train_df, test_df = train_test_split(df, test_size=50, random_state=42)

    # finding the best parameters
    final_w, final_b, learning_alpha, total_epochs, cost = find_parameters(train_df)

    # Predict the salary on the test data
    predict_salary_on_test(test_df, final_w, final_b)

    # Plot the graph epoch vs cost
    epochs_vs_cost(total_epochs, cost)


# Start of the code
if __name__ == '__main__':
    main()