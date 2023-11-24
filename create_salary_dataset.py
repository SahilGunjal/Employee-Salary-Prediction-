"""
Author: Sahil Sanjay Gunjal.
Version: Python 3.10
Problem: This code is used to create the random dataset of salaries, as per their age.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_dataset():
    """
    This method is used to create the random dataset which is then saved in the .csv file.
    :return: None
    """
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Generate 500 random ages between 22 and 65
    ages = np.random.randint(22, 66, size=500)

    slope = 500
    intercept = 20000
    noise = np.random.normal(0, 1000, size=500)

    salaries = slope * ages + intercept + noise

    salaries = np.round(salaries, decimals=2)
    df = pd.DataFrame({'Age': ages, 'Salary': salaries})
    df.to_csv('salary_data.csv',index=False)
    return ages, salaries


def visualize_data(ages, salaries):
    """
    Plot to visualize the data.
    :param ages: Ages list
    :param salaries: Salaries list
    :return: None
    """
    plt.scatter(ages, salaries, alpha=0.5)
    plt.title('Generated Data for Univariate Linear Regression')
    plt.xlabel('Age (years)')
    plt.ylabel('Salary ($)')
    plt.show()


def main():
    """
    This method is used to call the different functions which create the dataset and visualize it.
    :return:
    """
    ages, salaries = create_dataset()
    visualize_data(ages, salaries)


# Start of the code
if __name__ == '__main__':
    main()