# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Create a dataset containing employee positions, their corresponding levels, and salaries, and store it in a pandas DataFrame.
2. Select Level as the independent feature (X) and Salary as the dependent target variable (y)
3. Initialize a Decision Tree Regressor and train it using the given feature and target data.
4. Use the trained model to predict salaries for existing and new levels, and visualize the actual data points along with the decision tree regression curve.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
data = {
    'Position': ['Business Analyst', 'Junior Consultant', 'Senior Consultant',
                 'Manager', 'Country Manager', 'Region Manager',
                 'Partner', 'Senior Partner', 'C-level', 'CEO'],
    'Level': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000]
}

df = pd.DataFrame(data)
X = df[['Level']]     # Feature (Level)
y = df['Salary']      # Target (Salary)
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X, y)

y_pred = regressor.predict(X)
print("Predicted salaries:", y_pred)

level = np.array([[6.5]])
predicted_salary = regressor.predict(level)
print(f"Predicted Salary for level {level[0][0]}: {predicted_salary[0]}")

X_grid = np.arange(min(X.values), max(X.values)+0.01, 0.01)  # High-resolution for smoother curve
X_grid = X_grid.reshape(-1, 1)

plt.scatter(X, y, color='red', label='Actual Salary')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Decision Tree Prediction')
plt.title('Decision Tree Regression: Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()
Developed by: Jesron Shawn C J
RegisterNumber:  25012933
*/
```

## Output:
<Figure size 640x480 with 1 Axes><img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/4b043aa0-eecd-4a57-8408-dfb61336b6e3" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
