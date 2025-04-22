# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Load Data
2. Split Data
3. Scale Data
4. Train Model
5. Evaluate Model
   

## Program:
```python
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by:   SANJAY M
RegisterNumber: 212223230187 
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X = data.data[:, :3]
Y = np.column_stack((data.target,data.data[:,6]))
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

sgd = SGDRegressor(max_iter = 1000, tol = 1e-3)

multi_output_regressor = MultiOutputRegressor(sgd)
multi_output_regressor.fit(X_train,Y_train)

#predict on test data
Y_pred = multi_output_regressor.predict(X_test)

Y_test = scaler_Y.inverse_transform(Y_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)

mse = mean_squared_error(Y_test,Y_pred)
print(f"Mean Squared Error: {mse}")

print("\nPredicted Values:",Y_pred[:5])
```

## Output:
![image](https://github.com/user-attachments/assets/a4ef809d-f59e-4fec-8976-908ab1e3ece0)
![image](https://github.com/user-attachments/assets/2564be76-7e2c-4897-9e2c-7d056e7a7088)
![image](https://github.com/user-attachments/assets/8c1ff8c4-f70e-4c4e-ba8a-2bfa4feba3ee)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
