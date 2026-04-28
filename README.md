# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation: Load the California housing dataset, extract features (first three columns) and targets (target variable and sixth column), and split the data into training and testing sets.

2.Data Scaling: Standardize the feature and target data using StandardScaler to enhance model performance.

3.Model Training: Create a multi-output regression model with SGDRegressor and fit it to the training data.

4.Prediction and Evaluation: Predict values for the test set using the trained model, calculate the mean squared error, and print the predictions along with the squared error.


## Program:
```
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("house.csv")
#print(data.columns)
data.columns = data.columns.str.strip()

X = data[['Size', 'Bedrooms']]


y_price = data['Price']
y_occ = data['Occupants']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

price_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
occ_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)

price_model.fit(X_scaled, y_price)
occ_model.fit(X_scaled, y_occ)

size = float(input("Enter house size: "))
bed = int(input("Enter number of bedrooms: "))

new_data = scaler.transform([[size, bed]])

pred_price = price_model.predict(new_data)
pred_occ = occ_model.predict(new_data)

print("Predicted Price:", pred_price[0])
print("Predicted Occupants:", round(pred_occ[0]))
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: 
RegisterNumber:  
*/
```

## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)
<img width="924" height="804" alt="Screenshot 2026-04-27 142743" src="https://github.com/user-attachments/assets/2f8de9dd-d1ca-484f-99a5-9ae5818b4873" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
