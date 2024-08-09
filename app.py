import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'car_data.csv'
data = pd.read_csv(file_path)

data = pd.get_dummies(data, columns=['make', 'model'])

features = [col for col in data.columns if col != 'price']
X = data[features]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()

def get_user_input(features):
    make = input("Enter the car make: ")
    model = input("Enter the car model: ")
    year = int(input("Enter the car year: "))
    mileage = float(input("Enter the car mileage: "))
    
    dummy_columns = [col for col in features if col.startswith('make_') or col.startswith('model_')]
    
    user_data = {'year': year, 'mileage': mileage}
    
    for col in dummy_columns:
        user_data[col] = 0
    
    make_column = f'make_{make}'
    model_column = f'model_{model}'
    
    if make_column in user_data:
        user_data[make_column] = 1
    if model_column in user_data:
        user_data[model_column] = 1
    
    return [user_data.get(col, 0) for col in features]

new_car = get_user_input(features)

new_car = np.array([new_car])
new_car_scaled = scaler.transform(new_car)

predicted_price = model.predict(new_car_scaled)

def format_number_with_dots(number):
    integer_part = int(round(number)) 
    formatted_integer_part = ''
    
    for i, digit in enumerate(reversed(str(integer_part))):
        if i > 0 and i % 3 == 0:
            formatted_integer_part = '.' + formatted_integer_part
        formatted_integer_part = digit + formatted_integer_part
    
    return formatted_integer_part

formatted_price = format_number_with_dots(predicted_price[0])
print("Predicted Price:", formatted_price)
