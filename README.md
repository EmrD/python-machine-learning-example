# Car Price Prediction Model

This project involves a linear regression model to predict car prices based on various features such as make, model, year, and mileage. The model is trained on a dataset of car listings and can be used to estimate the price of a car given its attributes.

## Project Structure

1. **`app.py`**: Python script containing the code for data processing, model training, and prediction.
2. **`car_data.csv`**: Sample dataset used for training the model.

## Dataset

The dataset `car_data.csv` contains the following columns:
- `make`: The manufacturer of the car (e.g., Toyota, Honda).
- `model`: The specific model of the car (e.g., Corolla, Civic).
- `year`: The manufacturing year of the car.
- `mileage`: The mileage of the car in kilometers.
- `price`: The price of the car in USD.

### Sample Data

```csv
make,model,year,mileage,price
Toyota,Corolla,2015,50000,15000
Honda,Civic,2018,30000,20000
Ford,Focus,2017,40000,18000
Chevrolet,Malibu,2016,60000,16000
Nissan,Altima,2019,25000,22000
Hyundai,Elantra,2018,35000,19000
BMW,3 Series,2019,20000,25000
Audi,A4,2020,15000,28000
