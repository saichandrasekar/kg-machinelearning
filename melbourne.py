import pandas as pd
from sklearn.tree import DecisionTreeRegressor

def run_melbourne():
    melbourne_file_path = "S:/sai-py/workspace/kg-machinelearning/datasets/archive/melb_data.csv"
    melbourne_data = pd.read_csv(melbourne_file_path)
#    print(melbourne_data.describe())
#    print(melbourne_data.head)
    print(melbourne_data.columns)
    first_row = melbourne_data.describe()['Rooms']['mean']
    print(first_row)

    melbourne_data.dropna(axis=0)

    price = melbourne_data.Price
    print(price)

    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

    X = melbourne_data[melbourne_features]

    print(X.describe())
    print(X.head())

    melbourne_model = DecisionTreeRegressor(random_state=1)

    melbourne_model.fit(X, price)


    print("Making predictions for the following 5 houses:")
    print(X.head())
    print("The predictions are")
    print(melbourne_model.predict(X.head()))