import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from tabpy.tabpy_tools.client import Client

data = pd.read_csv('train.csv')
features = ['TotalCases', 'TotalRecovered', 'TotalDeaths', 'NewDeaths', 'NewRecovered']
target = ['NewCases']

X = data[features]
Y = data[target]

train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size=0.2, random_state=0)

def Linear(_arg1, _arg2, _arg3, _arg4, _arg5):
    X = pd.DataFrame({
        'TotalCases': _arg1,
        'TotalRecovered': _arg2,
        'TotalDeaths': _arg3,
        'NewDeaths': _arg4,
        'NewRecovered': _arg5,
    })
    model = LinearRegression()
    model.fit(train_X, train_Y)
    
    return model.predict(X).reshape(-1).astype(int).tolist()

def RegressionTree(_arg1, _arg2, _arg3, _arg4, _arg5):
    X = pd.DataFrame({
        'TotalCases': _arg1,
        'TotalRecovered': _arg2,
        'TotalDeaths': _arg3,
        'NewDeaths': _arg4,
        'NewRecovered': _arg5,
    })
    model = DecisionTreeRegressor(random_state=0)
    model.fit(train_X, train_Y)

    return model.predict(X).reshape(-1).astype(int).tolist()

client = Client('http://localhost:9004/')
client.deploy('New_Cases_Covid_prediction_Linear', 
              Linear,
              'Returns prediction of New cases for Covid-19.'
              , override = True)
client.deploy('New_Cases_Covid_prediction_Tree',
              RegressionTree,
              'Returns prediction of New cases for Covid-19.'
              , override = True)