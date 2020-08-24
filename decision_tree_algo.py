# -*- coding: utf-8 -*-

import pandas as pd
import pickle
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
# y = y.reshape(len(y), 1)
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)
# y_pred = regressor.predict([[6.5]])
pickle.dump(regressor, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[6.5]]))