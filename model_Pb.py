import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

data = pd.read_csv('alldata.csv')
data = data.drop(['Unnamed: 0'],axis=1)

X = data.drop(['Pb, psi'], axis=1)
y = data['Pb, psi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pre defined tuned parameters
params = {
    'n_estimators': 100,
    'learning_rate': 0.2,
    'max_depth': 5,
    'min_samples_split':  10,
    'min_samples_leaf': 4,
    'subsample': 0.8,
    'random_state': 42
}

best_regressor = GradientBoostingRegressor(**params)
best_regressor.fit(X_train,y_train)
y_pred = best_regressor.predict(X_test)



## Metrics on test data
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", rmse)
score_lr=r2_score(y_test, y_pred)
print("R2 score", score_lr)
apre = np.mean(((y_pred-y_test)/y_test)*100)
print("Average Percentage Relative Error", apre)
aapre= np.mean(np.abs(y_pred-y_test)/y_test)*100
print("Average Absolute Percentage Relative Error", aapre)
aapre_check = mean_absolute_percentage_error(y_test,y_pred)
print("AAPRE :",aapre_check)

# Pickle the model
with open('model_Pb.pkl', 'wb') as f:
    pickle.dump(best_regressor, f)