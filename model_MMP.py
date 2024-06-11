import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("master.csv")
#Split into features and target (Price)
X = data.drop(['H2S','CO2','N2','C1','C2','C3','C4','C5','C6','C7+','MMP(psi)'], axis = 1)
y = data['MMP(psi)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)
X_train.shape,X_test.shape, y_train.shape,y_test.shape

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn import metrics
import numpy as np
def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    aare = metrics.mean_absolute_percentage_error(true,predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('AARE', aare)

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
model = Sequential()
model.add(Dense(12, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal' ,activation='linear'))
model.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(0.009), metrics=['mse'])
model.summary()
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs =300)

test_pred = model.predict(X_test_scaled)
train_pred = model.predict(X_train_scaled)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

joblib.dump(scaler,'scaler.pkl')
model.save('model_MMP.h5')