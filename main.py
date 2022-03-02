import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer, Normalizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras import layers, Input, optimizers, losses, metrics, regularizers
from keras.models import Model, Sequential

DATA_PATH = './data'
PARAMS_FILE = 'params80.csv'
RAW_FILE = 'raw80.csv'
EXCEL_FILE = '80mpa.xlsx'
RM_ST = 2022
PAD = True
METRICS = [tf.keras.metrics.RootMeanSquaredError(), 'mean_absolute_percentage_error']

all_data = pd.read_csv(os.path.join(DATA_PATH, PARAMS_FILE)).iloc[:19424]
train, _ = train_test_split(all_data, test_size=0.05, random_state = RM_ST)
val, test = train_test_split(_, test_size=0.2, random_state =  RM_ST)

y_columns = all_data.columns[:8]

Y_train, Y_val, Y_test = train[y_columns], val[y_columns], test[y_columns]

Y_scaler = PowerTransformer()
Y_scaler.fit(Y_train)

y_train = Y_scaler.transform(Y_train)
y_val = Y_scaler.transform(Y_val)
y_test = Y_scaler.transform(Y_test)

file = open(os.path.join(DATA_PATH, 'seqs.pkl'), 'rb')
SEQS = pickle.load(file)
file.close()

TEMPX = np.concatenate([SEQS[k] for k in train.index])

X_scaler = RobustScaler()
X_scaler.fit(TEMPX)

x_train = np.array([X_scaler.transform(SEQS[k]) for k in train.index], dtype='object')
x_val = np.array([X_scaler.transform(SEQS[k]) for k in val.index], dtype= 'object')
x_test = np.array([X_scaler.transform(SEQS[k]) for k in test.index], dtype= 'object')

if PAD:
    x_train = pad_sequences(x_train, maxlen = 1000, padding = 'post', value = -999, dtype='float64')
    x_val = pad_sequences(x_val, maxlen = 1000, padding = 'post', value = -999, dtype='float64')
    x_test = pad_sequences(x_test, maxlen = 1000, padding = 'post', value = -999, dtype='float64')

# maxlen = 1000, max(map(len, SEQS.values()))

#%%
# =============================================================================
# Model Generation
# =============================================================================

opt = tf.keras.optimizers.Adam(learning_rate=0.05)

# Create separate inputs for time series and constants
input_ = Input(shape=x_train.shape[1:])

# Feed time_input through Masking and LSTM layers
mask = layers.Masking(mask_value=-999)(input_)
temp_vector = layers.LSTM(50, kernel_regularizer=regularizers.l1_l2(9.301e-7),
                          recurrent_regularizer=regularizers.l1_l2(0.004248),
                          bias_regularizer=regularizers.l1_l2(8.7213e-12))(mask)

for i in range(3):
    temp_vector = layers.Dense(40, kernel_regularizer=regularizers.l1_l2(0.0044339),
                            bias_regularizer=regularizers.l1_l2(0.01275), activation='relu')(temp_vector)

params = layers.Dense(8)(temp_vector)

# Instantiate model
model = Model(inputs=[input_], outputs=[params])

# Compile
model.compile(loss='huber_loss', optimizer=opt, metrics=METRICS)

model.fit(x_train, y_train, epochs=40, batch_size=100, verbose = 1)


y_true0 = Y_scaler.inverse_transform(y_train)
y_pred0 = Y_scaler.inverse_transform(model.predict(x_train))
                                     
y_true1 = Y_scaler.inverse_transform(y_val)
y_pred1 = Y_scaler.inverse_transform(model.predict(x_val))

y_true2 = Y_scaler.inverse_transform(y_test)
y_pred2 = Y_scaler.inverse_transform(model.predict(x_test))

err0 = abs(y_true0-y_pred0)/y_true0*100
err1 = abs(y_true1-y_pred1)/y_true1*100
err2 = abs(y_true2-y_pred2)/y_true2*100

np.savez('results/errors', err0 = err0, err1 = err1, err2 = err2)
np.savez('results/train', y0 = y_true0, y1 = y_pred0)
np.savez('results/val', y0 = y_true1, y1 = y_pred1)
np.savez('results/train', y0 = y_true2, y1 = y_pred2)
