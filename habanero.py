import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

DATA_PATH = './data'
PARAMS_FILE = 'params80.csv'
RAW_FILE = 'raw80.csv'
EXCEL_FILE = '80mpa.xlsx'

df = pd.read_csv(os.path.join(DATA_PATH, PARAMS_FILE)).iloc[:19424]

coeffs = [f'c_{i}' for i in range(16)]
coeffs = df[coeffs]

p = [np.poly1d(coeffs.loc[i]) for i in range(len(df))]

seqs = {}

for i in range(len(df)):
    x_steps = np.arange(0, df.x_end.loc[i], 10)
    seqs[i] = p[i](x_steps).reshape(-1, 1)

file = open(os.path.join(DATA_PATH, 'seqs.pkl'), 'wb')
pickle.dump(seqs, file)
file.close()