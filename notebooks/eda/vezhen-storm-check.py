# To run first run (using qrf-venv) python -m src.eagles_vezhen.data_preparation_eagles  

# Trimmed by Claude (Anthropic): removed the unused Stormglass request code,
# lat/lon and loop debugging, unused imports and variables, and notebook markers;
# figures are saved to ../images instead of being shown.

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs('../images', exist_ok=True)
plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'

x = np.load('../../output/data_eagles_vezhen/qrf/vezhen_x3.npy', allow_pickle=True)

features = [
    'SPEED', 'COURSE', 'currentSpeed', 'gust', 'swellHeight',
    'waveHeight', 'windSpeed', 'dif', 'COURSE-dif', 'delta_dif',
    'COURSE_vec_x', 'COURSE_vec_y', 'currentDirection_vec_x',
    'currentDirection_vec_y', 'swellDirection_vec_x',
    'swellDirection_vec_y', 'waveDirection_vec_x', 'waveDirection_vec_y',
    'windDirection_vec_x', 'windDirection_vec_y'
]
df_x = pd.DataFrame(x.reshape((-1, 5, 20))[:, 0, :], columns=features)

with open('../../output/data_eagles_vezhen/min-max-scaler-eagles-vezhen.pkl', 'rb') as f:
    scaler = pickle.load(f)

min_max_scale_cols = [
    'SPEED',
    'currentSpeed',
    'gust',
    'swellHeight',
    'waveHeight',
    'windSpeed',
]
vezhen_true = scaler.inverse_transform(df_x[min_max_scale_cols])

# Wind speed against the 17.5 threshold
plt.figure()
plt.plot(vezhen_true[:, 5])
plt.xlabel('Index in prepared dataframe')
plt.axhline(y=17.5, color='red')
plt.savefig('../images/vezhen-wind-speed.png', dpi=300)
plt.close()

print(np.arange(vezhen_true.shape[0])[vezhen_true[:, 5] > 17.5])

# Wave height
waveh_min = scaler.data_min_[4]
waveh_range = scaler.data_range_[4]
plt.figure()
((df_x['waveHeight'] + waveh_min) * waveh_range).plot()
plt.savefig('../images/vezhen-wave-height.png', dpi=300)
plt.close()

# Gust
plt.figure()
df_x['gust'].plot()
plt.savefig('../images/vezhen-gust.png', dpi=300)
plt.close()

# Speed over ground from the raw position report
vezhen_df = pd.read_csv('../../data/VEZHEN_PositionReport_5min_25Jan2025_26Jan2025.csv')
plt.figure()
vezhen_df['SPEED'].plot()
plt.savefig('../images/vezhen-sog.png', dpi=300)
plt.close()