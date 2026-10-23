# Run using venv environment
# This file is responsible for analysis of lstm-ae and yipeng-3 validation

# Changed in https://chatgpt.com/share/6accdefe-c8a0-83ed-acc5-e80de5d60be6

# Attribution:
# Cleaned and refactored with assistance from OpenAI ChatGPT (GPT-5.6 Luna).
#
# Changes made:
# - Removed Jupyter/nbconvert cell markers and empty notebook cells.
# - Removed unused imports and unnecessary exploratory code.
# - Preserved all model training, prediction, evaluation, and data-processing logic.
# - Converted notebook-only expression outputs into explicit print() calls.
# - Preserved printed hyperparameter, prediction-interval, threshold, and
#   evaluation results.
# - Saved all figures to ../images/.
# - Added plt.close() after every saved figure.
# - Ensured all input/intermediate NPY files are loaded from ../../output/.
# - Ensured all generated CSV and NPY files are saved under ../../output/.
# - Moved CSV files that were previously written to the current directory
#   into ../../output/.
# - Preserved the original model parameters, random seed, and output structure.
# - Simplified repeated dictionary-building logic with setdefault().
# - Corrected the original "liimt" typo in printed threshold labels.
#
# Model attribution:
# OpenAI ChatGPT, GPT-5.6 Luna.


import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras_tuner import GridSearch
from sklearn.metrics import mean_pinball_loss

sys.path.append('../..')

from src.yipeng3._helper_lstm_ae_functions import (
    lstm_ae,
    lstm_ae_hp,
)

import os

os.makedirs('../images', exist_ok=True)
os.makedirs('../../output', exist_ok=True)
os.makedirs('../../output/metrics', exist_ok=True)

# Load Yipeng 3 prediction data.
yipeng3_x = np.load(
    '../../output/data_yipeng3/lstm-ae/yipeng3_x.npy',
    allow_pickle=True,
)

yipeng3_y = np.load(
    '../../output/data_yipeng3/lstm-ae/yipeng3_y.npy',
    allow_pickle=True,
).reshape((-1, 5, 3))

yipeng3_output_low = np.load(
    '../../output/models/lstm-ae/yipeng3_output_low.npy',
    allow_pickle=True,
)

yipeng3_output_high = np.load(
    '../../output/models/lstm-ae/yipeng3_output_high.npy',
    allow_pickle=True,
)


# Initial prediction interval plot.
plt.plot(yipeng3_output_high[:, 0, 0])
plt.plot(yipeng3_output_low[:, 0, 0])
plt.plot(yipeng3_y[:, 0, 0])
plt.savefig('../images/yipeng3_initial_prediction_interval.png')
plt.close()


# Load train, validation, and test data.
X_train = np.load(
    '../../output/data_yipeng3/lstm-ae/X_train.npy'
)

y_train = np.load(
    '../../output/data_yipeng3/lstm-ae/y_train.npy'
)

X_test = np.load(
    '../../output/data_yipeng3/lstm-ae/X_test.npy'
)

y_test = np.load(
    '../../output/data_yipeng3/lstm-ae/y_test.npy'
)

X_val = np.load(
    '../../output/data_yipeng3/lstm-ae/X_val.npy'
)

y_val = np.load(
    '../../output/data_yipeng3/lstm-ae/y_val.npy'
)


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_train, y_train))
    .batch(64)
    .shuffle(buffer_size=10_000)
)

val_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_val, y_val))
    .batch(64)
    .shuffle(buffer_size=10_000)
)


tf.random.set_seed(42)


# Train the lower quantile model.
gs_low = GridSearch(
    hypermodel=lambda hp: lstm_ae_hp(
        tau=0.025,
        hp=hp,
        n_out_features=y_train.shape[2],
        n_out_timesteps=y_train.shape[1],
    ),
    objective='val_loss',
    directory='../../output/models/lstm-ae-yipeng3-08-16-1',
    project_name='lstm-ae-low',
)

best_low_params = gs_low.get_best_hyperparameters()[0].values

print('Best low-quantile hyperparameters:')
print(best_low_params)

model_low = lstm_ae(
    tau=0.025,
    n_out_features=3,
    n_out_timesteps=5,
    **best_low_params,
)

history_low = model_low.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=200,
    callbacks=EarlyStopping(patience=20),
)


# Train the upper quantile model.
gs_high = GridSearch(
    hypermodel=lambda hp: lstm_ae_hp(
        tau=0.975,
        hp=hp,
        n_out_features=y_train.shape[2],
        n_out_timesteps=y_train.shape[1],
    ),
    objective='val_loss',
    directory='../../output/models/lstm-ae-yipeng3-08-16-1',
    project_name='lstm-ae-low',
)

best_high_params = gs_high.get_best_hyperparameters()[0].values

model_high = lstm_ae(
    tau=0.975,
    n_out_features=3,
    n_out_timesteps=5,
    **best_high_params,
)

history_high = model_high.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=200,
    callbacks=EarlyStopping(patience=20),
)


# Lower quantile model training history.
plt.plot(history_low.history['loss'], label='loss')
plt.plot(history_low.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig('../images/lstm_ae_low_training_history.png')
plt.close()


# Upper quantile model training history.
plt.plot(history_high.history['loss'], label='loss')
plt.plot(history_high.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig('../images/lstm_ae_high_training_history.png')
plt.close()


# Generate Yipeng 3 prediction intervals.
yipeng3_output_high = model_high.predict(yipeng3_x)
yipeng3_output_low = model_low.predict(yipeng3_x)


# Yipeng 3 delta COG prediction interval.
plt.plot(yipeng3_output_high[:, 0, 0])
plt.plot(yipeng3_output_low[:, 0, 0])
plt.plot(yipeng3_y[:, 0, 0])
plt.savefig('../images/yipeng3_prediction_interval_delta_cog.png')
plt.close()

delta_cog_coverage = np.mean(
    (yipeng3_y[:, :, 0] > yipeng3_output_low[:, :, 0])
    & (yipeng3_y[:, :, 0] < yipeng3_output_high[:, :, 0])
)

print(
    'Yipeng3 delta_cog prediction interval coverage:',
    delta_cog_coverage,
)


# Yipeng 3 delta DIF prediction interval.
plt.plot(yipeng3_output_high[:, 0, 1])
plt.plot(yipeng3_output_low[:, 0, 1])
plt.plot(yipeng3_y[:, 0, 1])
plt.savefig('../images/yipeng3_prediction_interval_delta_dif.png')
plt.close()

delta_dif_coverage = np.mean(
    (yipeng3_y[:, :, 1] > yipeng3_output_low[:, :, 1])
    & (yipeng3_y[:, :, 1] < yipeng3_output_high[:, :, 1])
)

print(
    'Yipeng3 delta_dif prediction interval coverage:',
    delta_dif_coverage,
)


# Yipeng 3 DIF prediction interval.
plt.plot(yipeng3_output_high[:, 0, 2])
plt.plot(yipeng3_output_low[:, 0, 2])
plt.plot(yipeng3_y[:, 0, 2])
plt.savefig('../images/yipeng3_prediction_interval_dif.png')
plt.close()

dif_coverage = np.mean(
    (yipeng3_y[:, :, 2] > yipeng3_output_low[:, :, 2])
    & (yipeng3_y[:, :, 2] < yipeng3_output_high[:, :, 2])
)

print(
    'Yipeng3 dif prediction interval coverage:',
    dif_coverage,
)


# Calculate test-set prediction intervals and metrics.
y_high = model_high.predict(X_test)
y_low = model_low.predict(X_test)

picp_res = np.mean(
    (y_test < y_high) & (y_test > y_low),
    axis=0,
)

pinaw_res = np.mean(
    np.abs(y_high - y_low),
    axis=0,
)

loss_low_res = mean_pinball_loss(
    y_test.reshape((-1, 15)),
    y_low.reshape((-1, 15)),
    alpha=0.025,
    multioutput='raw_values',
)

loss_high_res = mean_pinball_loss(
    y_test.reshape((-1, 15)),
    y_high.reshape((-1, 15)),
    alpha=0.975,
    multioutput='raw_values',
)


# Save prediction arrays under ../../output/.
y_high.dump('../../output/y_high_lstm_yipeng3.npy')
y_low.dump('../../output/y_low_lstm_yipeng3.npy')


df_picp = pd.DataFrame(
    picp_res,
    columns=['delta_cog', 'delta_dif', 'dif'],
)
df_picp['time_step'] = list(range(5, 30, 5))

print('\nPICP:')
print(df_picp)


df_pinaw = pd.DataFrame(
    pinaw_res,
    columns=['delta_cog', 'delta_dif', 'dif'],
)
df_pinaw['time_step'] = list(range(5, 30, 5))

print('\nPINAW:')
print(df_pinaw)


df_loss_low = pd.DataFrame(
    loss_low_res.reshape((5, 3)),
    columns=['delta_cog', 'delta_dif', 'dif'],
)
df_loss_low['time_step'] = list(range(5, 30, 5))

print('\nLow-quantile pinball loss:')
print(df_loss_low)


df_loss_high = pd.DataFrame(
    loss_high_res.reshape((5, 3)),
    columns=['delta_cog', 'delta_dif', 'dif'],
)
df_loss_high['time_step'] = list(range(5, 30, 5))

print('\nHigh-quantile pinball loss:')
print(df_loss_high)


# Save all CSV results under ../../output/.
df_picp.to_csv(
    '../../output/metrics/picp-lstm.csv'
)

df_pinaw.to_csv(
    '../../output/metrics/pinaw-lstm.csv'
)

df_picp.to_csv(
    '../../output/metrics/df_picp_lstm_yipeng.csv'
)

df_pinaw.to_csv(
    '../../output/metrics/df_pinaw_lstm_yipeng.csv'
)

df_picp.to_csv(
    '../../output/metrics/picp-lstm.csv'
)

df_pinaw.to_csv(
    '../../output/metrics/pinaw-lstm.csv'
)

df_loss_low.to_csv(
    '../../output/metrics/loss_low-lstm.csv'
)

df_loss_high.to_csv(
    '../../output/metrics/loss_high-lstm.csv'
)


# Calculate validation-set prediction interval coverage.
y_val_low = model_low.predict(X_val)
y_val_high = model_high.predict(X_val)

val_mmsis = np.load(
    '../../output/data_yipeng3/lstm-ae/val-mmsis.npy',
    allow_pickle=True,
)

val_days = np.load(
    '../../output/data_yipeng3/lstm-ae/val-days.npy',
    allow_pickle=True,
)

in_pi_mask = (y_val > y_val_low) & (y_val < y_val_high)
in_pi_mask = in_pi_mask.reshape((-1, 5, 3))

group_results = {}

for ix, (mmsi, day) in enumerate(zip(val_mmsis, val_days)):
    group_results.setdefault((mmsi, day), []).append(in_pi_mask[ix])

df = []

for (mmsi, day), pi_masks in group_results.items():
    means = np.array(pi_masks).mean(axis=1).mean(axis=0)

    df.append(
        {
            'mmsi': mmsi,
            'day': day,
            'delta_cog': means[0],
            'delta_dif': means[1],
            'dif': means[2],
        }
    )

df = pd.DataFrame(df)

thresh_delta_cog = np.quantile(df['delta_cog'], 0.05)
thresh_delta_dif = np.quantile(df['delta_dif'], 0.05)
thresh_dif = np.quantile(df['dif'], 0.05)

print('delta_cog limit:', thresh_delta_cog)
print('delta_dif limit:', thresh_delta_dif)
print('dif limit:', thresh_dif)


# Calculate test-set prediction interval coverage.
in_pi_mask = (y_test > y_low) & (y_test < y_high)
in_pi_mask = in_pi_mask.reshape((-1, 5, 3))

test_mmsis = np.load(
    '../../output/data_yipeng3/lstm-ae/test-mmsis.npy',
    allow_pickle=True,
)

test_days = np.load(
    '../../output/data_yipeng3/lstm-ae/test-days.npy',
    allow_pickle=True,
)

group_results = {}

for ix, (mmsi, day) in enumerate(zip(test_mmsis, test_days)):
    group_results.setdefault((mmsi, day), []).append(in_pi_mask[ix])

df = []

for (mmsi, day), pi_masks in group_results.items():
    means = np.array(pi_masks).mean(axis=1).mean(axis=0)

    df.append(
        {
            'mmsi': mmsi,
            'day': day,
            'delta_cog': means[0],
            'delta_dif': means[1],
            'dif': means[2],
        }
    )

df = pd.DataFrame(df)

print(
    'Test delta_cog below threshold:',
    (df['delta_cog'] < thresh_delta_cog).mean(),
)

print(
    'Test dif below threshold:',
    (df['dif'] < thresh_dif).mean(),
)

print(
    'Test delta_dif below threshold:',
    (df['delta_dif'] < thresh_delta_dif).mean(),
)