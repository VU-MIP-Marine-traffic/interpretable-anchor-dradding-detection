import os
import numpy as np
from src.eagles_vezhen._helper_lstm_ae_functions import (
    lstm_ae,
    lstm_ae_hp,
)
from keras_tuner import (
    # RandomSearch,
    GridSearch
)
import tensorflow as tf
import pickle
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

BATCH_SIZE = 64

X_train = np.load('output/data_eagles_vezhen/lstm-ae/X_train2.npy')
y_train = np.load('output/data_eagles_vezhen/lstm-ae/y_train2.npy')
X_test = np.load('output/data_eagles_vezhen/lstm-ae/X_test2.npy')
y_test = np.load('output/data_eagles_vezhen/lstm-ae/y_test2.npy')
X_val = np.load('output/data_eagles_vezhen/lstm-ae/X_val2.npy')
y_val = np.load('output/data_eagles_vezhen/lstm-ae/y_val2.npy')

eagles_x = np.load("output/data_eagles_vezhen/lstm-ae/eagles_x3.npy", allow_pickle=True)
eagles_y = np.load("output/data_eagles_vezhen/lstm-ae/eagles_y3.npy", allow_pickle=True)
vezhen_x = np.load("output/data_eagles_vezhen/lstm-ae/vezhen_x3.npy", allow_pickle=True)
vezhen_y = np.load("output/data_eagles_vezhen/lstm-ae/vezhen_y3.npy", allow_pickle=True)

eagles_x = eagles_x.reshape((-1, 5, 20))
eagles_y = eagles_y.reshape((-1, 5, 3))
vezhen_x = vezhen_x.reshape((-1, 5, 20))
vezhen_y = vezhen_y.reshape((-1, 5, 3))


# https://www.tensorflow.org/tutorials/load_data/numpy:
train_dataset = tf.data\
    .Dataset\
    .from_tensor_slices(
        (X_train, y_train)
    )\
    .batch(BATCH_SIZE)\
    .shuffle(buffer_size=10_000)

val_dataset = tf.data\
    .Dataset\
    .from_tensor_slices(
        (X_val, y_val)
    )\
    .batch(BATCH_SIZE)\
    .shuffle(buffer_size=10_000)

# lower bound model
gs_low = GridSearch(
    hypermodel = lambda hp: lstm_ae_hp(
        tau=0.025,
        hp=hp,
        n_out_features=y_train.shape[2],
        n_out_timesteps=y_train.shape[1],
    ),
    objective = 'val_loss',
    directory='output/models/lstm-ae-09-13',
    project_name='lstm-ae-low'
)

gs_low.search(
    train_dataset,
    epochs=30,
    validation_data=val_dataset
)

hyper_low = gs_low.get_best_hyperparameters()[0]
model_low = lstm_ae(
    tau=0.025,
    n_out_features=3,
    n_out_timesteps=5,
    **hyper_low.values
)

history_low = model_low.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=200,
    callbacks=EarlyStopping(patience=20)
)

print(
    'latent_dim', hyper_low.get('latent_dim'),
    'drop_frac', hyper_low.get('drop_frac'),
    'l1', hyper_low.get('l1'),
)
print(gs_low.get_best_hyperparameters()[0])

eagles_pred = model_low.predict(eagles_x)
vezhen_pred = model_low.predict(vezhen_x)

os.makedirs('output/models/lstm-ae/', exist_ok=True)
eagles_pred.dump('output/models/lstm-ae/eagles_output_low2.npy')
vezhen_pred.dump('output/models/lstm-ae/vezhen_output_low2.npy')

with open('output/models/lstm-ae/history_low.pkl', 'wb+') as f:
    pickle.dump(history_low.history, f)

hyper_low = gs_low.get_best_hyperparameters()[0]
model_high = lstm_ae(
    tau=0.975,
    n_out_features=3,
    n_out_timesteps=5,
    **hyper_low.values
)

history_high = model_high.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=200,
    callbacks=EarlyStopping(patience=20)
)

print(
    'latent_dim', hyper_low.get('latent_dim'),
    'drop_frac', hyper_low.get('drop_frac'),
    'l1', hyper_low.get('l1'),
)

eagles_pred = model_high.predict(eagles_x)
vezhen_pred = model_high.predict(vezhen_x)

os.makedirs('output/models/lstm-ae/', exist_ok=True)
eagles_pred.dump('output/models/lstm-ae/eagles_output_high2.npy')
vezhen_pred.dump('output/models/lstm-ae/vezhen_output_high2.npy')

with open('output/models/lstm-ae/history_high.pkl', 'wb+') as f:
    pickle.dump(history_high.history, f)
