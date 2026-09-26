from src.yipeng3._helper_lstm_ae_functions import lstm_ae
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold, train_test_split
from itertools import product
from sklearn.metrics import mean_pinball_loss
import pandas as pd
import numpy as np
import tensorflow as tf

BATCH_SIZE = 64

def train_and_eval_lstm(X_train, X_test, y_train, y_test, best_hyper):

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)
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

    model_low = lstm_ae(
        tau=0.025,
        n_out_features=3,
        n_out_timesteps=5,
        **best_hyper
    )
    _ = model_low.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=200,
        callbacks=EarlyStopping(patience=20)
    )

    model_low.fit(X_train, y_train)
    model_high = lstm_ae(
        tau=0.975,
        n_out_features=3,
        n_out_timesteps=5,
        **best_hyper
    )
    _ = model_high.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=200,
        callbacks=EarlyStopping(patience=20)
    )

    model_high.fit(X_train, y_train)

    y_low = model_low.predict(X_test)
    y_high = model_high.predict(X_test)

    picp_res = np.mean( (y_test < y_high) & (y_test > y_low), axis=0 )
    pinaw_res = np.mean( np.abs(y_high - y_low), axis=0 )

    loss_low = []
    loss_high = []
    for timestep,feature in product(range(5), range(3)):
        loss_low.append(
            mean_pinball_loss(y_test[:, timestep, feature], y_low[:, timestep, feature], alpha=0.025)
        )
        loss_high.append(
            mean_pinball_loss(y_test[:, timestep, feature], y_high[:, timestep, feature], alpha=0.975)
        )
    loss_low = np.array(loss_low).reshape((-1, 3))
    loss_high = np.array(loss_high).reshape((-1, 3))

    df_picp = pd.DataFrame(picp_res, columns=['delta_cog', 'delta_dif', 'dif'])
    df_picp['time_step'] = list(range(5,30,5))
    df_pinaw = pd.DataFrame(pinaw_res, columns=['delta_cog', 'delta_dif', 'dif'])
    df_pinaw['time_step'] = list(range(5,30,5))

    df_loss_low = pd.DataFrame(loss_low, columns=['delta_cog', 'delta_dif', 'dif'])
    df_loss_low['time_step'] = list(range(5,30,5))
    df_loss_high = pd.DataFrame(loss_high, columns=['delta_cog', 'delta_dif', 'dif'])
    df_loss_high['time_step'] = list(range(5,30,5))

    return df_picp, df_pinaw, df_loss_low, df_loss_high

X_train = np.load('output/data_yipeng3/lstm-ae/X_train.npy')
y_train = np.load('output/data_yipeng3/lstm-ae/y_train.npy')
X_test = np.load('output/data_yipeng3/lstm-ae/X_test.npy')
y_test = np.load('output/data_yipeng3/lstm-ae/y_test.npy')
X_val = np.load('output/data_yipeng3/lstm-ae/X_val.npy')
y_val = np.load('output/data_yipeng3/lstm-ae/y_val.npy')

best_params = {
    'latent_dim': 5,
    'drop_frac': 0.05,
    'l1': 0.01
}

X = np.concatenate([X_train, X_val, X_test])
y = np.concatenate([y_train, y_val, y_test])
folds = KFold(n_splits=5, random_state=42, shuffle=True)

df_picp = pd.DataFrame()
df_pinaw = pd.DataFrame()
df_loss_low = pd.DataFrame()
df_loss_high = pd.DataFrame()

for ix, (train_idx, test_idx) in enumerate( folds.split(X,y) ):

    X_train = X[train_idx, :]
    y_train = y[train_idx, :]
    X_test = X[test_idx, :]
    y_test = y[test_idx, :]

    print('fold',ix,'shapes:', X_train.shape,y_train.shape,X_test.shape,y_test.shape)

    (
        df_picp_fold,
        df_pinaw_fold,
        df_loss_low_fold,
        df_loss_high_fold,
    ) = train_and_eval_lstm(X_train, X_test, y_train, y_test, best_params)

    df_picp_fold['Fold'] = ix
    df_pinaw_fold['Fold'] = ix
    df_loss_low_fold['Fold'] = ix
    df_loss_high_fold['Fold'] = ix

    df_picp = pd.concat([df_picp, df_picp_fold])
    df_pinaw = pd.concat([df_pinaw, df_pinaw_fold])
    df_loss_low = pd.concat([df_loss_low, df_loss_low_fold])
    df_loss_high = pd.concat([df_loss_high, df_loss_high_fold])

    print(df_loss_low)

df_picp.to_csv('picp-lstm-yp3.csv')
df_pinaw.to_csv('pinaw-lstm-yp3.csv')
df_loss_low.to_csv('loss_low-lstm-yp3.csv')
df_loss_high.to_csv('loss_high-lstm-yp3.csv')

