import pickle
import numpy as np
import quantile_forest as qrf
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import KFold

RFE_STEP = 2

X_train = np.load('output/data_eagles_vezhen/qrf/X_train.npy')
y_train = np.load('output/data_eagles_vezhen/qrf/y_train.npy')
X_test = np.load('output/data_eagles_vezhen/qrf/X_test.npy')
y_test = np.load('output/data_eagles_vezhen/qrf/y_test.npy')
X_val = np.load('output/data_eagles_vezhen/qrf/X_val.npy')
y_val = np.load('output/data_eagles_vezhen/qrf/y_val.npy')

SLIDINGW_STEPS = 5
N_OUT_FEATS = y_train.shape[-1]
N_INP_FEATS = X_train.shape[-1]

#merge test and validation to do 5-fold cross-validation
X_train_val = np.concat([X_train, X_val])
y_train_val = np.concat([y_train, y_val])
y_train_val = y_train_val.reshape((y_train_val.shape[0], y_train_val.shape[1]*y_train_val.shape[2]))

# hand filled from qrf-grid-search-and-anomaly-validation.ipynb
qrf_best_hyper = {
    'max_depth': 32,
    'max_samples': 0.5,
    'max_features': 'sqrt',
    'max_samples_leaf': 5,
}

tree_counts_losses = {
    250: [],
    200: [],
    150: [],
    100: [],
    50: [],
    25: [],
}

for tree_count in tree_counts_losses.keys():

    losses = []
    mdi_list = []
    for train_index, val_index in KFold(n_splits=5).split(X_train_val):

        X_train = X_train_val[train_index, :, :]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
        X_val = X_train_val[val_index, :, :]
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1]*X_val.shape[2])
        y_train = y_train_val[train_index, :]
        y_val = y_train_val[val_index, :]

        # train model
        qrf_model = qrf.RandomForestQuantileRegressor(
            n_jobs=-1,
            verbose=True,
            n_estimators=tree_count,
            **qrf_best_hyper
        )
        qrf_model.fit(X_train, y_train)

        # Calculate loss
        y_pred = qrf_model.predict(X_val, quantiles=[0.025])
        loss = mean_pinball_loss(
            y_true=y_val,
            y_pred=y_pred,
            alpha=0.025
        )
        losses.append(loss)

    loss_mean = np.mean(losses)
    loss_std = np.std(losses)

    print(loss_mean)
    print(loss_std)

    tree_counts_losses[tree_count] = losses

with open('output/models/qrf/qrf-tree-count-losses.pkl', 'wb+') as f:
    pickle.dump(tree_counts_losses, f)
