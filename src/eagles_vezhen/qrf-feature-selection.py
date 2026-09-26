import pickle
import numpy as np
import quantile_forest as qrf
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import KFold

RFE_STEP = 2

X_train = np.load('output/data_eagles_vezhen/X_train.npy')
y_train = np.load('output/data_eagles_vezhen/y_train.npy')
X_test = np.load('output/data_eagles_vezhen/X_test.npy')
y_test = np.load('output/data_eagles_vezhen/y_test.npy')
X_val = np.load('output/data_eagles_vezhen/X_val.npy')
y_val = np.load('output/data_eagles_vezhen/y_val.npy')

SLIDINGW_STEPS = 5
N_OUT_FEATS = y_train.shape[-1]
N_INP_FEATS = X_train.shape[-1]

#merge test and validation to do 5-fold cross-validation
X_train_val = np.concat([X_train, X_val])
y_train_val = np.concat([y_train, y_val])
y_train_val = y_train_val.reshape((y_train_val.shape[0], y_train_val.shape[1]*y_train_val.shape[2]))

#get the hyperparameters from prev experiment
with open('output/models/qrf/hyperparam-initial.pkl', 'rb+') as f:
    gcv_results = pickle.load( f )
qrf_best_hyper = gcv_results.best_params_

# feature selection
list_features = list(range(X_train.shape[-1]))
list_rm_features = []
feature_counts_loss = {}

# Make sure 3 are left (because this is the number of output features)
while len(list_rm_features) <= len(list_features) - 3:

    # setup data
    list_valid_features = np.setdiff1d(list_features, list_rm_features)

    # Make sure 3 are left (because this is the number of output features)
    # If we leave step==2, we will go down to 2 features, which is too little
    if len(list_valid_features) <= 4:
        RFE_STEP = 1

    X_tmp = X_train_val[:, :, list_valid_features]
    X_tmp = X_tmp.reshape(
        (X_tmp.shape[0], X_tmp.shape[1]*X_tmp.shape[2])
    )

    losses = []
    mdi_list = []
    for train_index, val_index in KFold(n_splits=5).split(X_train_val):

        X_train_tmp = X_tmp[train_index, :]
        y_train = y_train_val[train_index, :]
        X_val_tmp = X_tmp[val_index, :]
        y_val = y_train_val[val_index, :]

        # train model
        qrf_model = qrf.RandomForestQuantileRegressor(
            n_jobs=-1,
            verbose=True,
            **qrf_best_hyper
        )

        qrf_model.fit(X_train_tmp, y_train)

        # Calculate loss
        y_pred = qrf_model.predict(X_val_tmp, quantiles=[0.025])
        loss = mean_pinball_loss(
            y_true=y_val,
            y_pred=y_pred,
            alpha=0.025
        )

        if len(list_rm_features) not in feature_counts_loss:
            feature_counts_loss[len(list_rm_features)] = [loss]
        else:
            feature_counts_loss[len(list_rm_features)].append(loss)

        # update feat list
        mdi_scores = qrf_model.feature_importances_
        mdis = np.mean(
            mdi_scores.reshape( (X_train.shape[1], len(list_valid_features)) ),
            axis=0
        )

        losses.append(loss)
        mdi_list.append(mdis)

    loss = np.mean(losses)
    mdis = np.mean( np.stack(mdi_list, axis=1), axis=1)
    print(mdis.shape)

    # Source - https://stackoverflow.com/a/58830654
    # Posted by shaik moeed, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-05-18, License - CC BY-SA 4.0

    sorted_indices = sorted(list( range(mdis.shape[0]) ), key=lambda k: mdis[k])
    drop_indices = list(sorted_indices)[:RFE_STEP]
    drop = [list_valid_features[di] for di in drop_indices]

    list_rm_features.extend(drop)

    print('MDIs:')
    print(*list(mdis), sep='\n')
    print('\n', drop)
    print('\n', loss)

with open('output/models/qrf/qrf-remove-features.pkl', 'wb+') as f:
    pickle.dump(list_rm_features, f)

with open('output/models/qrf/qrf-featcounts-loss.pkl', 'wb+') as f:
    pickle.dump(feature_counts_loss, f)
