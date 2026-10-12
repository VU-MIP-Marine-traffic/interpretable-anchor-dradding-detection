# TODO: check this one

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import matplotlib.pyplot as plt
from quantile_forest import RandomForestQuantileRegressor


# 1. Validate on all variables on both ships on model with all parameters
# 2. Validate on selected variables on both ships no model with only relevant parameters

# In[2]:


X_train = np.load('../../output/data_eagles_vezhen/qrf/X_train2.npy')
y_train = np.load('../../output/data_eagles_vezhen/qrf/y_train2.npy')
X_test = np.load('../../output/data_eagles_vezhen/qrf/X_test2.npy')
y_test = np.load('../../output/data_eagles_vezhen/qrf/y_test2.npy')
X_val = np.load('../../output/data_eagles_vezhen/qrf/X_val2.npy')
y_val = np.load('../../output/data_eagles_vezhen/qrf/y_val2.npy')

SLIDINGW_STEPS = 5
N_OUT_FEATS = y_train.shape[-1]
N_INP_FEATS = X_train.shape[-1]

y_train = y_train.reshape((-1, SLIDINGW_STEPS*N_OUT_FEATS))
X_train = X_train.reshape((-1, SLIDINGW_STEPS*N_INP_FEATS))
y_val = y_val.reshape((-1, SLIDINGW_STEPS*N_OUT_FEATS))
X_val = X_val.reshape((-1, SLIDINGW_STEPS*N_INP_FEATS))
y_test = y_test.reshape((-1, SLIDINGW_STEPS*N_OUT_FEATS))
X_test = X_test.reshape((-1, SLIDINGW_STEPS*N_INP_FEATS))


# # Base model with all features

# In[3]:


with open('../../output/models/qrf/hyperparam-fourth.pkl', 'rb+') as f:
    hyperparam_res = pickle.load(f)

# hyperparam_res


# In[4]:


import pandas as pd
df_hyper = pd.DataFrame(hyperparam_res).melt()
df_hyper.columns = ['max_samples_leaf', 'max_features', 'max_samples', 'max_depth', 'value']
df_hyper['loss_low'] = df_hyper['value'].apply(lambda x: x[0])
df_hyper['loss_high'] = df_hyper['value'].apply(lambda x: x[1])
df_hyper.drop(columns=['value'], inplace=True)


# In[5]:


df_hyper.groupby('max_samples_leaf')\
    [['loss_low', 'loss_high']]\
    .agg(['mean', 'std'])


# In[6]:


df_hyper.groupby('max_samples')\
    [['loss_low', 'loss_high']]\
    .agg(['mean', 'std'])


# In[7]:


df_hyper.groupby('max_features')\
    [['loss_low', 'loss_high']]\
    .agg(['mean', 'std'])


# In[8]:


df_hyper.groupby('max_depth', dropna=False)\
    [['loss_low', 'loss_high']]\
    .agg(['mean', 'std'])


# In[9]:


df_hyper.iloc[df_hyper['loss_low'].argmin(), :]


# In[10]:


df_hyper.iloc[df_hyper['loss_high'].argmin(), :]


# In[ ]:


best_params = {
    'n_estimators': 50,
    'max_depth': 8,
    'max_samples': 1.0,
    'max_features': 0.75,
    'max_samples_leaf': 5,
}


# In[12]:


qrf_best = RandomForestQuantileRegressor(**best_params, n_jobs=-1)\
    .fit(X_train, y_train)


# In[13]:


y_pred = qrf_best.predict(X_test, quantiles=[0.025, 0.975])
y_low = y_pred[:, :, 0]
y_high = y_pred[:, :, 1]


# In[14]:


np.mean( np.abs(y_high - y_low) )


# In[15]:


np.mean( (y_test < y_high) & (y_test > y_low) )


# In[16]:


y_test =  y_test.reshape((-1,5,3))
y_low = y_low.reshape((-1,5,3))
y_high = y_high.reshape((-1,5,3))
picp_res = np.mean( (y_test < y_high) & (y_test > y_low), axis=0 )
pinaw_res = np.mean( np.abs(y_high - y_low), axis=0 )


# In[17]:


df_picp = pd.DataFrame(picp_res, columns=['delta_cog', 'delta_dif', 'dif'])
df_picp['time_step'] = list(range(5,30,5))
df_picp


# In[18]:


df_pinaw = pd.DataFrame(pinaw_res, columns=['delta_cog', 'delta_dif', 'dif'])
df_pinaw['time_step'] = list(range(5,30,5))
df_pinaw


# ### TODO: pridet grafiku

# In[19]:


y_val_pred = qrf_best.predict(X_val, quantiles=[0.025, 0.975])
y_val_low = y_val_pred[:, :, 0]
y_val_high = y_val_pred[:, :, 1]

val_mmsis = np.load('../../output/data_eagles_vezhen/qrf/val-mmsis2.npy', allow_pickle=True)
val_days = np.load('../../output/data_eagles_vezhen/qrf/val-days2.npy', allow_pickle=True)


# In[20]:


in_pi_mask = (y_val > y_val_low) & (y_val < y_val_high)
in_pi_mask = in_pi_mask.reshape((-1, 5, 3))


# In[21]:


group_results = {}
for ix, (mmsi, day) in enumerate(zip(val_mmsis, val_days)):
    
    if (mmsi, day) not in group_results:
        group_results[(mmsi, day)] = [ in_pi_mask[ix, :, :] ]
    else:
        group_results[(mmsi, day)].append( in_pi_mask[ix, :, :] )


# In[22]:


df = []

for (mmsi, day), pi_masks in group_results.items():
    means = np.array(pi_masks).mean(axis=1).mean(axis=0)
    df.append({
        'mmsi': mmsi,
        'day': day,
        'delta_cog': means[0],
        'delta_dif': means[1],
        'dif': means[2]
    })

df = pd.DataFrame(df)
df


# In[23]:


thresh_delta_cog = np.quantile(df['delta_cog'], 0.05)
thresh_delta_dif = np.quantile(df['delta_dif'], 0.05)
thresh_dif = np.quantile(df['dif'], 0.05)

print( 'delta_cog limit:', np.quantile(df['delta_cog'], 0.05) )
print( 'delta_dif liimt:', np.quantile(df['delta_dif'], 0.05) )
print( 'dif liimt:', np.quantile(df['dif'], 0.05) )


# In[24]:


in_pi_mask = (y_test > y_low) & (y_test < y_high)
in_pi_mask = in_pi_mask.reshape((-1, 5, 3))

test_mmsis = np.load('../../output/data_eagles_vezhen/qrf/test-mmsis2.npy', allow_pickle=True)
test_days = np.load('../../output/data_eagles_vezhen/qrf/test-days2.npy', allow_pickle=True)


# In[25]:


group_results = {}
for ix, (mmsi, day) in enumerate(zip(test_mmsis, test_days)):
    
    if (mmsi, day) not in group_results:
        group_results[(mmsi, day)] = [ in_pi_mask[ix, :, :] ]
    else:
        group_results[(mmsi, day)].append( in_pi_mask[ix, :, :] )


# In[26]:


df = []

for (mmsi, day), pi_masks in group_results.items():
    means = np.array(pi_masks).mean(axis=1).mean(axis=0)
    df.append({
        'mmsi': mmsi,
        'day': day,
        'delta_cog': means[0],
        'delta_dif': means[1],
        'dif': means[2]
    })

df = pd.DataFrame(df)


# In[27]:


(df['delta_cog'] < thresh_delta_cog).mean()


# In[28]:


(df['dif'] < thresh_dif).mean()


# In[29]:


(df['delta_dif'] < thresh_delta_dif).mean()


# In[30]:


vezhen_y = np.load('../../output/data_eagles_vezhen/qrf/vezhen_y2.npy', allow_pickle=True)\
    .reshape((-1, 5*3))
vezhen_x = np.load('../../output/data_eagles_vezhen/qrf/vezhen_x2.npy', allow_pickle=True)\
    .reshape((-1, 5*20))
eagles_y = np.load('../../output/data_eagles_vezhen/qrf/eagles_y2.npy', allow_pickle=True)\
    .reshape((-1, 5*3))
eagles_x = np.load('../../output/data_eagles_vezhen/qrf/eagles_x2.npy', allow_pickle=True)\
    .reshape((-1, 5*20))


# In[31]:


eagles_model_out = qrf_best.predict(eagles_x, quantiles=[0.025, 0.975])
eagles_low, eagles_high = eagles_model_out[:, :, 0], eagles_model_out[:, :, 1]


# In[32]:


eagles_low = eagles_low.reshape((eagles_low.shape[0], 5, 3))
eagles_high = eagles_high.reshape((eagles_high.shape[0], 5, 3))

cog_delta_eagles_low = eagles_low[:, 1, 0]
cog_delta_eagles_high = eagles_high[:, 1, 0]

delta_dif_eagles_low = eagles_low[:, 1, 1]
delta_dif_eagles_high = eagles_high[:, 1, 1]

dif_eagles_low = eagles_low[:, 1, 2]
dif_eagles_high = eagles_high[:, 1, 2]


# In[33]:


np.mean( (cog_delta_eagles_low[:154] < eagles_y[:154, 0]) & (cog_delta_eagles_high[:154] > eagles_y[:154, 0]) )


# In[34]:


np.mean( (delta_dif_eagles_low[:154] < eagles_y[:154, 1]) & (delta_dif_eagles_high[:154] > eagles_y[:154, 1]) )


# In[35]:


np.mean( (dif_eagles_low[:154] < eagles_y[:154, 2]) & (dif_eagles_high[:154] > eagles_y[:154, 2]) )


# In[36]:


plt.plot(cog_delta_eagles_low[:154])
plt.plot(cog_delta_eagles_high[:154])
plt.plot(eagles_y[:154, 0])


# In[37]:


plt.plot(delta_dif_eagles_low[:154])
plt.plot(delta_dif_eagles_high[:154])
plt.plot(eagles_y[:154, 1])


# In[38]:


plt.plot(dif_eagles_low[:154])
plt.plot(dif_eagles_high[:154])
plt.plot(eagles_y[:154, 2])


# In[39]:


import time
time.sleep(120)


# In[ ]:


np.mean( (
    (dif_eagles_low < eagles_y.reshape(eagles_y.shape[0], 5, 3)[:, 0, 2]) &
    (dif_eagles_high > eagles_y.reshape(eagles_y.shape[0], 5, 3)[:, 0, 2])
) )


# In[ ]:


import time
time.sleep(120)


# In[ ]:


plt.plot(delta_dif_eagles_low)
plt.plot(delta_dif_eagles_high)
plt.plot(eagles_y.reshape(eagles_y.shape[0], 5, 3)[:, 0, 1])


# In[ ]:


np.mean( (
    (delta_dif_eagles_low < eagles_y.reshape(eagles_y.shape[0], 5, 3)[:, 0, 1]) &
    (delta_dif_eagles_high > eagles_y.reshape(eagles_y.shape[0], 5, 3)[:, 0, 1])
) )


# In[ ]:


plt.plot(cog_delta_eagles_low)
plt.plot(cog_delta_eagles_high)
plt.plot(eagles_y.reshape(eagles_y.shape[0], 5, 3)[:, 0, 0])


# In[ ]:


np.mean( (
    (cog_delta_eagles_low < eagles_y.reshape(eagles_y.shape[0], 5, 3)[:, 0, 0]) &
    (cog_delta_eagles_high > eagles_y.reshape(eagles_y.shape[0], 5, 3)[:, 0, 0])
) )


# In[ ]:


vezhen_model_out = qrf_best.predict(vezhen_x, quantiles=[0.025, 0.975])
vezhen_low, vezhen_high = vezhen_model_out[:, :, 0], vezhen_model_out[:, :, 1]


# In[ ]:


vezhen_low = vezhen_low.reshape((vezhen_low.shape[0], 5, 3))
vezhen_high = vezhen_high.reshape((vezhen_high.shape[0], 5, 3))

cog_delta_vezhen_low = vezhen_low[:, 0, 0]
cog_delta_vezhen_high = vezhen_high[:, 0, 0]

delta_dif_vezhen_low = vezhen_low[:, 0, 1]
delta_dif_vezhen_high = vezhen_high[:, 0, 1]

dif_vezhen_low = vezhen_low[:, 0, 2]
dif_vezhen_high = vezhen_high[:, 0, 2]


# In[ ]:


plt.plot(dif_vezhen_low)
plt.plot(dif_vezhen_high)
plt.plot(vezhen_y.reshape(vezhen_y.shape[0], 5, 3)[:, 0, 2])


# In[ ]:


np.mean( (
    (dif_vezhen_low < vezhen_y.reshape(vezhen_y.shape[0], 5, 3)[:, 0, 2]) &
    (dif_vezhen_high > vezhen_y.reshape(vezhen_y.shape[0], 5, 3)[:, 0, 2])
) )


# In[ ]:


plt.plot(delta_dif_vezhen_low)
plt.plot(delta_dif_vezhen_high)
plt.plot(vezhen_y.reshape(vezhen_y.shape[0], 5, 3)[:, 0, 1])


# In[ ]:


np.mean( (
    (delta_dif_vezhen_low < vezhen_y.reshape(vezhen_y.shape[0], 5, 3)[:, 0, 1]) &
    (delta_dif_vezhen_high > vezhen_y.reshape(vezhen_y.shape[0], 5, 3)[:, 0, 1])
) )


# In[ ]:


plt.plot(cog_delta_vezhen_low)
plt.plot(cog_delta_vezhen_high)
plt.plot(vezhen_y[:, 0])


# In[ ]:


np.mean( (
    (cog_delta_vezhen_low < vezhen_y.reshape(vezhen_y.shape[0], 5, 3)[:, 0, 0]) &
    (cog_delta_vezhen_high > vezhen_y.reshape(vezhen_y.shape[0], 5, 3)[:, 0, 0])
) )


# # Base model with only selected features

# In[ ]:


with open('../../output/models/qrf/qrf-featcounts-loss.pkl', 'rb+') as f:
    feat_counts = pickle.load(f)
with open('../../output/models/qrf/qrf-remove-features.pkl', 'rb+') as f:
    rm_feats = pickle.load(f)


# In[ ]:


import seaborn as sns


# In[ ]:


df_featcounts_loss = pd.DataFrame(feat_counts).melt()

df_featcounts_loss['variable'] = 20-df_featcounts_loss['variable']

sns.boxplot(
    data=df_featcounts_loss,
    x='variable',
    y='value'
)

plt.xlabel('Number of features')
plt.ylabel('Loss')


# In[ ]:


df = pd.read_csv('../../output/data_eagles_vezhen/df-prepared-eagle.csv')
df.drop(columns=['day', 'MMSI']).columns


# In[ ]:


rm_feats


# In[ ]:


X_train_sel = X_train\
    .reshape((-1, 5, 20))\
    [:, :, [7, 18, 19]]\
    .reshape((-1, 5*3))

qrf_best = RandomForestQuantileRegressor(**best_params)\
    .fit(X_train_sel, y_train)


# In[ ]:


vezhen_x_sel = vezhen_x\
    .reshape((-1, 5, 20))\
    [:, :, [7, 18, 19]]\
    .reshape((-1, 5*3))

vezhen_model_out = qrf_best.predict(vezhen_x_sel, quantiles=[0.025, 0.975])
vezhen_low, vezhen_high = vezhen_model_out[:, :, 0], vezhen_model_out[:, :, 1]


# In[ ]:


plt.plot(dif_vezhen_low)
plt.plot(dif_vezhen_high)
plt.plot(vezhen_y.reshape(vezhen_y.shape[0], 5, 3)[:, 0, 2])


# In[ ]:


np.mean( (
    (dif_vezhen_low < vezhen_y.reshape(vezhen_y.shape[0], 5, 3)[:, 0, 2]) &
    (dif_vezhen_high > vezhen_y.reshape(vezhen_y.shape[0], 5, 3)[:, 0, 2])
) )


# In[ ]:


plt.plot(delta_dif_vezhen_low)
plt.plot(delta_dif_vezhen_high)
plt.plot(vezhen_y.reshape(vezhen_y.shape[0], 5, 3)[:, 0, 1])


# In[ ]:


np.mean( (
    (delta_dif_vezhen_low < vezhen_y.reshape(vezhen_y.shape[0], 5, 3)[:, 0, 1]) &
    (delta_dif_vezhen_high > vezhen_y.reshape(vezhen_y.shape[0], 5, 3)[:, 0, 1])
) )


# In[ ]:


plt.plot(cog_delta_vezhen_low)
plt.plot(cog_delta_vezhen_high)
plt.plot(vezhen_y[:, 0])


# In[ ]:


np.mean( (
    (cog_delta_vezhen_low < vezhen_y.reshape(vezhen_y.shape[0], 5, 3)[:, 0, 0]) &
    (cog_delta_vezhen_high > vezhen_y.reshape(vezhen_y.shape[0], 5, 3)[:, 0, 0])
) )


# In[ ]:


eagles_x = eagles_x\
    .reshape((-1, 5, 20))\
    [:, :, [7, 18, 19]]\
    .reshape((-1, 5*3))

eagles_model_out = qrf_best.predict(eagles_x, quantiles=[0.025, 0.975])
eagles_low, eagles_high = eagles_model_out[:, :, 0], eagles_model_out[:, :, 1]


# In[ ]:


eagles_low = eagles_low.reshape((eagles_low.shape[0], 5, 3))
eagles_high = eagles_high.reshape((eagles_high.shape[0], 5, 3))

cog_delta_eagles_low = eagles_low[:, 0, 0]
cog_delta_eagles_high = eagles_high[:, 0, 0]

delta_dif_eagles_low = eagles_low[:, 0, 1]
delta_dif_eagles_high = eagles_high[:, 0, 1]

dif_eagles_low = eagles_low[:, 0, 2]
dif_eagles_high = eagles_high[:, 0, 2]


# In[ ]:


plt.plot(cog_delta_eagles_low)
plt.plot(cog_delta_eagles_high)
plt.plot(eagles_y[:, 0])


# In[ ]:


np.mean( (
    (cog_delta_eagles_low < eagles_y.reshape(eagles_y.shape[0], 5, 3)[:, 0, 0]) &
    (cog_delta_eagles_high > eagles_y.reshape(eagles_y.shape[0], 5, 3)[:, 0, 0])
) )


# In[ ]:


plt.plot(dif_eagles_low)
plt.plot(dif_eagles_high)
plt.plot(eagles_y.reshape(eagles_y.shape[0], 5, 3)[:, 0, 2])


# In[ ]:


np.mean( (
    (dif_eagles_low < eagles_y.reshape(eagles_y.shape[0], 5, 3)[:, 0, 2]) &
    (dif_eagles_high > eagles_y.reshape(eagles_y.shape[0], 5, 3)[:, 0, 2])
) )


# In[ ]:


plt.plot(delta_dif_eagles_low)
plt.plot(delta_dif_eagles_high)
plt.plot(eagles_y.reshape(eagles_y.shape[0], 5, 3)[:, 0, 1])


# In[ ]:


np.mean( (
    (delta_dif_eagles_low < eagles_y.reshape(eagles_y.shape[0], 5, 3)[:, 0, 1]) &
    (delta_dif_eagles_high > eagles_y.reshape(eagles_y.shape[0], 5, 3)[:, 0, 1])
) )

