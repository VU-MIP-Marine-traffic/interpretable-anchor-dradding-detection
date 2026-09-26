import matplotlib.pyplot as plt
import pandas as pd

raw_df = pd.read_csv('data/414270000_20241118_sample(in).csv')

from datetime import timedelta, date

from itertools import product
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import cKDTree

import sys
if len(sys.argv) < 2:
    MODE = 'qrf'
elif sys.argv[1] == 'lstm-ae':
    MODE = 'lstm-ae'
else:
    MODE = 'qrf'

print(MODE)

from src.yipeng3.data_preparation_normal import (
    STEP_BACK_DEFAULT,
    STEP_FORW_DEFAULT,

    DOWN_SAMPLE_MINUTES,

    merge_with_meteo,
    feature_engineering
)

# Data preparation adapted from 1data_preaparation_eagles.py

meteo_df = pd.read_csv(
    'data/weather_df_yipeng3_v2.csv',
    parse_dates=['time']
)

latitudes = meteo_df['Latitude'].unique()
longitudes = meteo_df['Longitude'].unique()
lat_lon_grid = list(product(latitudes, longitudes))
lat_lon_tree = cKDTree(lat_lon_grid)

meteo_df = meteo_df\
    .set_index( ['time', 'Latitude', 'Longitude'])\
    .drop(columns=['Unnamed: 0'])

# course and heading limit
raw_df = raw_df[raw_df['course'] >= 0.0]
raw_df = raw_df[raw_df['heading'] >= 0.0]
raw_df = raw_df[raw_df['course'] <= 360.0]
raw_df = raw_df[raw_df['heading'] <= 360.0]

# cap speed 
raw_df.loc[raw_df['speed'] > 25, 'speed'] = np.nan

# We will groupby day also, so that we do not have huge gaps to interpolate
raw_df['# Timestamp'] = pd.to_datetime(raw_df['static_timestamp'])

# Source - https://stackoverflow.com/a/49210679
# Posted by MaFF, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-16, License - CC BY-SA 4.0
raw_df['# Timestamp'] = raw_df['# Timestamp'].dt.tz_localize(None)

raw_df.drop(columns=['static_timestamp'], inplace=True)
grouped_df = raw_df.set_index('# Timestamp').groupby('mmsi')

dfs = []
for mmsi, mmsi_df in grouped_df:

    mmsi_df = mmsi_df\
        [['latitude', 'longitude', 'speed', 'course', 'heading']]\
        .drop_duplicates()
    mmsi_df['mmsi'] = mmsi

    print(mmsi_df.shape)

    # interpolate
    mmsi_df = mmsi_df\
        .resample(f'{DOWN_SAMPLE_MINUTES*60}s')\
        .first()\
        .dropna(subset=['mmsi'])\
        .reset_index()

    print(mmsi_df.shape)

    # Source: https://stackoverflow.com/questions/28773342/truncate-timestamp-column-to-hour-precision-in-pandas-dataframe
    mmsi_df['hour'] = mmsi_df['# Timestamp'].dt.round('h')
    mmsi_df['day'] = mmsi_df['# Timestamp'].dt.date

    mmsi_df = mmsi_df.rename(columns={
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'heading': 'Heading',
        'course': 'COG',
        'mmsi': 'MMSI'
    })

    # merge with meteorological data
    mmsi_df = mmsi_df.apply(
        lambda row: merge_with_meteo(row, meteo_df, lat_lon_grid, lat_lon_tree),
        axis='columns'
    )
    mmsi_df = mmsi_df.drop(columns=['hour'])

    mmsi_df.sort_values('# Timestamp')

    # create features such as dif and delta dif
    mmsi_df, timestamps = feature_engineering(mmsi_df)
    mmsi_df = mmsi_df.dropna()

    print(mmsi_df.shape)

    mmsi_df['MMSI'] = int(mmsi)
    dfs.append( mmsi_df )

df = pd.concat( dfs, ignore_index=True )

min_max_scale_cols = [
    # 'LATITUDE',
    # 'LONGITUDE',
    'speed',
    'currentSpeed',
    'gust',
    'swellHeight',
    'waveHeight',
    'windSpeed',
]

df.drop(columns=['Latitude', 'Longitude'], inplace=True)

scaler = MinMaxScaler(feature_range=(-1, 1))
df[min_max_scale_cols] = scaler.fit_transform(
    df[min_max_scale_cols].values
)

def prepare_time_series(
    df: pd.DataFrame,
    n_lags: int = STEP_BACK_DEFAULT,
    n_future_lag: int = STEP_FORW_DEFAULT
):
    outputs = [ 'COURSE-dif', 'delta_dif', 'dif' ]
    n_features = df.shape[1]

    # so the input would be in accordance with:
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM#call-arguments
    X_arr = np.empty( (df.shape[0], n_lags, n_features) )
    y_arr = np.empty( (df.shape[0], n_future_lag, len(outputs)) )

    for ix, col in enumerate( df.columns ):
        for lag in range( 1, n_lags+1 ):
            X_arr[:, lag-1, ix] = df[col].shift(-1 * lag)

    for ix, col in enumerate( outputs ):
        for lag in range( 1, n_lags+1 ):
            y_arr[:, lag-1, ix] = df[col].shift(lag)

    return X_arr[n_lags:-n_lags, :, :], y_arr[n_lags:-n_lags, :, :]

print(df.columns)
data = {}

for mmsi, mmsi_df in df.groupby('MMSI'):

    na_mask = mmsi_df.isna().any(axis=1)
    mmsi_df = mmsi_df.loc[~na_mask, :].copy()
    mmsi_df.drop( columns=['MMSI', 'day'], inplace=True )

    print(mmsi_df.columns)

    #  Isimt NA vel "for good measure"
    na_mask = mmsi_df.isna().any(axis=1)
    mmsi_df = mmsi_df.loc[~na_mask, :].copy()

    X_arr, y_arr = prepare_time_series(mmsi_df)
    data[mmsi] = (X_arr, y_arr)

vezhen_x, vezhen_y = data[414270000]

vezhen_x.dump(f'output/data_yipeng3/{MODE}/yipeng3_x.npy')
vezhen_y.dump(f'output/data_yipeng3/{MODE}/yipeng3_y.npy')
