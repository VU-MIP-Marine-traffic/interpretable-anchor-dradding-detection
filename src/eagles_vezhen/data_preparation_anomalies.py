import pandas as pd

eagles_df = pd.read_csv('data/EAGLE_S_PositionReport_5min_25Dec2024_26Dec2024.csv')
vezhen_df = pd.read_csv('data/VEZHEN_PositionReport_5min_25Jan2025_26Jan2025.csv')

raw_df = pd.concat([eagles_df, vezhen_df])

from datetime import datetime

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

from src.eagles_vezhen.data_preparation_eagles import (
    STEP_BACK_DEFAULT,
    STEP_FORW_DEFAULT,

    DOWN_SAMPLE_MINUTES,

    merge_with_meteo,
    feature_engineering
)

TOLERANCE = 0.5

LON_MIN = 18.5 - TOLERANCE
LON_MAX = 25.5 + TOLERANCE
LAT_MIN = 57.0 - TOLERANCE
LAT_MAX = 60.5 + TOLERANCE

EAGLES_SEVER = datetime(2024, 12, 25, 10, 0)
EAGLES_TAKEOVER = datetime(2024, 12, 25, 16, 40)

EAGLES_MMSI = 518998865
VEZHEN_MMSI = 229659000

meteo_df = pd.read_csv(
    'data/weather_df_eagles_vezhen3.csv',
    parse_dates=['time']
)

latitudes = meteo_df['Latitude'].unique()
longitudes = meteo_df['Longitude'].unique()
lat_lon_grid = list(product(latitudes, longitudes))
lat_lon_tree = cKDTree(lat_lon_grid)

meteo_df = meteo_df\
    .set_index( ['time', 'Latitude', 'Longitude'])\
    .drop(columns=['Unnamed: 0'])

# COG and Heading limit
raw_df = raw_df[raw_df['COURSE'] >= 0.0]
raw_df = raw_df[raw_df['HEADING'] >= 0.0]
raw_df = raw_df[raw_df['COURSE'] <= 360.0]
raw_df = raw_df[raw_df['HEADING'] <= 360.0]

# cap speed
raw_df.loc[raw_df['SPEED'] > 25, 'SPEED'] = np.nan

# Keep only underway data
# Moored > 1 knot is considered underway
nav_stat_mapping = {
    'Under way using engine': 0,
    'Under way sailing': 1,
    'Not under command': 2,
    'Moored': 5,
    'Under way sailing': 8
}
raw_df.loc[
    (raw_df['NAVSTAT'] == nav_stat_mapping['Moored']) &
    (raw_df['SPEED'] < 1)
] = np.nan

# WE DONT FILTER NAVSTAT IN OTHER WAYS - WE KEEP EVERYTHING

# We will groupby day also, so that we do not have huge gaps to interpolate
raw_df['DATE TIME (UTC)'] = pd.to_datetime(raw_df['DATE TIME (UTC)'])

# Filter to make sure we have enough data of normal ships
raw_df = raw_df[
    ( (raw_df['LATITUDE'].between(LAT_MIN, LAT_MAX)) & (raw_df['LONGITUDE'].between(LON_MIN, LON_MAX)) ) |
    ( (raw_df['MMSI'] == EAGLES_MMSI) & (raw_df['DATE TIME (UTC)'] > EAGLES_SEVER) )
]

# Filter to make sure that the EAGLE S take-over is not included
raw_df = raw_df[
    ( (raw_df['MMSI'] == EAGLES_MMSI) & (raw_df['DATE TIME (UTC)'] < EAGLES_TAKEOVER) ) |
    ( raw_df['MMSI'] == VEZHEN_MMSI)
]

print(
    raw_df.groupby('MMSI')['DATE TIME (UTC)'].min()
)

grouped_df = raw_df.set_index('DATE TIME (UTC)').groupby('MMSI')


dfs = []
for mmsi, mmsi_df in grouped_df:

    mmsi_df = mmsi_df\
        [['LATITUDE', 'LONGITUDE', 'SPEED', 'COURSE', 'HEADING']]\
        .drop_duplicates()
    mmsi_df['MMSI'] = mmsi

    # interpolate
    mmsi_df = mmsi_df\
        .resample(f'{DOWN_SAMPLE_MINUTES*60}s')\
        .mean()\
        .dropna(subset=['MMSI'])\
        .reset_index()

    # Source: https://stackoverflow.com/questions/28773342/truncate-timestamp-column-to-hour-precision-in-pandas-dataframe
    mmsi_df['hour'] = mmsi_df['DATE TIME (UTC)'].dt.round('h')
    mmsi_df['day'] = mmsi_df['DATE TIME (UTC)'].dt.date

    # merge with meteorological data
    mmsi_df = mmsi_df.apply(
        lambda row: merge_with_meteo(row, meteo_df, lat_lon_grid, lat_lon_tree),
        axis='columns'
    )
    mmsi_df = mmsi_df.drop(columns=['hour'])

    mmsi_df.sort_values('DATE TIME (UTC)')

    # create features such as dif and delta dif
    mmsi_df, timestamps = feature_engineering(mmsi_df)
    mmsi_df = mmsi_df.dropna()

    print(mmsi, timestamps.min(), timestamps.max())

    mmsi_df['MMSI'] = int(mmsi)
    dfs.append( mmsi_df )

df = pd.concat( dfs, ignore_index=True )

min_max_scale_cols = [
    'SPEED',
    'currentSpeed',
    'gust',
    'swellHeight',
    'waveHeight',
    'windSpeed',
]

df.drop(columns=['LATITUDE', 'LONGITUDE'], inplace=True)

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

    #  Remove NA for good measure
    na_mask = mmsi_df.isna().any(axis=1)
    mmsi_df = mmsi_df.loc[~na_mask, :].copy()

    X_arr, y_arr = prepare_time_series(mmsi_df)
    data[mmsi] = (X_arr, y_arr)

eagles_x, eagles_y = data[EAGLES_MMSI]
vezhen_x, vezhen_y = data[VEZHEN_MMSI]

eagles_x.dump(f'output/data_eagles_vezhen/{MODE}/eagles_x3.npy')
eagles_y.dump(f'output/data_eagles_vezhen/{MODE}/eagles_y3.npy')
vezhen_x.dump(f'output/data_eagles_vezhen/{MODE}/vezhen_x3.npy')
vezhen_y.dump(f'output/data_eagles_vezhen/{MODE}/vezhen_y3.npy')
