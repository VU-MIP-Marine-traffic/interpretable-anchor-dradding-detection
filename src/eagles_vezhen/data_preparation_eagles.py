"""
In this file we prepare the AIS data from VesselFinder.
"""

#debugger
import pdb

import os
import sys
import pickle
import random
from datetime import timedelta, date

if len(sys.argv) < 2:
    MODE = 'qrf'
elif sys.argv[1] == 'lstm-ae':
    MODE = 'lstm-ae'
else:
    MODE = 'qrf'

print(MODE)

import warnings
from itertools import product

import pandas as pd
from pandas.errors import PerformanceWarning
import numpy as np

from src.eagles_vezhen.meteorological_data_collection import construct_coordinate_grid
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Sliding window constants
STEP_BACK_DEFAULT = 5
STEP_FORW_DEFAULT = 5
ROLLING_WINDOW_SIZE = 2

# Average number of minutes to which down-sample
DOWN_SAMPLE_MINUTES = 5

# Maximum gap in the series to discard in minutes
GAP_LIMIT = 120

# columns to which sine and cosine is going to be generated
ANGLE_COLS = [
    'COURSE',
    'currentDirection',
    'swellDirection',
    'waveDirection',
    'windDirection'
]
OUTPUTS = [ 'COURSE-dif', 'delta_dif', 'dif' ]

os.makedirs(f'output/data_eagles_vezhen/{MODE}', exist_ok=True)

def merge_with_meteo(
    row: pd.Series,
    weather_df: pd.DataFrame,
    coordinate_grid: list,
    coordinate_tree: cKDTree
) -> pd.Series:

    # query returns distances (d) and indices (i)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html
    _, indices = coordinate_tree.query(
        (row['LATITUDE'], row['LONGITUDE']), k=1
    )
    lat, lon = coordinate_grid[indices]

    time = pd.Timestamp( row['hour'] )

    weather_row = weather_df.loc[(time, lat, lon), :]

    data = np.concatenate( 
        [row.values, weather_row.values[0]]
    )

    columns = list(row.index) + list(weather_df.columns)
    return pd.Series(data=data, index=columns)


def feature_engineering(df: pd.DataFrame) -> tuple:

    df['date_diff'] = df['DATE TIME (UTC)'].diff(1) / timedelta(minutes=1)

    # Calculate dif
    df['dif'] = ( df['HEADING'] - df['COURSE']  + 180) % 360 - 180
    df['dif'] = df['dif'] / df['date_diff']

    df['COURSE-dif'] = ( df['COURSE'].diff(1)  + 180) % 360 - 180

    # Do min-max scaling to [-1, 1]
    df['COURSE-dif'] = df['COURSE-dif'] / df['date_diff'] / 180 * DOWN_SAMPLE_MINUTES

    df['dif'] = df['dif'] / 180 * DOWN_SAMPLE_MINUTES
    df['delta_dif'] = df['dif'].diff(1) / 2

    df.drop( columns=['HEADING'], inplace=True )
    for col in ANGLE_COLS:
        df[f'{col}_vec_x'] = np.cos(df[col])
        df[f'{col}_vec_y'] = np.sin(df[col])

    # Drop all angle columns apart from  COURSE
    drop_angle = np.setdiff1d(ANGLE_COLS, ['COURSE'])
    df.drop(columns=drop_angle, inplace=True)

    for col in OUTPUTS:
        df[col] = np.cbrt(df[col])\
            .rolling(ROLLING_WINDOW_SIZE)\
            .mean()

    columns_to_drop = ['DATE TIME (UTC)', 'MMSI', 'date_diff']

    timestamps = df['DATE TIME (UTC)'].values
    df.drop(columns=columns_to_drop, inplace=True)
    return df, timestamps

def prepare_time_series(
    df: pd.DataFrame,
    n_lags: int = STEP_BACK_DEFAULT,
    n_future_lag: int = STEP_FORW_DEFAULT
):
    n_features = df.shape[1]

    # so the input would be in accordance with:
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM#call-arguments
    X_arr = np.empty( (df.shape[0], n_lags, n_features) )
    y_arr = np.empty( (df.shape[0], n_future_lag, len(OUTPUTS)) )

    for ix, col in enumerate( df.columns ):
        for lag in range( 1, n_lags+1 ):
            X_arr[:, lag-1, ix] = df[col].shift(-1 * lag)

    for ix, col in enumerate( OUTPUTS ):
        for lag in range( 1, n_lags+1 ):
            y_arr[:, lag-1, ix] = df[col].shift(lag)

    return X_arr[n_lags:-n_lags, :, :], y_arr[n_lags:-n_lags, :, :]


def partition_groups(mmsi_day):

    groups_shuffled = mmsi_day.copy()
    random.shuffle( mmsi_day )

    n = len(mmsi_day)
    n_train = int(0.7 * n)
    n_val = int( 0.85 * n )

    return (
        groups_shuffled[:n_train],
        groups_shuffled[n_train:n_val],
        groups_shuffled[n_val:]
    )


def belongs_in(mmsi_day, group_list):

    mmsi, day = mmsi_day

    for (mmsi_group, day_group) in group_list:
        if mmsi_group == mmsi and day_group == day:
            return True

    return False


if __name__ == "__main__":

    raw_df = pd.read_csv('data/Vessels_PositionReport_5min_25Dec2024_25Jan2025.csv')

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

    print( 'AIS signals', raw_df.shape[0] )

    # COG and Heading limit
    raw_df = raw_df[raw_df['COURSE'] >= 0.0]
    raw_df = raw_df[raw_df['HEADING'] >= 0.0]
    raw_df = raw_df[raw_df['COURSE'] <= 360.0]
    raw_df = raw_df[raw_df['HEADING'] <= 360.0]

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

    # Drop anyone not underway
    raw_df = raw_df[
        (raw_df['NAVSTAT'] == nav_stat_mapping['Under way using engine']) |
        (raw_df['NAVSTAT'] == nav_stat_mapping['Under way sailing']) |
        (raw_df['NAVSTAT'] == nav_stat_mapping['Moored']) |
        (raw_df['NAVSTAT'].isna())
    ]

    # We will groupby day also, so that we do not have huge gaps to interpolate
    raw_df['DATE TIME (UTC)'] = pd.to_datetime(raw_df['DATE TIME (UTC)'])
    raw_df['day'] = raw_df['DATE TIME (UTC)'].dt.date

    grouped_df = raw_df\
        .set_index('DATE TIME (UTC)')\
        .groupby(['MMSI', 'day'])

    dfs = []
    all_time_diffs = []
    missing_count = 0
    above_interpolation_limit = 0
    mmsi_timestamps = {}

    for (mmsi, day), group_df in grouped_df:

        group_df = group_df\
            [['LATITUDE', 'LONGITUDE', 'SPEED', 'COURSE', 'HEADING']]\
            .drop_duplicates()

        # We define a track to be at least 50 measurements,
        #   and does not have a gap bigger than 2 hours
        time_diffs = group_df.reset_index()\
            ['DATE TIME (UTC)']\
            .diff(1)

        time_diffs = time_diffs / timedelta(minutes=1)
        all_time_diffs.extend( list(time_diffs) )

        # Skip, if the time difference is too big
        max_time_diff = max(all_time_diffs)
        if max_time_diff > GAP_LIMIT:
            continue

        # interpolate
        down_sample_seconds = DOWN_SAMPLE_MINUTES * 60
        group_df = group_df[['LATITUDE', 'LONGITUDE', 'SPEED', 'COURSE', 'HEADING']]\
            .resample(f'{down_sample_seconds}s')\
            .first()

        group_df['MMSI'] = mmsi
        group_df['day'] = day

        # At least 50, bcause input - 25 minutes, output - 25 minutes
        if group_df.dropna().shape[0] < 10:
            continue

        missing_count += np.sum( group_df.isnull() )
        group_df = group_df.dropna().reset_index()

        # Source: https://stackoverflow.com/questions/28773342/truncate-timestamp-column-to-hour-precision-in-pandas-dataframe
        group_df['hour'] = group_df['DATE TIME (UTC)'].dt.round('h')

        # merge with meteorological data
        group_df = group_df.apply(
            lambda row: merge_with_meteo(row, meteo_df, lat_lon_grid, lat_lon_tree),
            axis='columns'
        )
        group_df = group_df.drop(columns=['hour'])

        # create features such as dif and delta dif
        group_df, timestamps = feature_engineering(group_df)

        # Pasalinti NA ir atnaujinti timestamps pagal tai
        na_mask = group_df.isna().any(axis=1)

        group_df = group_df.loc[~na_mask, :].copy()
        timestamps = timestamps[~na_mask].copy()

        mmsi_timestamps[ (mmsi, day) ] = timestamps

        group_df['MMSI'] = int(mmsi)
        group_df['day'] = day
        dfs.append( group_df )

    df = pd.concat( dfs, ignore_index=True )
    df.to_csv(f'output/data_eagles_vezhen/{MODE}/df-prepared-eagle.csv', index=False)
    np.save('output/gaps2.npy', np.array(all_time_diffs))

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

    # https://stackoverflow.com/questions/41993565/save-minmaxscaler-model-in-sklearn
    with open(f'output/data_eagles_vezhen/{MODE}/min-max-scaler-eagles-vezhen.pkl', 'wb+') as f:
        pickle.dump( scaler, f)

    print(df.columns)

    # Prepare time series and concatenate
    # initialize arrays as null
    X_train, y_train, X_val, y_val, X_test, y_test = [None] * 6

    total_mmsis_final = 0

    train_groups, val_groups, test_groups = partition_groups( 
        df[['MMSI', 'day']].drop_duplicates().to_numpy().tolist()
    )
    train_counts, val_counts, test_counts = 0, 0, 0

    all_mmsis = []
    all_days = []

    val_mmsis = []
    val_days = []

    for (mmsi, day), mmsi_df in df.groupby(['MMSI', 'day']):

        # Pasalinti NA ir trackinti timestamps
        na_mask = mmsi_df.isna().any(axis=1)
        mmsi_df = mmsi_df.loc[~na_mask, :].copy()

        mmsi_timestamps[(mmsi, day)] = mmsi_timestamps\
            [(mmsi, day)]\
            [~na_mask].copy()

        mmsi_df.drop( columns=['MMSI', 'day'], inplace=True )

        mmsi_timestamps[(mmsi, day)] = mmsi_timestamps\
            [(mmsi, day)]\
            [~na_mask].copy()

        X_arr, y_arr = prepare_time_series(mmsi_df)

        if not X_arr.shape[0] >= 5:
            continue

        total_mmsis_final += 1

        # split the data
        if belongs_in( (mmsi, day), train_groups):

            train_counts += 1

            if X_train is None:
                X_train, y_train = X_arr, y_arr
            else:
                X_train = np.concatenate([X_train, X_arr], axis=0)
                y_train = np.concatenate([y_train, y_arr], axis=0)

        elif belongs_in((mmsi, day), val_groups):

            val_counts += 1

            if X_val is None:
                X_val, y_val = X_arr, y_arr
            else:
                X_val = np.concatenate([X_val, X_arr], axis=0)
                y_val = np.concatenate([y_val, y_arr], axis=0)

            val_mmsis.extend( [mmsi]*X_arr.shape[0] )
            val_days.extend( [day]*X_arr.shape[0] )

        else:

            test_counts += 1

            if X_test is None:
                X_test, y_test = X_arr, y_arr
            else:
                X_test = np.concatenate([X_test, X_arr], axis=0)
                y_test = np.concatenate([y_test, y_arr], axis=0)

            all_mmsis.extend( [mmsi]*X_arr.shape[0] )
            all_days.extend( [day]*X_arr.shape[0] )

    np.save(f'output/data_eagles_vezhen/{MODE}/X_train2.npy', X_train)
    np.save(f'output/data_eagles_vezhen/{MODE}/y_train2.npy', y_train)
    np.save(f'output/data_eagles_vezhen/{MODE}/X_val2.npy', X_val)
    np.save(f'output/data_eagles_vezhen/{MODE}/y_val2.npy', y_val)
    np.save(f'output/data_eagles_vezhen/{MODE}/X_test2.npy', X_test)
    np.save(f'output/data_eagles_vezhen/{MODE}/y_test2.npy', y_test)

    with open(f'output/data_eagles_vezhen/{MODE}/timestamps-dict2.npy', 'wb+') as f:
        pickle.dump(mmsi_timestamps, f)

    np.array(all_mmsis).dump(f'output/data_eagles_vezhen/{MODE}/test-mmsis2.npy')
    np.array(all_days).dump(f'output/data_eagles_vezhen/{MODE}/test-days2.npy')

    np.array(val_mmsis).dump(f'output/data_eagles_vezhen/{MODE}/val-mmsis2.npy')
    np.array(val_days).dump(f'output/data_eagles_vezhen/{MODE}/val-days2.npy')

    print('Data information:\n', '-'*20)
    print('Total tracks:', total_mmsis_final)

    print('Shapes of tensors:')
    print('\nTrain:')
    print(X_train.shape)
    print(y_train.shape)
    print('Number of ships:', train_counts)

    print('\nVal:')
    print(X_val.shape[0])
    print('Number of ships:', val_counts)

    print('\nTest:')
    print(X_test.shape[0])
    print('Number of ships:', test_counts)

    print('Total sequences:', X_train.shape[0] + X_val.shape[0] + X_test.shape[0])
