import pandas as pd
import numpy as np
import scipy
from pyproj import Geod
from tqdm import tqdm
import os
from datetime import timedelta

dfs = []

for dpath in tqdm(os.listdir('data/ais/')):

    dfs_tmp = []
    for file_path in os.listdir(f'data/ais/{dpath}'):
        dfs_tmp.append(pd.read_csv(f'data/ais/{dpath}/{file_path}'))
    df = pd.concat(dfs_tmp)

    df['# Timestamp'] = pd.to_datetime(df['# Timestamp'], dayfirst=True)
    df['day'] = df['# Timestamp'].dt.normalize().dt.date

    dfs_tmp = []
    gaps = []

    for (mmsi, day), df_group in df.groupby(['MMSI', 'day']):

        time_diffs = df_group.sort_values('# Timestamp')\
            ['# Timestamp']\
            .diff(1)
        time_diffs = time_diffs / timedelta(minutes=1)
        time_diffs = time_diffs.dropna().tolist()

        if not time_diffs:
            continue

        gaps.extend(time_diffs)

        df_group = df_group.drop_duplicates()\
            .set_index('# Timestamp')\
            .sort_index()\
            .resample('60s')\
            .first()\
            .reset_index()\
            .dropna(subset=['MMSI', 'Latitude', 'Longitude', 'Heading', 'SOG'])

        time_diffs = df_group.sort_values('# Timestamp')\
            ['# Timestamp']\
            .diff(1)
        time_diffs = time_diffs / timedelta(minutes=1)
        time_diffs = time_diffs.dropna().tolist()

        if not time_diffs:
            continue

        if max(time_diffs) >= 120:
            continue

        df_group['date_diff'] = df_group['# Timestamp'].diff(1) / timedelta(minutes=1)
        df_group["dLat/dt"] = ((df_group['Latitude'].shift(-1) - df_group['Latitude']) / (df_group['date_diff'] / 60)).fillna(0)
        df_group["dLong/dt"] = ((df_group['Longitude'].shift(-1) - df_group['Longitude']) / (df_group['date_diff'] / 60)).fillna(0)
        df_group["Latitude_next"] = df_group['Latitude'].shift(-1).fillna(df_group['Latitude'])
        df_group["Longitude_next"] = df_group['Longitude'].shift(-1).fillna(df_group['Longitude'])

        geod = Geod(ellps='WGS84')

        for index, row in df_group.iterrows():
            _, _, distance = geod.inv(row['Longitude'], row['Latitude'], row['Longitude_next'], row['Latitude_next'])
            df_group.loc[index, 'distance'] = distance

        if max(df_group['distance']) >= 800:
            continue

        df_group['MMSI'] = mmsi
        df_group['day'] = day 
        dfs_tmp.append(df_group)

    dfs.append(pd.concat(dfs_tmp))

df = pd.concat(dfs)
print( df.columns )

os.makedirs('data/prepared/baseline', exist_ok=True)

# Source - https://stackoverflow.com/a/77042000
# Posted by F.D.Castel
# Retrieved 2026-06-16, License - CC BY-SA 4.0
df.to_parquet(
    'data/prepared/baseline/dataset_raw_baseline.parquet',
    partition_cols=['day'],
    existing_data_behavior='delete_matching'
)
