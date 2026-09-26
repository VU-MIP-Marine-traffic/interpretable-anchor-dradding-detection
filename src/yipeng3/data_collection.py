import requests
import gc
import zipfile
from io import BytesIO
import os
from datetime import date
from tqdm import tqdm
import pandas as pd

LAT_MIN, LAT_MAX = 54.5, 56.5
LON_MIN, LON_MAX = 13.0, 18.0
COLUMNS = [
    '# Timestamp',
    'MMSI',
    'Latitude',
    'Longitude',
    'Navigational status',
    'ROT',
    'SOG',
    'COG',
    'Heading',
    'Ship type',
    'Cargo type',
    'Width',
    'Length',
]

dates = pd.date_range(
    start=date(2025, 2, 27),
    end=date(2025, 3, 1)
)


if __name__ == "__main__":

    for data_date in tqdm(dates):

        year = data_date.year
        file_name = f"aisdk-{data_date.to_pydatetime().strftime('%Y-%m-%d')}"
        response = requests.get(f'http://aisdata.ais.dk/{file_name}.zip')
        
        if response.status_code == 404:
            response = requests.get(f'http://aisdata.ais.dk/{year}/{file_name}.zip')

        bytes_content = BytesIO(response.content)
        # https://discuss.python.org/t/trying-to-scrape-and-download-zipfiles/35399/2
        zip_file = zipfile.ZipFile(bytes_content)
        zip_file.extractall()

        del bytes_content

        gc.collect()

        df = pd.read_csv(
            f'{file_name}.csv',
            chunksize=2**20
        )

        os.makedirs(f'data/ais/{file_name}', exist_ok=True)
        for ix, df_chunk in enumerate(df):
            df_tmp = df_chunk[
                (df_chunk['Latitude'].between(LAT_MIN, LAT_MAX)) &
                (df_chunk['Longitude'].between(LON_MIN, LON_MAX)) &
                ((df_chunk['Ship type'] == 'Cargo') | (df_chunk['Ship type'] == 'Tanker')) &
                ((df_chunk['Navigational status'] == 'Under way using engine') |
                (df_chunk['Navigational status'] == 'Under way sailing'))
            ][COLUMNS].copy()
            df_tmp['Cargo type'] = df_tmp['Cargo type'].fillna('No additional information')
            df_tmp.to_csv(f'data/ais/{file_name}/ais-{ix}.csv', index=False)

        os.remove(f'{file_name}.csv')

        del df
        gc.collect()
