""" Meteorological data acquisition for Yi-Peng 3 data and AIS data around it """

import requests
import pandas as pd
import numpy as np
from datetime import date

from datetime import datetime, timedelta
import os
import traceback

from dotenv import load_dotenv

from time import sleep
import json
from itertools import product

from src.yipeng3.data_collection import (
    LON_MIN,
    LON_MAX,
    LAT_MIN,
    LAT_MAX,
)

load_dotenv()
API_KEY = os.getenv("API_KEY")

# Metreological data column names for importing
DEFAULT_COLS = [
    "currentDirection", 
    "currentSpeed", 
    "gust",
    "swellDirection",
    "swellHeight",
    "waveDirection",
    "waveHeight",
    "windDirection",
    "windSpeed",
]

# Specify what organizations use for what meteo features
ORGANIZATIONS = {
    'currentDirection': 'meto',
    'currentSpeed': 'meto',
    'gust': 'noaa',
    'swellDirection': 'meteo',
    'swellHeight': 'meteo',
    'waveDirection': 'fmi',
    'waveHeight': 'fmi',
    'windDirection': 'noaa',
    'windSpeed': 'noaa',
}

# The definition of the grid. In this case it is 40x40 grid

dates = [
    dt.to_pydatetime()
    for dt in pd.date_range(start=date(2024,11,1),end=date(2025,3,1))
]
N_DATES = len(dates)

def get_stormglass_req(
    lon: float,
    lat: float,
    timestamp_start,
    timestamp_end,
    cols: list = DEFAULT_COLS
):
    """ Source: https://docs.stormglass.io/#/authentication """

    try:
        response = requests.get(
            'https://api.stormglass.io/v2/weather/point',
            params={
                'lat': f"{lat:.6f}",
                'lng': f"{lon:.6f}",
                'params': ','.join(cols),
                'start': timestamp_start,
                'end': timestamp_end
            },
            headers={
                'Authorization': API_KEY
            }
        )

    except:
        traceback.print_exc()
        return None

    # Handle the case when it breaks
    if response.status_code != 200:
        print(response.content)
        print("Failed to get resources with response code:", response.status_code)
        print(response.url)
        return None

    return response.content

def parse_response(response):

    res = json.loads(response)

    result = []
    for data in res['hours']:

        # At this point each feature has different organizations
        # For each feature pick a single organization
        parsed_data = pick_organizations(data)

        parsed_data["Longitude"] = res['meta']['lng'] 
        parsed_data["Latitude"] = res['meta']['lat'] 

        result.append(parsed_data)

    return result

N_CALLS = 0
def get_queries(
    day: datetime, 
    coordinate_grid: list
):
    stormglass_responses = []

    # Get responses
    for ix, (lat, lon) in enumerate(coordinate_grid):

        # Go day by day
        ts_min = day
        ts_max = day + timedelta(days=1)

        print(f"Get (longitude: {lon}, latitude: {lat}) for {ts_min}")

        response = get_stormglass_req(lon, lat, ts_min, ts_max)

        # Sleep for 20 seconds, if it fails and try again. stormglass does send 5XX errors a lot
        if response is None:
            print("Response count", ix+1)
            print(f"Didn't get ({lat, lon}) for {ts_min}")

            sleep(20)
            response = get_stormglass_req(lon, lat, ts_min, ts_max)

            if response is None:
                print('Failed to get data')
                break

        stormglass_responses.append(response)

    return stormglass_responses


def pick_organizations(data_different_organizations: dict) -> dict:

    output: dict = {}

    for measure, values_organizations in data_different_organizations.items():
        if measure != 'time':
            try:
                organization = ORGANIZATIONS[measure]
                output[measure] = values_organizations[organization]

            except KeyError:
                print(measure, values_organizations)

        else:
            time = datetime.strptime(
                values_organizations.split('+')[0],
                "%Y-%m-%dT%H:%M:%S"
            )
            output[measure] = time

    return output


def get_weather_df(
    dates: list,
    coordinate_grid: list,
    file_path: str
):

    stormglass_responses = []

    for day in dates:
        day_responses = get_queries(day, coordinate_grid)
        stormglass_responses.extend( day_responses )

    # Parse responses
    weather_list = []
    for weather_data_point in stormglass_responses:
        parsed_json = parse_response(weather_data_point)
        weather_list.append( parsed_json )

    weather_df = pd.concat([pd.DataFrame(dt) for dt in weather_list])
    weather_df.to_csv(file_path)

    return weather_df


def construct_coordinate_grid(step=0.5) -> list:
    lat_arr = np.arange(LAT_MIN, LAT_MAX+step, step=step)
    lon_arr = np.arange(LON_MIN, LON_MAX+step, step=step)

    return list(product(lat_arr, lon_arr))


if __name__ == "__main__":

    lat_lon_grid = construct_coordinate_grid(step=0.5)
    get_weather_df(dates, lat_lon_grid, 'data/weather_df_yipeng3_v2.csv')
