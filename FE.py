import pandas as pd
import numpy as np
import os

from pandas.core import base
import sp_utils as ut
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from sklearn.neighbors import DistanceMetric

def dummy_fe():
    path = r'D:\McDonald\refactoring\StorePanelling\Russia\input\sample_fe'
    files = os.listdir(path)
    df = pd.concat([ut.load_flat_file(os.path.join(path, file_)) for file_ in files])
    return df


def neighbour_store_count(df_lat_lon, max_distance_km):
    """neighbour_store_count 

    To find number of stores present in x km of radius

    Parameters
    ----------
    df_lat_lon : pd.DataFrame
        'LATITUDE', 'LONGITUDE' of each store. (index of dataframe must be the store id)
    max_distance_km : float
        max distance in km

    Returns
    -------
    dict
        number of neighbouring stores in max_distance_km radius
    """
    df_lat_lon = df_lat_lon[['LATITUDE', 'LONGITUDE']].dropna()
    haversine = DistanceMetric.get_metric('haversine')
    dists = haversine.pairwise(df_lat_lon.values)
    # radius of earth = 6371 km
    df = 6371 * dists
    df = pd.DataFrame(df, index=list(df_lat_lon.index), columns=list(df_lat_lon.index))
    df = df.unstack().reset_index()
    df.columns = ['store1', 'store2', 'distance_in_km']
    df = df[df['distance_in_km']<=max_distance_km]
    return df.groupby(['store1'])['store2'].nunique().to_dict()