#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 21:23:21 2024

@author: byeonghwa
"""


import os
import geopandas as gpd
from geopy.geocoders import Nominatim
from datetime import datetime

def get_street_name(row):
    try:
        location = geolocator.reverse(row['coordinates'])
        address = location.raw.get('address', {})
        country = address['country']

        if country == 'Canada':
            address_string = ', '.join([address[key] for key in keys_ca if key in address])
        elif country == 'United States':
            address_string = ', '.join([address[key] for key in keys_us if key in address])
        else:
            address_string = 'error'

        return address_string
    except Exception as e:
        return None

os.chdir('put_your_working_directory')

keys_us = ['road', 'hamlet','city', 'state', 'country']
keys_ca = ['road', 'neighbourhood','city', 'state', 'country']

base_df = gpd.read_file('put_your_commercial_boundary_file')
target_df = base_df.copy()

target_df['geometry'] = target_df['geometry'].centroid
target_df = target_df.to_crs('EPSG:4326')

target_df['coordinates'] = target_df.apply(lambda x: (x['geometry'].y, x['geometry'].x), axis=1)

start = datetime.now()

geolocator = Nominatim(user_agent='by', timeout = 10)

target_df['address'] = target_df.apply(get_street_name, axis=1)
    
end = datetime.now()
print('Duration: {}'.format(end - start))

base_df['address'] = target_df['address']

base_df.to_file('put_your_output_name', driver = 'GPKG')
