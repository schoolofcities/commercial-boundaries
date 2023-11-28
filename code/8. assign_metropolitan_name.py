#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import geopandas as gpd

msa_df = gpd.read_parquet('./top_50_ma_us_ca.parquet')
cb_df = gpd.read_file('./MA_commercial_boundary/Commercial_boundaries.gpkg')
msa_df = msa_df.to_crs('EPSG:3857')
cb_df = cb_df.to_crs('EPSG:3857')

cb_df = cb_df[['CB_ID', 'Area(m2)', 'geometry']]

inter = gpd.overlay(cb_df, msa_df, how='intersection')
inter['area'] = inter.geometry.area
inter.sort_values(by='area', inplace=True)
inter.drop_duplicates(subset='CB_ID', keep='last', inplace=True)
inter.drop(columns=['area'], inplace=True)

out_bound_id = list(inter['CB_ID'].unique().tolist())

out_df = cb_df.loc[~cb_df['CB_ID'].isin(out_bound_id)]

out_bound_inter = out_df.sjoin_nearest(msa_df)
out_bound_inter = out_bound_inter.drop('index_right', axis = 1)

assign_name = pd.concat([inter, out_bound_inter])
assign_name = assign_name.reset_index(drop = True)

assign_name = gpd.GeoDataFrame(assign_name, geometry = 'geometry')

assign_name = assign_name.rename({'NAME':'MSA_NAME',
                                  'STATE':'STATE/PROVINCE'}, axis = 1)

if len(cb_df) == len(assign_name):
    print('process worked well!')
else:
    print('something wrong')

out_nm = 'put your out name'
assign_name.to_file(out_nm, driver = 'GPKG')