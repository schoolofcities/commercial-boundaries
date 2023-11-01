#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import geopandas as gpd

msa_df = gpd.read_parquet('./top_50_ma_us_ca.parquet')
cb_df = gpd.read_file('.//MA_commercial_boundary/Commercial_boundaries.gpkg')

msa_df = msa_df.to_crs('EPSG:3857')
cb_df = cb_df.to_crs('EPSG:3857')

assign_name = cb_df.sjoin_nearest(msa_df)
assign_name = assign_name.drop('index_right', axis = 1)

out_nm = 'put your out name'
assign_name.to_file(out_nm, driver = 'GPKG')