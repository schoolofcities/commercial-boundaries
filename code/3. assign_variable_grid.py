#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import geopandas as gpd
import pandas as pd
from datetime import datetime
from tools import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

os.chdir('put your path')
job_path = './jobs'
bound_path = './bound'
grid_path = './MA_grid_bound'

jobs_df = gpd.GeoDataFrame()

for (path, dirs, files) in os.walk(job_path):
    for fname in files:
        ext = os.path.splitext(fname)[-1]
        if (ext == '.parquet'):
            input_file = "%s/%s" % (path, fname)
            temp_jobs = gpd.read_parquet(input_file)
            try:
                temp_jobs = temp_jobs.rename({'v_CA16_5774..Worked.at.usual.place':'jobs'},
                                             axis = 1)    
            except:
                pass
            temp_jobs['area'] = temp_jobs['geometry'].area/10**6
            temp_jobs = temp_jobs[['jobs', 'area', 'geometry']]
            jobs_df = pd.concat([jobs_df, temp_jobs])
         
jobs_df['jobs'] = jobs_df['jobs']/jobs_df['area']
jobs_df = jobs_df[['jobs', 'geometry']]


col_list = ['retailers', 'office', 'retail']

for (path, dirs, files) in os.walk(grid_path):
    for fname in files:
        ext = os.path.splitext(fname)[-1]
        if (ext == '.parquet'):
            input_file = "%s/%s" % (path, fname)
            city = fname.replace('.parquet', '').split('_')[-1]
            
            #out directory and file name
            out_nm = './training_data/MA_training_data/' + city+'_training_grid.parquet'

            print(city)
            start = datetime.now()
            bound_data = gpd.read_parquet(input_file)
            bound_data = bound_data[['id', 'geometry']]
            
            spatial_index = bound_data.sindex
            bound_data.has_sindex
            
            for col_nm in col_list:
                target_dir = './' + col_nm + '/' + city + '_' + col_nm +'.parquet'
                count_data = gpd.read_parquet(target_dir)
                bound_data = assign_variable(bound_data, count_data, col_nm)
            
            bound_data['retailers'] = bound_data['retailers'] + bound_data['retail']
            bound_data = bound_data.drop('retail', axis = 1)
                
            dfsjoin = gpd.sjoin(bound_data, jobs_df, op="intersects")
            dfpivot = pd.pivot_table(dfsjoin, index = 'id', values='jobs',aggfunc = 'sum')
            dfpivot = dfpivot.reset_index(drop = False)
            dfpivot = dfpivot[['id', 'jobs']]
                        
            bound_data = bound_data.merge(dfpivot, how = 'left', on = 'id')
            bound_data = bound_data.fillna(0)
                                        
            bound_data.to_parquet(out_nm, index = False)
            end = datetime.now()
            print('Duration: {}'.format(end - start))

