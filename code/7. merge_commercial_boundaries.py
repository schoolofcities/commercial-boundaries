#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from datetime import datetime

#the path in which identified commercial boundaries are lcoated
cb_path = 'put your path'

#read the commercial boundaires of all metropolitan areas into one file
all_cb= gpd.GeoDataFrame()
for (path, dirs, files) in os.walk(cb_path):
    for fname in files:
        ext = os.path.splitext(fname)[-1]
        if (ext == '.parquet'):
            input_file = "%s/%s" % (path, fname)
            print(input_file)
            temp_cb = gpd.read_parquet(input_file)
            temp_cb = temp_cb.drop(['out_bound', 'out_bound_mean'], axis =1)
            
            all_cb = pd.concat([all_cb,temp_cb])
            
all_cb = all_cb.reset_index(drop = True)
all_cb['Area(m2)'] = all_cb.geometry.area

#remove the smaller polygons among the overlapped ones
start = datetime.now()          
all_cb.geometry = all_cb.geometry.buffer(-0.1)
to_remove = []
for i, row in tqdm(all_cb.iterrows()):
    for j, other_row in all_cb.iterrows():
        if i != j and row.geometry.intersects(other_row.geometry):

            if row.geometry.area < other_row.geometry.area:
                to_remove.append(i)
                break
end = datetime.now()
print('Duration: {}'.format(end - start))
all_cb = all_cb.drop(to_remove)

all_cb.geometry = all_cb.geometry.buffer(0.1)
all_cb = all_cb.reset_index(drop = True)
all_cb = all_cb.reset_index(drop = False)
all_cb = all_cb.rename({'index':'CB_ID'}, axis = 1)
#save the merged commercial boundaries
out_nm = 'put your path'
all_cb.to_parquet(out_nm)
