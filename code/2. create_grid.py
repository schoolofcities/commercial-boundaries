#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 

import os
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
from tqdm import tqdm
from datetime import datetime

os.chdir("put your base directory")
ma_america = gpd.read_parquet('put the given metropolitan area files (top_50_ma_us_ca.parquet)')
ma_america = ma_america.to_crs('EPSG:3857')

for x in ma_america.index:
    temp_ma = ma_america.loc[ma_america.index == x].copy()
    nm = temp_ma['NAME'].tolist()[0]
    
    out_grid_nm = './MA_grid_bound/base_grid_'+nm+'.parquet'
    
    if os.path.isfile(out_grid_nm):
        pass
    else:
        print(nm)
        start = datetime.now()
        width = 50
        height = 50
    
        xmin, ymin, xmax, ymax = temp_ma.total_bounds
        rows = int(np.ceil((ymax-ymin) /  height))
        cols = int(np.ceil((xmax-xmin) / width))
        XleftOrigin = xmin
        XrightOrigin = xmin + width
        YtopOrigin = ymax
        YbottomOrigin = ymax- height
        polygons = []
        for i in tqdm(range(cols)):
            Ytop = YtopOrigin
            Ybottom =YbottomOrigin
            for j in range(rows):
                polygons.append(Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)])) 
                Ytop -= height
                Ybottom -= height
            XleftOrigin += width
            XrightOrigin += width
            
        grid = gpd.GeoDataFrame({'geometry':polygons})
        grid = grid.reset_index(drop = False)
        grid.columns = ['id', 'geometry']
        grid.crs = 'EPSG:3857'
        
        #set the dir and file name of grid output
        out_grid_nm = './MA_grid_bound/base_grid_'+nm+'.parquet'
        grid.to_parquet(out_grid_nm)
        
        end = datetime.now()
        print('Duration: {}'.format(end - start))
