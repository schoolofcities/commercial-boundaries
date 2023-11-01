#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import geopandas as gpd
from datetime import datetime
from tools import *

#the path in which training grids are located
training_data_path = 'put your path'

var_list = ['retailers', 'office', 'jobs']

for (path, dirs, files) in os.walk(training_data_path):
    for fname in files:
        ext = os.path.splitext(fname)[-1]
        if (ext == '.parquet'):
            input_file = "%s/%s" % (path, fname)
            city = fname.split('_')[0]
            print(input_file)
            

            start = datetime.now()
            fin_out = convert_array(input_file, var_list)
            
            np_nm = './data/MA_base_training_array/array_' + city+'.npy.gz'
            save_gz(np_nm, fin_out)
            end = datetime.now()
            print('Duration: {}'.format(end - start))

