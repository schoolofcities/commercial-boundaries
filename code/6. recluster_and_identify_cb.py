#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pandas as pd

import datetime

os.chdir('/Users/byeonghwa/Desktop/all/retail_boundary/github_commit/code')
from tools import *

#nlabel from the training result
nlabel = 58

training_data_path = './data/MA_base_training_array'
out_array_path = './result/MA_output_array/'+str(nlabel)

#conduct reclustering and create reclustering dictionary
recluster_dict = reclustering(training_data_path, out_array_path, nlabel)


for (path, dirs, files) in os.walk(out_array_path):
    for fname in files:
        ext = os.path.splitext(fname)[-1]
        if (ext == '.gz'):
            input_file = "%s/%s" % (path, fname)
            city = fname.split('_')[2].split('.')[0]
            start = datetime.now()
            print(city)
            
            #create recluster results
            recluster_and_assign_att(input_file,city,recluster_dict)
            
            #identify commercial boundary from the recluster results
            commercial_bound_recluster = 1 #the identified cluster as commercial boundary based on the result of reclustering
            out_nm = './result/MA_output_grid/'+str(nlabel)+'/MA_output_attri_' + city+'.parquet'
            identify_commercial_boundary(out_nm, city, commercial_bound_recluster)
            
            end = datetime.now()
            print('Duration: {}'.format(end - start))


