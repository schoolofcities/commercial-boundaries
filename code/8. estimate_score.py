#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:30:36 2024

@author: byeonghwa
"""

import numpy as np
import geopandas as gpd
import os
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

cb = gpd.read_parquet('put_your_merged_cb_results')

cb = cb[['index', 'cluster', 'geometry', 'retailers', 'office', 'jobs','area']]

sel_list = ['retailers', 'office', 'jobs']

for sel in sel_list:
    cb[sel] = cb[sel].apply(lambda x: np.log1p(x))

ss = StandardScaler()
cb[sel_list] = ss.fit_transform(cb[sel_list])

pca = PCA(n_components=1)
pca.fit(cb[sel_list])
cb['commercial_score'] = pca.transform(cb[sel_list])
print('pca_ratio', pca.explained_variance_ratio_)
print('pca_value', pca.singular_values_)

plt.figure(figsize=(10,6))
plt.hist(cb['commercial_score'], bins=20, color = 'Orange', alpha=0.5)
plt.xlabel('Commercial Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("put_your_output_figure", dpi=400)
plt.show()

cb = cb[['index', 'geometry', 'area', 'commercial_score']]

cb.to_parquet('put_your_processed_cb_results_dir')
