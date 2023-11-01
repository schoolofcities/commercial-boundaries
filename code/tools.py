#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:59:35 2023

@author: byeonghwa
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd

from tqdm import tqdm

from shapely.geometry import shape as sp
from shapely.geometry import MultiPolygon, Polygon


from rasterio.features import shapes
from rasterio.transform import from_bounds

import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.preprocessing import MinMaxScaler

import random

import gzip

from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def save_gz(out_nm, array):
    """
    Save the numpy array into gz file format

    Parameters
    ----------
    out_nm : str
        path to which files will be saved
    array : numpy array
        numpy array to save
    """
    
    try:
        with gzip.open(out_nm, 'wb') as f:
            np.save(f, array)
        print(f"Array saved to {out_nm}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

def load_gz(file_path):
    """
    Load the saved numpy array as gz file format

    Parameters
    ----------
    file_path : str
        path to which file was saved
    """
    
    try:
        with gzip.open(file_path, 'rb') as f:
            loaded_data = np.load(f)
            
        return loaded_data
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def count(bound_data, count_data, index_id, col_nm):
    """
    Count variables

    Parameters
    ----------
    bound_data : geoparquet
        base grid data
    count_data : geoparquet
        target variable data
    index_id : str
        the name of index id
    col_nm : str
        the name of variable
    """
    
    count_data['count'] = 1
    dfsjoin = gpd.sjoin(bound_data, count_data, predicate="intersects")
    dfpivot = pd.pivot_table(dfsjoin, index = index_id, values = 'count',aggfunc = 'sum')
    dfpivot[index_id] = dfpivot.index
    dfpivot = dfpivot[[index_id, 'count']]
    dfpivot = dfpivot.rename({'count': col_nm}, axis = 1)
    dfpivot = dfpivot.reset_index(drop = True)
    
    return dfpivot

def assign_variable(bound_data, count_data, col_nm):
    """
    Assign variables into grid

    Parameters
    ----------
    bound_data : geoparquet
        base grid data
    count_data : geoparquet
        target variable data
    col_nm : str
        the name of variable
    """
    
    count_pivot = count(bound_data, count_data, 'id', col_nm)
    bound_data = bound_data.merge(count_pivot, how = 'left', on = 'id')
    bound_data = bound_data.fillna(0)
    
    return bound_data

def convert_array(file_nm, var_list):
    """
    Convert the grid based geoparquet into numpy array

    Parameters
    ----------
    file_nm : str
        the name of geoparquet
    var_list : list
        the attributes of geoparquet data
    """
    bound_df = gpd.read_parquet(file_nm)
    
    start = datetime.now()    
    # define the grid size
    pixel_size = 50
    
    bbox = bound_df.total_bounds
    x_res = int((bbox[2] - bbox[0]) / pixel_size)
    y_res = int((bbox[3] - bbox[1]) / pixel_size)
    
    result_array = np.empty((len(var_list), y_res, x_res), dtype=np.float32)
    
    for k, var in enumerate(var_list):
        grid_array = np.zeros((y_res, x_res), dtype=np.float32)
    
        col_indices = ((bound_df['geometry'].bounds['minx'] - bbox[0]) / pixel_size).astype(int)
        row_indices = ((bbox[3] - bound_df['geometry'].bounds['maxy']) / pixel_size).astype(int)
    
        grid_array[row_indices, col_indices] = bound_df[var].values
    
        result_array[k] = grid_array
            
    end = datetime.now()
    print('Duration: {}'.format(end - start))
       
    return result_array

def local_minmax_transform(array):
    """
    Normalise the given array

    Parameters
    ----------
    array : numpy array
        the given array

    """
    
    mm= MinMaxScaler(feature_range=(0.5, 1))
    
    #create empty array based on the input array size
    scaled_np = np.empty_like(array)
    
    for i in range(array.shape[0]):
        slice_flat = array[i].flatten()
        
        non_zero_indices = np.nonzero(slice_flat)
        
        scaled_non_zero = mm.fit_transform(slice_flat[non_zero_indices].reshape(-1, 1))
        
        scaled_slice_flat = np.zeros_like(slice_flat, dtype=float)
        scaled_slice_flat[non_zero_indices] = scaled_non_zero.flatten()
        
        scaled_slice = scaled_slice_flat.reshape(array[i].shape)
        scaled_np[i] = scaled_slice
        
    return scaled_np
      
def random_crop(input_array, width, height):
    """
    Conduct crop

    Parameters
    ----------
    input_array : numpy array
        the given array
    width : int
        the width of grid
    height : int
        the width of height
    """
    
    assert input_array.shape[-2] >= height
    assert input_array.shape[-1] >= width
    x = random.randint(0, abs(input_array.shape[-1] - width))
    y = random.randint(0, abs(input_array.shape[-2] - height))
    input_array = input_array[:,y:y+height, x:x+width]
    return input_array

def crop_array(array, num_example, window_size):
    """
    Randomly crop array

    Parameters
    ----------
    array : numpy array
        the given array
    num_example : int
        the number of creating crop array
    window_size : int
        the size of array
    """
    
    i = 0    
    for x in range(num_example):
        temp_crop = random_crop(array,window_size,window_size)
        temp_crop = np.expand_dims(temp_crop, axis=0)
        if i == 0:
            ran_crop = temp_crop.copy()
            i+=1
        else:
            ran_crop = np.vstack([ran_crop,temp_crop])
    return ran_crop
  

def reclustering(training_data_path, out_array_path, nlabel):
    """
    Exploring reclustering result

    Parameters
    ----------
    training_data_path : str
        the path in which training data are located
    out_array_path : str
        the path in which clustering result data are located
    nlabel : int
        nlabel value
    """
    
    column_list = ['cluster', 'retailers', 'office', 'jobs']
    k = 0
    for (path, dirs, files) in os.walk(out_array_path):
        for fname in files:
            ext = os.path.splitext(fname)[-1]
            if (ext == '.gz'):
                clu_file = "%s/%s" % (path, fname)
                print(clu_file)
                city = fname.split('_')[2].split('.')[0]
                input_file = "%s/array_%s.npy.gz" % (training_data_path, city)
                
                cluster_np = load_gz(clu_file).flatten()
                input_array = load_gz(input_file)
                input_array = input_array[(0,1,2),:,:]
                
                for x in range(input_array.shape[0]):
                    temp_np = input_array[x].flatten()
                    cluster_np = np.column_stack([cluster_np, temp_np])
                
                
                temp_df = pd.DataFrame(cluster_np, columns = column_list)
                temp_df = temp_df.loc[(temp_df[column_list[1]] > 0) | (temp_df[column_list[2]] > 0)]


                if k == 0:
                    agg_df = temp_df.copy()
                    k+=1
                else:
                    agg_df = pd.concat([agg_df,temp_df])


    var_list = ['retailers', 'office', 'jobs']
    
    recluster_dic = ac_recluster(agg_df, var_list, nlabel)
    
    return recluster_dic

def ac_recluster(agg_df, column_list, nlabel):
    """
    perform agglomerative clustering

    Parameters
    ----------
    agg_df : pandas dataframe
        pandas dataframe containing cluster and their attributes
    column_list : list
        list containing the variables
    nlabel : int
        nlabel value

    """
    pivot_df = pd.pivot_table(agg_df, index='cluster', aggfunc='sum')

    #find optimal number of clusters
    sill_score = {}
    for k in range(2, len(pivot_df)):
        clustering = AgglomerativeClustering(n_clusters = k).fit(pivot_df)
        labels = clustering.labels_

        score = silhouette_score(pivot_df, labels, metric='euclidean')
        sill_score[k] = score
        
    x_values = list(sill_score.keys())
    y_values = list(sill_score.values())
    plt.figure(figsize=(8,5))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color = 'orange')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Scores')
    plt.tight_layout()
    plt.show()
    
    print(sorted(sill_score.items(), key=lambda x:x[1], reverse=True))
    
    opt_k = max(sill_score, key = sill_score.get)
    
    #perform agglomerative clustering based on the optimal k
    clustering = AgglomerativeClustering(opt_k).fit(pivot_df)
    print(pivot_df)
    
    #draw dendrogram    
    linkage_matrix = linkage(pivot_df, "ward")
    
    plt.figure(figsize=(8, 6))
    dendrogram(linkage_matrix, orientation="right", labels = np.array(pivot_df.index).astype(int).tolist(), distance_sort='descending')
    plt.title('Agglomerative Clustering Dendrogram')
    plt.tight_layout()
    plt.show()
    
    
    #reclustering result
    pivot_df['recluster'] = clustering.labels_

    pivot_df = pivot_df.reset_index(drop = False)

    #create dict key is recluster label and values are the list of previous cluster
    #belonging to the recluster label
    recluster_dict = {}
    for x in range(opt_k):
        test_recluster_list = pivot_df.loc[pivot_df['recluster'] == x]['cluster'].tolist()
        recluster_dict[x] = test_recluster_list
    
    return recluster_dict
                    
def recluster_and_assign_att(input_file, city, recluster_dict):
    """
    This function is to update the cluster into the recluster result
    and to assign the variables (office, retail and jobs) into the recluster result
    
    Parameters
    ----------
    input_file : str
        input file path
    city : str
        the name of city
    recluster_dict : dictionary
        the recluster dictionary containing the information about previous cluster
        and recluster labels

    """
    var_list = ['retailers', 'office', 'jobs']
    #create reclustering array
    from_array = load_gz(input_file)
    exist_list = [item for recl_list in recluster_dict.values() for item in recl_list]
    not_mask = ~np.isin(from_array, exist_list)
    from_array[not_mask] = -1
    input_array = from_array.copy()
    for x, recl_list in recluster_dict.items():
        mask = np.isin(from_array, recl_list)
        input_array[mask] = x
        exist_list.extend(recl_list)
   
    #convert the reclustering array into geoparquet
    #read base training grid
    tg_path = './training_data/MA_training_data/' + city + '_training_grid.parquet'
    df = gpd.read_parquet(tg_path)    
    
    minx, miny, maxx, maxy = df.geometry.total_bounds
  
    transform = from_bounds(minx, miny, maxx, maxy, input_array.shape[1], input_array.shape[0])
    
    preds_grid = gpd.GeoDataFrame()
    for vec in shapes(input_array.astype(np.int16),transform=transform):
        if vec[1] >= 0:
            temp_grid = gpd.GeoDataFrame(geometry=[MultiPolygon([sp(vec[0])])])
            temp_grid['cluster'] = vec[1]
            preds_grid = pd.concat([preds_grid, temp_grid])
    
    preds_grid = preds_grid.reset_index(drop = True)
    preds_grid.crs = df.crs
    preds_grid = preds_grid.reset_index(drop = False)
    
    #remove single cell
    preds_grid = preds_grid[(preds_grid.geometry.apply(lambda x: x.area) > 2500)]
    
    preds_grid = preds_grid.drop('index', axis = 1)
    preds_grid = preds_grid.reset_index(drop = True)
    preds_grid = preds_grid.reset_index(drop = False)
    merged_grid = preds_grid.copy()
       
    #create directory
    os.makedirs('./result/MA_output_cluster/'+input_file.split('_')[-2], exist_ok=True)

    #save reclustering result as geoparquet format
    out_cluster = './result/MA_output_cluster/'+input_file.split('_')[-2]+'/MA_output_'+city+'.parquet'
    merged_grid.to_parquet(out_cluster, index = False)

    #assign attributes into the reclustering results
    df['geometry'] = df['geometry'].centroid

    inter = gpd.sjoin(merged_grid, df, how='left', predicate='intersects')

    sum_attributes = inter.groupby(['index', 'cluster'])[var_list].sum().reset_index()
    merged_grid = merged_grid.merge(sum_attributes, on=['index', 'cluster'], how='left')
    
    #create directory
    os.makedirs('./result/MA_output_grid/'+input_file.split('_')[-2], exist_ok=True)
    #save reclustering with attributes data as geoparquet format
    out_grid = './result/MA_output_grid/'+input_file.split('_')[-2]+'/MA_output_attri_' + city+'.parquet'
    merged_grid.to_parquet(out_grid, index = False)


def identify_commercial_boundary(input_file, city, commercial_cluster):
    """
    Identify commercial boundary from the recluster results

    Parameters
    ----------
    input_file : str
        input file path
    city : str
        the name of city
    
    commercial_cluster : int
        the identified cluster as commercial boundary based on the result of reclustering
    """
    
    #read reclustering results
    att_df = gpd.read_parquet(input_file)
    
    os.makedirs('./result/MA_commercial_boundary/'+input_file.split('/')[-2], exist_ok=True)
    
    cb = att_df[
        (att_df['cluster'] == commercial_cluster) & 
        ((att_df['retailers'] >= 5) | (att_df['office'] >= 5)) & 
        (att_df.geometry.apply(lambda x: x.area) > 2500)
    ][['index', 'cluster', 'geometry']]
    
    cb = cb[['index', 'geometry']]
    #save directory
    out_cb = './result/MA_commercial_boundary/'+input_file.split('/')[-2]+'/'+city+'_commercial_boundary.parquet'
    cb.to_parquet(out_cb, index = False)


    