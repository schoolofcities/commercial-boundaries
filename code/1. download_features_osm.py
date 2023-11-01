#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import osmnx
import osmnx as ox
import os
import geopandas as gpd
import pandas as pd

def down_object(maxy,miny,maxx,minx,tags, geo_type, crs):
    objects = ox.features.features_from_bbox(maxy,miny,maxx,minx,tags)
    objects = objects.loc[objects.geometry.type==geo_type]
    objects = objects.to_crs('EPSG:'+str(crs))
    objects.set_crs={'EPSG:'+str(crs)}
    
    return objects

def overall_process(bound_data, city, crs):
    bound_data = bound_data.to_crs('EPSG:4326')
    bound_data.set_crs={'EPSG:4326'}
    
    minx, miny, maxx, maxy = bound_data.geometry.total_bounds
    
    all_retailer = gpd.GeoDataFrame()
    
    #download retail units
    crs = crs
   
    tag_list = ['restaurant', 'cafe', 'fast_food', 'bank', 'pharmacy', 'bar', 'pub', 
                'pub','post_office', 'marketplace', 'nightclub', 'bureau_de_change', 'food_court']
    
    for tag in tag_list:
        try:
            tags = {'amenity': tag}   
            retailers = down_object(maxy,miny,maxx,minx,tags, 'Point', crs)
            retailers = retailers.reset_index(drop = False)
            retailers = retailers[['osmid', 'geometry']]
            all_retailer = pd.concat([all_retailer, retailers])
            
            tags = {'amenity': tag}   
            retailers = down_object(maxy,miny,maxx,minx,tags, 'Polygon', crs)
            retailers = retailers.reset_index(drop = False)
            retailers = retailers[['osmid', 'geometry']]
            all_retailer = pd.concat([all_retailer, retailers])
        except:
            pass

        
    #shop_tag
    try:
        tags = {'shop': True}
        all_shops = down_object(maxy,miny,maxx,minx,tags, 'Point', crs)
        all_shops = all_shops.reset_index(drop = False)
        all_shops = all_shops[['osmid', 'geometry']]   
        
        all_retailer = pd.concat([all_retailer, all_shops])
        
        tags = {'shop': True}
        all_shops = down_object(maxy,miny,maxx,minx,tags, 'Polygon', crs)
        all_shops = all_shops.reset_index(drop = False)
        all_shops = all_shops[['osmid', 'geometry']]   
        
        all_retailer = pd.concat([all_retailer, all_shops])
    except:
        pass
    
    #set your save dir and filename for retailers
    retailer_dir = './retailers/' + city + '_retailers.parquet'
    
    all_retailer = all_retailer.drop_duplicates(subset=['osmid'])
    all_retailer.to_parquet(retailer_dir)

    #download retail buildings
    building_list = ['retail', 'supermarket']
    retail = gpd.GeoDataFrame()
    for building in building_list:
        try:
            tags = {'building': building}
            temp_retail = down_object(maxy,miny,maxx,minx,tags, 'Polygon', crs)
            temp_retail = temp_retail.reset_index(drop = False)
            temp_retail = temp_retail[['osmid', 'geometry']]
            
            retail = pd.concat([retail, temp_retail])
        except:
            pass
    
    retail = retail.drop_duplicates(subset=['osmid'])
    #set your save dir and filename for retail building
    retail_dir = './retail/'+ city + '_retail.parquet'
    retail.to_parquet(retail_dir)
    
    
    #download office
    building_list = ['office']
    office = gpd.GeoDataFrame()
    for building in building_list:
        try:
            tags = {'building': building}
            temp_office = down_object(maxy,miny,maxx,minx,tags, 'Polygon', crs)
            temp_office = temp_office.reset_index(drop = False)
            temp_office = temp_office[['osmid', 'geometry']]
            
            office = pd.concat([office, temp_office])
        except:
            pass
    
    tag_list = ['accountant', 'government', 'company', 'construction_company', 'consulting', 'coworking', 'diplomatic', 
                'employment_agency','financial', 'financial_advisor', 'government', 'it', 'lawyer', 'newspaper', 'ngo', 'tax_advisor']
    for tag in tag_list:
        try:
            tags = {'office': tag}   
            temp_office = down_object(maxy,miny,maxx,minx,tags, 'Point', crs)
            temp_office = temp_office.reset_index(drop = False)
            temp_office = temp_office[['osmid', 'geometry']]
            office = pd.concat([office, temp_office])
        
            temp_office = down_object(maxy,miny,maxx,minx,tags, 'Polygon', crs)
            temp_office = temp_office.reset_index(drop = False)
            temp_office = temp_office[['osmid', 'geometry']]
            office = pd.concat([office, temp_office])
        except:
            pass

    office = office.drop_duplicates(subset=['osmid'])
    #set your save dir and filename for office
    office_dir = './office/'+ city + '_office.parquet'
    office.to_parquet(office_dir)

   
os.chdir("put your base directory")
ma_america = gpd.read_parquet('put the given metropolitan area files (top_50_ma_us_ca.parquet)')


for x in ma_america.index:
    temp_ma = ma_america.loc[ma_america.index == x].copy()
    city = temp_ma['NAME'].tolist()[0].split(',')[0]
    city = city.replace(' ', '-')
    print(city)

    crs = 4326
    
    overall_process(temp_ma, city, crs)
