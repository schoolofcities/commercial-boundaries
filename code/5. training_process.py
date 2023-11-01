#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from tools import *
import random
from tqdm import tqdm

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import DataLoader
import model


device = torch.device("cuda:0" if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else "cpu")
print(device)

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


training_data_path = './data/MA_base_training_array'

num_example = 1000
window_size=100
batch_size=32


###############################################################################
#create random crop
mm = MinMaxScaler(feature_range=(10, 1000))

size_dict = {}

i = 0
for (path, dirs, files) in os.walk(training_data_path):
    for fname in files:
        ext = os.path.splitext(fname)[-1]
        if (ext == '.gz'):
            input_file = "%s/%s" % (path, fname)
            print(input_file)
            city = fname.split('_')[-1].split('.')[0]

            size_dict[city] = os.path.getsize(input_file)

size_df = pd.DataFrame([size_dict]).T
size_df.columns = ['size']

size_df[['size']] = mm.fit_transform(size_df[['size']])
size_df['size'] = size_df['size'].astype(int)

size_dict = size_df.to_dict()['size']
print(size_dict)

i = 0
for (path, dirs, files) in os.walk(training_data_path):
    for fname in files:
        ext = os.path.splitext(fname)[-1]
        if (ext == '.gz'):
            input_file = "%s/%s" % (path, fname)
            print(input_file)
            city = fname.split('_')[-1].split('.')[0]
            fin_out = load_gz(input_file)
            fin_out = local_minmax_transform(fin_out)
            print(fin_out.shape)

            temp_crop = crop_array(fin_out, size_dict[city], window_size)

            if i == 0:
                ran_crop = temp_crop.copy()
                i += 1
            else:
                ran_crop = np.vstack([ran_crop, temp_crop])

print(ran_crop.shape)
#save path and name of random crop array
out_nm = './data/random_crop/local_minmax_random_crop_weight_100.npy.gz'
save_gz(out_nm, ran_crop)

##############################################################################
#create tensor
data = torch.Tensor(ran_crop).to(device)
print(data.shape)

training_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
##############################################################################
#train model

#hyperparameters
input_dim = data.shape[1]
nChannel = 64
lr_list = [0.1,0.01,0.001]
stepsize_sim = 1
step_con_list = [1,5,10,50]
epochs_list = [3,4,5]
minLabels=3

#overall result of grid search
results = {}

#conduct grid search
for epochs in epochs_list:
    for learning_rate in lr_list:
        for stepsize_con in step_con_list:
            # similarity loss definition
            loss_fn = torch.nn.CrossEntropyLoss()
            # continuity loss definition
            loss_hpy = torch.nn.L1Loss(size_average = True)
            loss_hpz = torch.nn.L1Loss(size_average = True)
            
            HPy_target = torch.zeros(ran_crop.shape[-2]-1, ran_crop.shape[-1], nChannel, batch_size).to(device)
            HPz_target = torch.zeros(ran_crop.shape[-2], ran_crop.shape[-1]-1, nChannel, batch_size).to(device)
            
            model_net = model.MyNet(input_dim, nChannel).to(device)
            
            optimizer = torch.optim.SGD(model_net.parameters(), lr=learning_rate, momentum=0.9)
            
            a = 0
            repeat = 0
            
            
            losses_train = []
            for e in range(epochs):
                for batch_idx, data in enumerate(tqdm(training_loader)):
                    optimizer.zero_grad()
                    output = model_net(data).to(device)
                    output= output.permute(2, 3, 1,0).contiguous().view(-1, nChannel)
            
                    outputHP = output.reshape((data.shape[-2], data.shape[-1], nChannel, batch_size))
                    HPy = outputHP[1:, :, :, :] - outputHP[0:-1, :, :, :]
                    HPz = outputHP[:, 1:, :, :] - outputHP[:, 0:-1, :, :]
                    lhpy = loss_hpy(HPy,HPy_target)
                    lhpz = loss_hpz(HPz,HPz_target)
            
                    ignore, target = torch.max( output, 1 )
                    im_target = target.data.cpu().numpy()
                    nLabels = len(np.unique(im_target))
            
                    loss = stepsize_sim * loss_fn(output, target) + stepsize_con * (lhpy + lhpz)
            
                    loss.backward()
                    optimizer.step()
                    losses_train.append(loss.item())
            
                    if nLabels <= minLabels:
                        print ("nLabels", nLabels, "reached minLabels", minLabels, ".")
                        break
            
                print (e, '/', epochs, '|', ' label num :', nLabels, ' | loss :', loss.item())
                if a == 0:
                    repeat_labels = nLabels
                    a+=1
                else:
                    if repeat_labels == nLabels:
                        repeat += 1
                    else:
                        repeat_labels = nLabels
                        repeat = 0
            
                print(repeat)
                print(repeat_labels)
                if repeat == 2:
                    break
            
            print('Saving results')
            #save the trained model
            model_weights_svname = './saved/MA_colab_model_weight_'+str(nLabels)+'_'+str(stepsize_con)+'.pth'
            torch.save(model_net.state_dict(), model_weights_svname)
            
            #save the loss of trained model
            loss_df = pd.DataFrame(losses_train, columns = ['loss'])
            loss_df.to_csv('./saved/MA_colab_training_los_'+str(nLabels)+'_'+str(stepsize_con)+'.csv', index = False, sep = ',')
            
            results[(epochs, learning_rate, stepsize_con, nLabels)] = loss.item()

#create pandas dataframe based on results
results_df = pd.DataFrame.from_dict(
    results, orient='index', columns=['Loss'])
results_df = results_df.reset_index(drop=False)
results_df = results_df.rename({'index': 'parameters'}, axis=1)

results_df['Loss_rank'] = results_df['RMSE'].rank(method='min', ascending=True)
results_df = results_df.sort_values(by='Loss_rank', ascending=True)
results_df = results_df.reset_index(drop=True)

final_parameter = results_df.at[0, 'parameters']

##############################################################################
#prediction

nLabels = final_parameter[-1]
for (path, dirs, files) in os.walk(training_data_path):
    for fname in files:
        ext = os.path.splitext(fname)[-1]
        if (ext == '.gz'):
            input_file = "%s/%s" % (path, fname)
            city = fname.split('_')[1].split('.')[0]
            #set the dir and name of out array(clustering result) name
            out_nm = './result/MA_output_array/'+str(nLabels)+'/MA_output_'+city+'_'+str(nLabels)+'_'+str(stepsize_con)+'.gz'

            print(input_file)
            model_net = model.MyNet(input_dim, nChannel).to(device)
            model_net.load_state_dict(torch.load('./saved/MA_colab_model_weight_'+str(nLabels)+'_'+str(stepsize_con)+'.pth'))

            fin_out = load_gz(input_file)
            fin_out = local_minmax_transform(fin_out)
            print(fin_out.shape)

            all_data = torch.Tensor(np.expand_dims(fin_out,axis = 0)).to(device)
            with torch.no_grad():
                model_net.eval()
                output = model_net(all_data).to(device)
                output = output.permute(2, 3, 1,0).contiguous().view( -1, nChannel)
                ignore, target = torch.max(output, 1)
                im_target = target.data.cpu().numpy()

                im_target = im_target.reshape(fin_out.shape[-2], fin_out.shape[-1])
            save_gz(out_nm, im_target)

            plt.figure(figsize=(10,10))
            plt.imshow(im_target)
            plt.tight_layout()
            #save the map of classification result
            plt.savefig('./result/MA_cluster_figure/MA_cluster_map_'+city+'_'+str(nLabels)+'_'+str(stepsize_con)+'.png', dpi = 400)
            plt.close()
