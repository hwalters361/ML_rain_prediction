# import pickle
import numpy as np
from torch.utils.data import Dataset
import torch
import os
import xarray as xr
import pandas as pd
from random import shuffle
import shutil
from tqdm import tqdm
from time import time
import datetime
import matplotlib.pyplot as plt
import pickle

class ClimateDataset:
    '''Loads data files in a dataframe df'''
    
    def __init__(self, data_dir, sin_embed=True, 
                 time_dim=200, dtype='sst', ssta=False):
        self.data_dir = data_dir
        self.dtype = dtype
        self.sin_embed = sin_embed
        self.time_dim = time_dim
        self.original_data, self.latitude, self.longitude = self.load_data()
        self.months = np.array(self.original_data['M'])
        self.years = np.array(self.original_data['Y'])
        self.T = (self.years - np.min(self.years))*12 + self.months
        self.seasons = ((self.months-1)//4).astype(bool).astype(int)
        self.one_hot_months = self.one_hot(self.months-1)
        self.time_encoding = self.positional_encoding(
            self.months, d_model=self.time_dim, base=10000)
        ##Convert to Numpy Array
        if dtype=='sst':
            col = 'ssta' if ssta else 'sst'
            print(f'self.original_data[col].shape'
                  f': {self.original_data[col].shape}')
            self.original_data=np.stack(self.original_data[col])
            print(f'self.original_data.shape : {self.original_data.shape}')
            self.original_data = self.original_data[:,0,0,:,:]
        elif dtype=='ppt':
            self.original_data=np.stack(self.original_data['ppt'])
        else:
            print(f'Data from column {dtype} is not available!')
            exit()
            
        self.mask = self.get_mask()
        
    def one_hot(self, x):
        x = x.astype(int)
        one_hot = np.zeros((x.size, x.max()+1))
        one_hot[np.arange(x.size),x] = 1
        return one_hot
    
    def positional_encoding(self, position, d_model, base=10000):
        # Initialize a zero vector
        if isinstance(position, (np.ndarray, pd.Series, list)):
            pos_vector = np.zeros((d_model, len(position)))
        else:
            pos_vector = np.zeros((d_model, ))
        # Compute the positional encodings
        for i in range(d_model):
            if i % 2 == 0:
                # Apply the sin transformation for even indices
                pos_vector[i] = np.sin(position / (base ** (2 * i / d_model)))
            else:
                # Apply the cos transformation for odd indices
                pos_vector[i] = np.cos(position / (
                    base ** (2 * (i - 1) / d_model)))
        return pos_vector.T

                  
    def get_filenames(self, directory):
        return sorted(os.listdir(directory))

    def get_path(self, directory, filename):
        return os.path.join(directory, filename)

    def get_basename(self, path):
        return os.path.basename(path)

    def get_month_year(self, file_name):
        '''Get month and year from file name'''
        if self.dtype=='ppt':
            time = pd.Timestamp('-'.join(
                file_name.split('.')[0].split('_')[1:]))
            month, year = time.month, time.year
        elif self.dtype=='sst':
            year, month = (file_name.split('.')[-2][-6:-2], 
                           file_name.split('.')[-2][-2:])
            month, year = int(month), int(year)
        else:
            print(f'dtype {self.dtype} not implemented. Use ppt or sst.')
            month, year = None, None
        return month, year

    def read_ppt(self, file_data, min_lon=0, max_lon=150, 
                 coarsen_lat=4, coarsen_lon=4):
        file_data = file_data.rename_vars({list(file_data.keys())[1] : 'ppt', 
                                        'longitude':'lon', 'latitude':'lat'})
        
        #Coarsen data to reduce resolution
        file_data = file_data.coarsen(latitude=coarsen_lat, 
                                      longitude=coarsen_lon, 
                                      boundary = "trim").mean()
        # file_data = file_data.sel(longitude=slice(-300, -100))
        ##Slice for west coast
        file_data = file_data.sel(longitude=slice(min_lon, max_lon))
        # xr_data.append()
        columns = list('FYMD')
        columns[-1] = 'ppt'   
        return file_data.ppt.values, columns 
    
    def read_sst(self, file_data):
            # Coarsen data to reduce resolution
            file_data = file_data.coarsen(
                lat = 2, lon = 2, boundary = "trim").mean()
            ##Slice for west coast
            # file_data = file_data.sel(lon=slice(100, 400))
            vals = [np.flip(file_data.sst.values, axis=2)[:,:,3:39,:], 
                    np.flip(file_data.ssta.values, axis=2)[:,:,3:39,:]]
            columns = list('FYMDA')
            columns[-2], columns[-1] = 'sst', 'ssta'
            return vals, columns
        
    def read_xr(self, path, min_lon=0, max_lon=150, 
                coarsen_lat=4, coarsen_lon=4):
        '''Read xarray file'''
        file_data = xr.load_dataset(path)
        file_name = self.get_basename(path)
        month, year = self.get_month_year(file_name)
        xr_data = [file_name, year, month]
        
        if self.dtype=='ppt':
            vals, columns = self.read_ppt(
                file_data, min_lon=0, max_lon=150, 
                coarsen_lat=4, coarsen_lon=4)
            xr_data.append(vals)
        elif self.dtype=='sst':
            vals, columns = self.read_sst(file_data)
            xr_data.extend(vals) 
        else:
            print(f'dtype {self.dtype} not implemented. Use ppt or sst.')
        longitude, latitude = file_data.lon.values, file_data.lat.values
        return xr_data, longitude, latitude, columns

    def load_data(self):
        '''Load directory files in a pandas dataframe. 
        Assumes all data has the same longitude and latitude values'''
        filenames = self.get_filenames(self.data_dir)
        data_info_list = []
        for i, file_name in tqdm(enumerate(filenames)):
            path = self.get_path(self.data_dir, file_name)
            xr_data, longitude, latitude, columns = self.read_xr(path)
            data_info_list.append(xr_data)
        df = pd.DataFrame([p for p in data_info_list], columns=columns)
        return df, latitude, longitude

    def normalize(self, avg=None, std=None, per_pixel=False, threshold=1):
        mask = np.broadcast_to(1 - self.mask, 
                               self.original_data.shape).astype(bool)
        masked_array = np.ma.masked_array(self.original_data, mask=mask)
        if avg is None:
            if per_pixel:
                avg = np.mean(masked_array, axis=0, keepdims=True)
            else:
                avg = np.mean(masked_array, keepdims=True)
        x0 = masked_array - avg
        if std is None:
            if per_pixel:
                std = np.std(x0, axis=0, keepdims=True)
            else:
                std = np.std(x0, keepdims=True)
        self.original_data = (x0 / std)      
        return avg, std
    
    
    def min_max_normalize(self, minim=None, maxim=None, 
                          per_pixel=True, minmax_range=2):
        mask = np.broadcast_to(1 - self.mask, 
                               self.original_data.shape).astype(bool)
        masked_array = np.ma.masked_array(self.original_data, mask=mask)
        if minim is None and maxim is None:
            if per_pixel:
                minim = np.min(masked_array, axis=0, keepdims=True)
                maxim = np.max(masked_array, axis=0, keepdims=True)
            else:
                minim = np.min(masked_array, keepdims=True)
                maxim = np.max(masked_array, keepdims=True)
        R = maxim - minim
        if minmax_range == 1:
            self.original_data = (masked_array - minim) / R 
        elif minmax_range == 2:
            self.original_data = 2 * (masked_array - minim) / R - 1
        else:
            print(f'Invalid minmax_range value: {minmax_range}. Pick 1 or 2.')
            exit()
        return minim, maxim
    
    
    def get_mask(self, index_example=0):
        out = self.original_data[index_example]
        mask = ~np.isnan(out)
        mask = torch.from_numpy(mask).float()
        return mask
    
    def get_time_encoding(self, index_example):
        if self.sin_embed:
            time_encoding_arr = torch.from_numpy(
                self.time_encoding[index_example]).float()
        else:
            time_encoding_arr = torch.from_numpy(
                self.one_hot_months[index_example]).float()
        return time_encoding_arr
                
          
    def __getitem__(self, index):
        out = self.original_data[index]
        # Replace NaN values
        out = np.nan_to_num(out, nan=0.0)
        out = torch.from_numpy(out).float()
        time_encoding = self.get_time_encoding(index)
        return (torch.unsqueeze(out, dim=0), 
                torch.unsqueeze(self.mask, dim=0), 
                time_encoding, self.months[index])
                
        # return (torch.unsqueeze(out.flatten(), dim=0), 
        #         torch.unsqueeze(self.mask.flatten(), dim=0), 
        #         time_encoding, self.months[index])
        
    def __len__(self):
        return len(self.original_data)
    
def get_train_test_dirs(directory, shuffle_flag):
    dir_ext = '_shuffle' if shuffle_flag else ''
    train_dir = os.path.join(directory, 'train'+dir_ext)
    test_dir = os.path.join(directory, 'test'+dir_ext)    
    return train_dir, test_dir

def split_data(directory, split=80, shuffle_flag=False, overwrite=True):
    assert split <= 100
    assert split >= 0
    filenames = sorted(os.listdir(directory))
    #Shuffle dataframe rows
    if shuffle_flag:
        shuffle(filenames)
    num_data = len(filenames)
    split_idx = int(num_data * split / 100.0)
    print('split_idx: ', split_idx)
    ## Copy data into train and test directories
    train_dir, test_dir = get_train_test_dirs(directory, shuffle_flag)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    # # if overwrite:
    # #     for f in os.listdir(train_dir):
    # #         os.remove(f)
    # #     for f in os.listdir(test_dir):
    # #         os.remove(f)
    for base_f in filenames[:split_idx]:
        if os.path.isfile(os.path.join(directory, base_f)):
            shutil.copy(os.path.join(directory, base_f), 
                        os.path.join(train_dir, base_f))
    for base_f in filenames[split_idx:]:
        if os.path.isfile(os.path.join(directory, base_f)):
            shutil.copy(os.path.join(directory, base_f), 
                        os.path.join(test_dir, base_f))
    return train_dir, test_dir
    

def build_dataloader(cfg, save_dir=None):
    # Load dataset params
    data_dir = cfg.get('User', 'data_dir')
    batch_size = cfg.getint('DataFrame', 'batch_size')
    num_workers = cfg.getint('DataFrame', 'num_workers')
    split = cfg.getboolean('DataFrame', 'split')
    split_percent = cfg.getint('DataFrame', 'split_percent') 
    shuffle_flag = cfg.getboolean('DataFrame', 'shuffle')
    overwrite = cfg.getboolean('DataFrame', 'overwrite')
    saved_root = cfg.get('User', 'saved_root')
    normalize_per_pixel = cfg.getboolean('DataFrame', 'normalize_per_pixel')
    time_dim = cfg.getint('DataFrame', 'time_dim')
    sin_embed = cfg.getboolean('DataFrame', 'sin_embed')
    dataset_name = cfg.get('DataFrame', 'dataset_name')
    normalization_type = cfg.get('DataFrame', 'normalization_type')
    minmax_range = cfg.getint('DataFrame', 'minmax_range')
    
    #split data
    if split:
        train_dir, test_dir = split_data(
            data_dir, split=split_percent, 
            shuffle_flag=shuffle_flag, overwrite=overwrite)
    else:
        train_dir, test_dir = get_train_test_dirs(data_dir, shuffle_flag)
        
    train_dataset = ClimateDataset(
        train_dir, sin_embed=sin_embed, time_dim=time_dim, dtype='sst')
    test_dataset = ClimateDataset(
        test_dir, sin_embed=sin_embed, time_dim=time_dim, dtype='sst')
    
    if normalization_type == 'minmax':
        minim, maxim = train_dataset.min_max_normalize(
            minim=None, maxim=None, per_pixel=normalize_per_pixel, 
            minmax_range=minmax_range)
        
        test_dataset.min_max_normalize(
            minim, maxim, per_pixel=normalize_per_pixel, 
            minmax_range=minmax_range)
        
    elif normalization_type == 'standard':
        avg, std = train_dataset.normalize(per_pixel=normalize_per_pixel)
        avg, std = test_dataset.normalize(
            avg, std, per_pixel=normalize_per_pixel)
    else:
        print(f'Normalization type {normalization_type} not implemented.')
        exit()

    if normalize_per_pixel:
        if save_dir is None:
            save_dir = saved_root
        if normalization_type == 'minmax':
            minim.dump(os.path.join(save_dir, 'SST_min_'+dataset_name+'.pkl'))
            maxim.dump(os.path.join(save_dir, 'SST_max_'+dataset_name+'.pkl'))
        elif normalization_type == 'standard':
            avg.dump(os.path.join(save_dir, 'SST_avg_'+dataset_name+'.pkl'))
            std.dump(os.path.join(save_dir, 'SST_std_'+dataset_name+'.pkl'))
        else:
            print(f'Normalization type {normalization_type} not implemented.')
            exit()
    
    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=shuffle, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=shuffle, num_workers=num_workers)
        
    train_num, test_num = train_dataset.__len__(), test_dataset.__len__()

    return train_dataloader, test_dataloader, train_num, test_num
    