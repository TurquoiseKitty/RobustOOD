
from torch.utils.data.dataset import Dataset
from base.torchvision_dataset import TorchvisionDataset
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import ipdb


class PTC_Dataset(TorchvisionDataset):

    def __init__(self, root,test,plot):
        self.scaler = StandardScaler()
        self.train_set=PTC(root,test,plot,train=True,scaler=self.scaler)
        self.test_set =PTC(root,test,plot,train=False,scaler=self.scaler)


    
class PTC:
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, root,test,plot,train,scaler):
        super(PTC, self).__init__()
        self.train=train
        if train:
            # read root+covs.csv as self.side_train_set
            # read root+c.csv as self.data_train_set
            side_train_set = pd.read_csv(root+"covs.csv",header=None).to_numpy()
            data_train_set = pd.read_csv(root+"c.csv",header=None).to_numpy()
            # if the dim of side_train_set is 1, reshape it to (n,1)
            if len(side_train_set.shape)==1:
                side_train_set=side_train_set.reshape(-1,1)
            # if the dim of data_train_set is 1, reshape it to (n,1)
            if len(data_train_set.shape)==1:
                data_train_set=data_train_set.reshape(-1,1)

            scaler.fit_transform(side_train_set)
            scaled_side_train_set=scaler.transform(side_train_set)
            scaled_side_train_set=100*scaled_side_train_set

            self.data_train_set=torch.tensor(data_train_set.astype(np.float32))
            self.side_train_set=torch.tensor(scaled_side_train_set.astype(np.float32))

        else:
            side_test_set = pd.read_csv(test+"covs.csv",header=None).to_numpy()
            #data_test_set = pd.read_csv(test+"c.csv",header=None)
            # if the dim of side_test_set is 1, reshape it to (n,1)
            if len(side_test_set.shape)==1:
                side_test_set=side_test_set.reshape(-1,1)
            
            scaled_side_test_set=scaler.transform(side_test_set)
            scaled_side_test_set=100*scaled_side_test_set

            if plot is not None:
                # read plot data
                side_plot_set = pd.read_csv(plot+"covs.csv",header=None)
                #data_plot_set = pd.read_csv(plot+"c.csv",header=None)

                # scale plot data
                scaled_side_plot_set=scaler.transform(side_plot_set)
                scaled_side_plot_set=100*scaled_side_plot_set

                # concatenate test data and plot data
                #data_test_set = np.concatenate([data_test_set,data_plot_set],axis=0)
                scaled_side_test_set = np.concatenate([scaled_side_test_set,scaled_side_plot_set],axis=0)

            
            #self.data_test_set=torch.tensor(data_test_set.astype(np.float32))
            self.side_test_set=torch.tensor(scaled_side_test_set.astype(np.float32))
        

    def __getitem__(self, index):
        if self.train:
            side, data = self.side_train_set[index], self.data_train_set[index]
        else: 
            side, data = self.side_test_set[index],0
        return (side, data, index)
    
    
    def __len__(self):
        if self.train:
            return len(self.side_train_set)
        else:
            return len(self.side_test_set)
        
        
        