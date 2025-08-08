from torch.utils.data import DataLoader,Dataset,Subset
import pandas as pd
import os
import pickle
import numpy as np
import torch

class pkl_Dataset(Dataset):
    def __init__(self,dataset='c16',train='train',multi_scale= 'both',source='ctrans'):
        super().__init__()
        self.dataset = dataset
        self.train = train
        self.multi_scale = multi_scale
        self.source = source
        assert self.multi_scale in ['both','high','low'],"multi_scale must be both,high or low in string type"
        assert self.train in ['train','test'],"train must be train or test in string type"
        config ={
            'c16':{
                'train': '/home/yjk/graduate_work/indices/c16_train.csv',
                'test': '/home/yjk/graduate_work/indices/c16_test.csv'
            },
            'tcga':{ #8:2
                'train': '/home/yjk/graduate_work/indices/tcga_train.csv',
                'test': '/home/yjk/graduate_work/indices/tcga_test.csv'
            }
        }
        if self.train == 'train':
            self.data_path = config[self.dataset][self.train]
            self.data = pd.read_csv(self.data_path)
        else:
            self.data_path = config[self.dataset][self.train]
            self.data = pd.read_csv(self.data_path)
        
        self.label_config ={
            'normal':0,
            'tumor':1,
            'TCGA-LUSC':0,
            'TCGA-LUAD':1,
        }


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image_path = self.data.iloc[index, 0]
        # print(image_path)
        label = self.label_config[self.data.iloc[index, 1]]
            
        if self.source == 'ctrans':
            high_pkl_path = image_path + '_high_feature_ctrans.pkl'
            low_pkl_path = image_path + '_low_feature_ctrans.pkl'

        concat_feature = []
        with open(high_pkl_path, 'rb') as f:
            high_feature = pickle.load(f)
        with open(low_pkl_path, 'rb') as f:
            low_feature = pickle.load(f)
        low_coord_to_feature = {item['coord']:item['feature'] for item in low_feature}
        if self.multi_scale == 'both':
            for  i in range(len(high_feature)):
                if high_feature[i]['low_coord'] in low_coord_to_feature.keys(): 
                    high_array = np.array(high_feature[i]['feature'])
                    low_array = np.array(low_coord_to_feature[high_feature[i]['low_coord']])
                    feature = np.concatenate([high_array,low_array]) 
                    concat_feature.append(feature)
            final_feature = np.stack(concat_feature)
        elif self.multi_scale == 'high':
            for i in range(len(high_feature)):
                concat_feature.append(high_feature[i]['feature'])
            final_feature = np.stack(concat_feature)
        elif self.multi_scale == 'low':
            for i in range(len(low_feature)):
                concat_feature.append(low_feature[i]['feature'])
            final_feature = np.stack(concat_feature)
        final_feature = torch.from_numpy(final_feature).to(torch.float32)
        label = torch.tensor(label,dtype=torch.long)

        return final_feature,label

if __name__ == '__main__':
    train_dataset = pkl_Dataset('tcga','train','high','moco')
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False)
    # print(len(train_loader))
    for feature, label in train_loader:
        print(feature.shape, label)
