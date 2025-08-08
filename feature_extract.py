import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader,Dataset
import os
from tqdm import tqdm
import pickle
from ctranspath.ctrans import ctranspath
import copy

import timm
import torch


def process_batch_data(feature,current_coord,low_coord):
    process_list = []
    for i in range(feature.shape[0]):
        process_dict = {}
        process_dict['coord'] = current_coord[i]
        process_dict['low_coord'] = low_coord[i]
        process_dict['feature'] = feature[i].detach().cpu().numpy()
        process_list.append(process_dict)    
    return process_list

class double_magnification(Dataset):
    def __init__(self,wsi_path,resolution):
        super(double_magnification, self).__init__()
        self.wsi_path  = wsi_path
        self.patch_list = []
        self.resolution = resolution
        if self.resolution == 'low':
            for item in os.listdir(self.wsi_path):
                item_path = os.path.join(self.wsi_path,item)
                if item_path.endswith('.jpeg'):
                    self.patch_list.append(item_path)
        elif self.resolution == 'high':
            for item in os.listdir(self.wsi_path):
                item_path = os.path.join(self.wsi_path,item)
                if os.path.isdir(item_path):
                    for img in os.listdir(item_path):
                        if img.endswith('.jpeg'):
                            self.patch_list.append(os.path.join(item_path,img))
        self.patch_list = sorted(self.patch_list)
        self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
    def __len__(self):
        return len(self.patch_list)
    
    def __getitem__(self, idx):
        img_path = self.patch_list[idx]
        current_coord = os.path.splitext(os.path.basename(img_path))[0]
        if self.resolution == 'low':
            low_coord = current_coord
        elif self.resolution == 'high':
            low_coord = os.path.basename(os.path.dirname(img_path))
        img = self.transfer_img2tensor(img_path)
        return img,current_coord,low_coord

    def transfer_img2tensor(self, img_path):
        img = Image.open(img_path).convert('RGB')
        # print(type(img))
        img = self.transform(img)
        img = img.to(dtype=torch.float32)
        return img
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load(r'ctranspath/ctranspath.pth')
    model.load_state_dict(td['model'], strict=True)

    high_device = "cuda:0"
    low_device = "cuda:0"

    high_reso_model = copy.deepcopy(model).to(high_device).eval()
    low_reso_model = copy.deepcopy(model).to(low_device).eval()
    roots = ['/home/Public/yjk/WSI/C16/train/normal','/home/Public/yjk/WSI/C16/train/tumor','/home/Public/yjk/WSI/C16/test','/home/Public/yjk/WSI/TCGA-LUAD','/home/Public/yjk/WSI/TCGA-LUSC']
    for root in tqdm(roots):
        print(root)
        for patient_folder in tqdm(os.listdir(root)):
            high_save_path = os.path.join(root,f'{patient_folder}_high_feature_ctrans.pkl')
            low_save_path = os.path.join(root,f'{patient_folder}_low_feature_ctrans.pkl')
            high_list = []
            low_list = []
            patient_folder_path = os.path.join(root, patient_folder)
            if patient_folder.endswith('.txt') or patient_folder.endswith('.pkl'):
                continue
            high_reso_dataset = double_magnification(patient_folder_path, 'high')
            low_reso_dataset = double_magnification(patient_folder_path, 'low')
            high_reso_dataloader = DataLoader(high_reso_dataset, batch_size=256, shuffle=False, num_workers=4)
            low_reso_dataloader = DataLoader(low_reso_dataset, batch_size=256, shuffle=False, num_workers=4)
            
            for high_idx,(img,current_coord,low_coord) in enumerate(high_reso_dataloader):
                with torch.no_grad():
                    img = img.to(high_device)
                    high_reso_model.to(device=high_device)
                    output = high_reso_model(img)
                result = process_batch_data(output,current_coord,low_coord)
                high_list.extend(result)

            with open(high_save_path,'wb') as f:
                pickle.dump(high_list,f)

            for low_idx,(img,current_coord,low_coord) in enumerate(low_reso_dataloader):
                with torch.no_grad():
                    img = img.to(low_device)
                    low_reso_model.to(device=low_device)
                    output = low_reso_model(img)
                result = process_batch_data(output,current_coord,low_coord)
                low_list.extend(result)
                
            with open(low_save_path,'wb') as f:
                pickle.dump(low_list,f)
    
    print("Finished")