import pandas as pd
import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from ctrans import ctranspath


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]
)

img_csv=pd.read_csv(r'./test_list.csv')

model = ctranspath()
model.head = nn.Identity()
td = torch.load(r'./ctranspath.pth')
model.load_state_dict(td['model'], strict=True)


model.eval()
with torch.no_grad():
    for batch in database_loader:
        features = model(batch)
        features = features.cpu().numpy()


#