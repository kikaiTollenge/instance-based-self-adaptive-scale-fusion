import torch.nn as nn
import torch
import torch.nn.functional as F
from model.adaptiveScaleFusion import *

    
class GatedAttention(nn.Module):
    def __init__(self, config):
        super(GatedAttention, self).__init__()
        self.M = config['input_size']
        self.num_class = 2
        self.L = int(self.M / 2)
        self.fusion_type = config['fusion_type']
        self.alpha = config['alpha']
        self.concat_type = config['concat_type']

        self.factor = config['factor']
        assert self.concat_type in ['concat','none'],'concat_type must be concat or none'
        assert self.fusion_type in ['weighted','custom'],'type must be weighted or custom'
        self.ATTENTION_BRANCHES = 1

        # self.ReduceDim = nn.Linear(self.M, self.L)
        self.multi_weights = AdapetiveScaleFusion(self.M, self.fusion_type, self.alpha, self.concat_type)

        self.attention_V = nn.Sequential(
            nn.Linear(self.M * self.factor, self.L), # matrix V
            nn.Tanh(),
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M * self.factor, self.L), # matrix U
            nn.Sigmoid(),
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES*self.factor, self.num_class)
        )
    

    def forward(self, x, multi_scale = True):
        x = x.squeeze(0)
        if multi_scale  == True:
            high_feature, low_feature = decouple_feature(x)
            # high_feature = self.ReduceDim(high_feature)
            # low_feature = self.ReduceDim(low_feature)
            x = self.multi_weights(high_feature,low_feature)
        # else:
        #     x = self.ReduceDim(x)
        H = x

        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_logistic = self.classifier(Z)

        return Y_logistic
    
if __name__ == '__main__':
    model = GatedAttention(input_size=768,fusion_type='weighted',concat_type='none')
    x = torch.randn(392, 1536)
    output = model(x,True)
    print(output.shape)