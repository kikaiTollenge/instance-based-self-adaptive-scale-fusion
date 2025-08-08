import torch.nn as nn
import torch
import torch.nn.functional as F

def decouple_feature(x):
    feature_length = int(x.shape[1] / 2)
    high_feature = x[:,:feature_length]
    low_feature = x[:,feature_length:]
    return high_feature,low_feature

class AdapetiveScaleFusion(nn.Module):
    def __init__(self, input_size=2048,fusion_type = 'weighted',alpha = 0.0,concat_type = 'concat'):
        super().__init__()
        self.input_size = input_size
        self.fusion_type = fusion_type
        self.alpha = alpha
        self.concat_type = concat_type
        self.high_Q = nn.Sequential(nn.Linear(self.input_size, self.input_size),nn.Dropout(0.5)) 
        self.low_Q = nn.Sequential(nn.Linear(self.input_size, self.input_size),nn.Dropout(0.5))

    
    def forward(self, high_feature, low_feature):
        if self.fusion_type == 'weighted':
            high_K = self.high_Q(high_feature)
            low_K = self.low_Q(low_feature)


            high_sim = F.cosine_similarity(high_feature, high_K, dim=1).unsqueeze(1)
            low_sim = F.cosine_similarity(low_feature, low_K, dim=1).unsqueeze(1)


            weights = torch.cat([high_sim, low_sim], dim=1)
            weights = F.softmax(weights, dim=1)

            weights_high = weights[:, 0].unsqueeze(1)
            weights_low = weights[:, 1].unsqueeze(1)


            feature = weights_high * high_feature + weights_low * low_feature

            if self.concat_type == 'concat':
                feature_add_index = torch.argmax(weights, dim=1)
                feature_add = torch.stack([
                    high_feature[i] if idx == 0 else low_feature[i]
                    for i, idx in enumerate(feature_add_index)
                ])
                feature = torch.cat([feature, feature_add], dim=1)

        elif self.fusion_type == 'custom':
            feature = self.alpha * high_feature + (1.0 - self.alpha) * low_feature
            if self.concat_type == 'concat':
                if self.alpha >= (1.0 - self.alpha):
                    feature = torch.cat([feature, high_feature], dim=1)
                else:
                    feature = torch.cat([feature, low_feature], dim=1)

        return feature

if __name__ == '__main__':
    model = AdapetiveScaleFusion()
    high_feature = torch.randn(10,2048)
    low_feature = torch.randn(10,2048)
    feature = model(high_feature,low_feature)
    print(feature.shape)