from model.vision_transformer import ViT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import copy
# L2 norm
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class build_vision_transformer(nn.Module):
    def __init__(self, num_classes, cfg):
        super(build_vision_transformer, self).__init__()
        self.in_planes = 768

        self.base = ViT(img_size=[cfg.H,cfg.W],
            stride_size=cfg.STRIDE_SIZE,
            drop_path_rate=cfg.DROP_PATH,
            drop_rate=cfg.DROP_OUT,
            attn_drop_rate=cfg.ATT_DROP_RATE)        

        self.base.load_param(cfg.PRETRAIN_PATH)

        print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.l2norm = Normalize(2)
        self.collab_module = CollaborativeMatchingModule()

        self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_1.apply(weights_init_classifier)
        self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_2.apply(weights_init_classifier)
        self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_3.apply(weights_init_classifier)
        self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_4.apply(weights_init_classifier)
        self.classifier_5 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_5.apply(weights_init_classifier)

        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)
        self.bottleneck_5 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_5.bias.requires_grad_(False)
        self.bottleneck_5.apply(weights_init_kaiming)

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )


       



    def forward(self, x):
         
        features = self.base(x)  #[96,211,768]
        
        b1_feat = self.b1(features) 
        global_feat = b1_feat[:, 0]  #[96,768]
        
        feature_length = features.size(1) - 1  #210
        patch_length = feature_length // 5   #42
        token = features[:, 0:1]
        
        x = features[:, 1:]   #[96,210,768]
        
        # lf_1
        b1_local_feat = x[:, :patch_length]
      #  print(b1_local_feat.shape)
      #  print(token.shape)
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        # lf_5
        b5_local_feat = x[:, patch_length*4:patch_length*5]
        b5_local_feat = self.b2(torch.cat((token, b5_local_feat), dim=1))
        local_feat_5 = b5_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)
        local_feat_5_bn = self.bottleneck_5(local_feat_5)
        
        if self.training:
            cls_score = self.classifier(feat)
            cls_score_1 = self.classifier_1(local_feat_1_bn)
            cls_score_2 = self.classifier_2(local_feat_2_bn)
            cls_score_3 = self.classifier_3(local_feat_3_bn)
            cls_score_4 = self.classifier_4(local_feat_4_bn)
            cls_score_5 = self.classifier_5(local_feat_5_bn)

            return cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4, cls_score_5, global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_4, local_feat_5

        else:
         #   feat = torch.cat([global_feat, local_feat_1 / 5, local_feat_2 / 5, local_feat_3 / 5, local_feat_4 / 5, local_feat_5 / 5], dim=1)
        
            feat = torch.cat([feat, local_feat_1_bn / 10, local_feat_2_bn / 10, local_feat_3_bn / 10, local_feat_4_bn / 10, local_feat_5_bn / 10], dim=1)
            return self.l2norm(feat)
          
    
    
     


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
        
        
        
class CollaborativeMatchingModule(nn.Module):
    def __init__(self):
        super(CollaborativeMatchingModule, self).__init__()
      #  self.conv1 = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, x):
        # Initial feature fusion
        visible_features, infrared_features = x.chunk(2,0)
        synergetic_feature = visible_features + infrared_features

        # Apply 1x1 convolutions
    #    visible_conv = self.conv1(visible_features)
    #    infrared_conv = self.conv1(infrared_features)
    #    synergetic_conv = self.conv1(synergetic_feature)

        # Element-wise multiplication and softmax normalization
        A_vi = F.softmax(visible_features * synergetic_feature, dim=1)
        A_ir = F.softmax(infrared_features * synergetic_feature, dim=1)

        # Enhance initial features by embedding the matched information
        enhanced_visible = visible_features + visible_features * A_vi
        enhanced_infrared = infrared_features + infrared_features * A_ir

        return torch.cat([enhanced_visible, enhanced_infrared])       
  