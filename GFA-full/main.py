import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from cam import GradCAM, show_cam_on_image, center_crop_img
from model.make_model import build_vision_transformer
from config.config import cfg

class ReshapeTransform:
    def __init__(self, model):
        self.h = 21
        self.w = 10

    def __call__(self, x):  #[210,768]
        # [batch_size, num_tokens, token_dim]
   #     print(x.shape)
        result = x[:,:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     768)
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result



class ReshapeTransform1:
    def __init__(self, model):
        self.h = 21
        self.w = 10

    def __call__(self, x):  #[210,768]
        # [batch_size, num_tokens, token_dim]
   #     print(x.shape)
        result = x[:,1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     768)
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result

model = build_vision_transformer(num_classes=395,cfg = cfg)
#print(model)
weights_path = "/home/jiaqi/Baseline9/save_model/sysulr_0.00035_time_20250221_204053_adamw_best.t"
checkpoint = torch.load(weights_path, map_location="cuda" if torch.cuda.is_available() else "cpu")


if 'net' in checkpoint:
    model.load_state_dict(checkpoint['net'], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)


#target_layers1 = [model.base.patch_embed.proj]
#target_layers2 = [model.base.pos_drop]
target_layers3 = [model.base.blocks[-1].norm1]




#target_layers = [model.base.patch_embed]


data_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
                                     
# load image
img_path = "0069_c06_s202528_f4280_nir.jpg"
img = Image.open(img_path)
img = img.convert('RGB')
img = img.resize((128,256), Image.BICUBIC)
img = np.array(img, dtype=np.uint8)
#img = center_crop_img(img, 128,256)


plt.imshow(img)
plt.axis('off')
plt.show()


#print(img.shape)   #[1,256,128]


# [C, H, W]
img_tensor = data_transform(img)


input_tensor = torch.unsqueeze(img_tensor, dim=0)
#print(input_tensor.shape)    #[1,3,256,128]


#cam1 = GradCAM(model=model,
#            target_layers=target_layers1,
#            use_cuda=False,
#            reshape_transform=ReshapeTransform(model))
#cam2 = GradCAM(model=model,
#            target_layers=target_layers2,
#            use_cuda=False,
#            reshape_transform=ReshapeTransform1(model))
cam3 = GradCAM(model=model,
            target_layers=target_layers3,
            use_cuda=False,
            reshape_transform=ReshapeTransform1(model))            



#grayscale_cam1 = cam1(input_tensor=input_tensor, target_category=None)
#grayscale_cam2 = cam2(input_tensor=input_tensor, target_category=None)
grayscale_cam3 = cam3(input_tensor=input_tensor, target_category=None)
#print(grayscale_cam.shape)  #[1,256,128]
#print(type(grayscale_cam))  #numpy



#visualization1 = show_cam_on_image(img/255., grayscale_cam1, use_rgb=True)
#visualization2 = show_cam_on_image(img/255., grayscale_cam2, use_rgb=True)
visualization3 = show_cam_on_image(img/255., grayscale_cam3, use_rgb=True)




#plt.imshow(visualization1)
#plt.axis('off')
#plt.show()



#plt.imshow(visualization2)
#plt.axis('off')
#plt.show()



plt.imshow(visualization3)
plt.axis('off')
plt.show()