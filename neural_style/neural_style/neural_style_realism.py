from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image


import torchvision.transforms as transforms
import torchvision.models as models

import copy
import os
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispostivo de renderización utilizado: ",device)

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize((imsize,imsize)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

unloader = transforms.ToPILImage()  # reconvert into PIL image

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def save_image(tensor,path_to_save):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    image.save(path_to_save)

content_img = image_loader("images/turtle.jpg")
style_img = image_loader("images/wave.jpg")
#original_img = torch.randn(content_img.data.size(), device=device, requires_grad=True)
original_img = content_img.clone().requires_grad_(True)

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.chosen_features = ['0','5','10','19','28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self,x):
        features = []
        for layer_num,layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

model = VGG().to(device).eval()

## Hiperparametros
total_steps = 6000
alpha = 1
beta = 1000000 
show_every = 300
results_dir = 'results/photo_realistic/'

optimiced = optim.Adam([original_img])

for step in range(total_steps):
    def closure():   
        # correct the values of updated input image
        original_img.data.clamp_(0, 1) 
        
        optimiced.zero_grad()
        generated_features = model(original_img)
        content_features = model(content_img)
        style_features = model(style_img)
        
        style_loss = content_loss = 0
        for gen_feature,cont_feature,style_feature in zip(
            generated_features,content_features,style_features
        ):
            batch_size,channel,height,width = gen_feature.shape
            content_loss += torch.mean((gen_feature-cont_feature)**2)

            G = gen_feature.view(channel,height*width).mm(
                gen_feature.view(channel,height*width).t()
            )
            A = style_feature.view(channel,height*width).mm(
                style_feature.view(channel,height*width).t()
            )
            style_loss += torch.mean((G-A)**2)

        
        # Añadir el término de regularización a la pérdida total
        total_loss = alpha * content_loss + beta * style_loss
        optimiced.zero_grad()
        total_loss.backward()
        
        if step % show_every == 0:
            save_image(original_img,os.path.join(results_dir,'generated-{}.png'.format(step)))
        
        return alpha*content_loss + beta*style_loss
    print("Step: ",step) 
    optimiced.step(closure)
    
    # a last correction...
    original_img.data.clamp_(0, 1)
