#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO

from medmnist.utils import montage2d
# In[2]:
# # We first work on a 2D dataset

# In[9]:


data_flag = 'pneumoniamnist'
# data_flag = 'breastmnist'
download = True

NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])


# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

pil_dataset = DataClass(split='train', download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

sel = np.random.choice(100, size=4, replace=False)

montage_img = montage2d(imgs=train_dataset.imgs,
                                n_channels=n_channels,
                                sel=sel)
#montage_img.save('tmp.jpg')
#print(type(montage_img))
Inputs,Targets=None,None
for inputs, targets in tqdm(train_loader):
    if Inputs is None:
        Inputs=inputs
    else:
        Inputs=torch.cat((Inputs,inputs))

    if Targets is None:
        Targets=targets
    else:
        Targets=torch.cat((Targets,targets))
torch.save(Inputs,'pneumonia_data.pth')
torch.save(Targets,'pneumonia_label.pth')