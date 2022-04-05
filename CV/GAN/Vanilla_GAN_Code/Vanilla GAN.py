#!/usr/bin/env python
# coding: utf-8

# ## 1) Libraries & Hyperparameters

# In[4]:


import os
import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


# Hyperparameter

epoch = 100
batch_size = 256
learning_rate = 0.0002
z_size = 100


# ## 2) Generator

# In[6]:


# Generator receives random noise z and create 1*28*28 image

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        
        def block(input_dim, output_dim, normalize=True):
            layers = [nn.Linear(input_dim, output_dim)]
            
            if normalize:
                layers.append(nn.BatchNorm1d(output_dim, momentum=0.8))
            
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            return layers
    
        self.model = nn.Sequential(
            *block(z_size, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 1 * 28 * 28),
            nn.Tanh()
        )
        
    def forward(self,z):
        out = self.model(z)
        out = out.view(batch_size,1,28,28)
        return out


# ## 3) Discriminator

# In[7]:


# Discriminator receives 1*28*28 image and returns a float number 0~1

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(1 * 28 * 28, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        
    def forward(self,x):
        out = x.view(batch_size, -1)
        out = self.model(out)
        return out


# ## 4) Data Load

# In[10]:


transforms_train = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

mnist_train = dset.MNIST("./",
                         train=True,
                        transform = transforms_train,
                        download=False)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=8)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
#print(torch.cuda.get_device_name(0))

generator = Generator().to(device)
discriminator = Discriminator().to(device)


# ## 5) Loss Function & Optimizer

# In[11]:


loss_func = nn.BCELoss()
loss_func.to(device)

gen_optim = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
dis_optim = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5,0.999))


ones_label = torch.ones(batch_size,1).to(device)
zeros_label = torch.zeros(batch_size,1).to(device)


# ## 6) Train Model

# In[13]:


get_ipython().run_cell_magic('time', '', '\ntry:\n    os.mkdir("./result")\nexcept:\n    pass\n\nfor i in range(epoch):\n    for j,(image, _) in enumerate(train_loader):\n        image = image.to(device)\n        \n         # Generator Train\n        gen_optim.zero_grad()\n        \n        # Fake Data\n        z = init.kaiming_normal_(torch.Tensor(batch_size, z_size), a=0.2, mode=\'fan_in\', nonlinearity=\'leaky_relu\').to(device)\n        #z = torch.normal(mean=0, std=1, size=(batch_size,z_size)).to(device)\n        gen_fake = generator(z)\n        dis_fake = discriminator(gen_fake)\n        \n        gen_loss = loss_func(dis_fake, ones_label)\n        gen_loss.backward()\n        gen_optim.step()\n        \n        # discriminator train\n        dis_optim.zero_grad()\n        \n        # Real Data\n        dis_real = discriminator(image)\n        \n        # calculate gradient after sum two losses\n        real_loss = loss_func(dis_real,ones_label)\n        fake_loss = loss_func(discriminator(gen_fake.detach()), zeros_label)\n\n        dis_loss = (real_loss + fake_loss)/2\n        dis_loss.backward()\n        dis_optim.step()\n    \n    #torch.save([generator,discriminator], \'./model/vanilla_gan.pkl\')\n    v_utils.save_image(gen_fake.cpu().data[0:100],\'./result/gen_{}_{}.png\'.format(i,j), nrow=10)\n    print("{}th epoch gen_loss: {} dis_loss:{}".format(i,gen_loss.data,dis_loss.data))')


# In[ ]:


from glob import glob 

for i in range(epoch):
    print(i)
    file_list = glob("./result/gen_{}_*.png".format(i))
    img_per_epoch = len(file_list)
    for idx,j in enumerate(file_list):
        img = plt.imread(j)
        plt.subplot(1,img_per_epoch,idx+1)
        plt.imshow(img)
    plt.show()

