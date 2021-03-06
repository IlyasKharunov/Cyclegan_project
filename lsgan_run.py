#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter 
import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from collections import namedtuple

from code.utils import ReplayBuffer
from code.utils import LambdaLR
from code.utils import weights_init_normal
from code.datasetsdrive import ImageDataset
from code.cyclegan_model import *


# In[ ]:


Params = namedtuple('Params', ['startepoch', 'n_epochs', 
                               'batchSize', 'dataroot', 
                               'lr', 'decay_epoch', 
                               'size', 'input_nc', 
                               'output_nc', 'cuda', 
                               'n_cpu', 'resume', 
                               'gpu_ids'])


# In[ ]:


opt = Params(startepoch = 0, n_epochs = 200, 
             batchSize = 1, dataroot = 'database', 
             lr = 0.0002, decay_epoch = 100, 
             size = (128,128), input_nc = 3, 
             output_nc = 3, cuda = True, 
             n_cpu = 0, resume = False, 
             gpu_ids = [0,1])


# In[ ]:


netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)


# In[ ]:


netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)


# In[ ]:


base = 'models/'

#load model to finetune
if opt.resume == False:
  h2z = torch.load(f'{base}pretrained/horse2zebra.pth')
  z2h = torch.load(f'{base}pretrained/zebra2horse.pth')

  stat1 = netG_A2B.state_dict()
  stat2 = netG_B2A.state_dict()

  for i in stat1:
      stat1[i] = h2z[i]

  for i in stat1:
      stat2[i] = z2h[i]

  netG_A2B.load_state_dict(stat1)
  netG_B2A.load_state_dict(stat2)

#load model to resume training
if opt.resume == True:
    state_dicts = []
    # original saved file with DataParallel
    state_dicts.append(torch.load(f'{base}outputls/netD_A.pth'))
    state_dicts.append(torch.load(f'{base}outputls/netD_B.pth'))
    state_dicts.append(torch.load(f'{base}outputls/netG_A2B.pth'))
    state_dicts.append(torch.load(f'{base}outputls/netG_B2A.pth'))
    # create new OrderedDict that does not contain `module.`
    new_state_dicts = [OrderedDict() for i in range(4)]
    for i in range(len(state_dicts)):
        for k, v in state_dicts[i].items():
            name = k[7:] # remove `module.`
            new_state_dicts[i][name] = v
    # load params
    netD_A.load_state_dict(torch.load(new_state_dict[0]))
    netD_B.load_state_dict(torch.load(new_state_dict[1]))
    netG_A2B.load_state_dict(torch.load(new_state_dict[2]))
    netG_B2A.load_state_dict(torch.load(new_state_dict[3]))
    del state_dicts
    del new_state_dicts


# In[ ]:


#transfer to cuda device
netG_A2B.to(opt.gpu_ids[0])
netG_B2A.to(opt.gpu_ids[0])
netD_A.to(opt.gpu_ids[0])
netD_B.to(opt.gpu_ids[0])
#netG_A2B = torch.nn.DataParallel(netG_A2B, opt.gpu_ids)
#netG_B2A = torch.nn.DataParallel(netG_B2A, opt.gpu_ids)
#netD_A = torch.nn.DataParallel(netD_A, opt.gpu_ids)
#netD_B = torch.nn.DataParallel(netD_B, opt.gpu_ids)


# In[ ]:


# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()


# In[ ]:


# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.startepoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.startepoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.startepoch, opt.decay_epoch).step)


# In[ ]:


# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size[0], opt.size[1])
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size[0], opt.size[1])
output_A = Tensor(opt.batchSize, opt.input_nc, opt.size[0], opt.size[1])
output_B = Tensor(opt.batchSize, opt.input_nc, opt.size[0], opt.size[1])
target_real = Tensor(opt.batchSize).fill_(1.0)
target_fake = Tensor(opt.batchSize).fill_(0.0)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()


# In[ ]:


# Dataset loader
transforms_ = [ transforms.Resize(opt.size, Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

dataset = ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True)

dataloader = DataLoader(dataset, batch_size=opt.batchSize, 
                        shuffle=True, num_workers=opt.n_cpu)


# In[ ]:


# Loss plot
logger = SummaryWriter(filename_suffix='first', log_dir='/content/logs')
###################################


# In[ ]:


###### Training ######
log_loss_G_Summarized = 0
log_loss_G_Identity = 0
log_loss_G_GAN = 0
log_loss_G_Cycle = 0
log_loss_D_Summarized = 0

shift = len(dataloader)
for epoch in range(opt.startepoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = input_A.copy_(batch['A'])
        real_B = input_B.copy_(batch['B'])

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        out_A = output_A.copy_(fake_A)
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        out_B = output_B.copy_(fake_B)
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################
        # Progress report (http://localhost:8097)
        if i % 10 == 0:
          logger.add_scalar("loss_G/Summarized", log_loss_G_Summarized/10, i + shift*epoch)
          logger.add_scalar("loss_G/Identity", log_loss_G_Identity/10, i + shift*epoch)
          logger.add_scalar("loss_G/GAN", log_loss_G_GAN/10, i + shift*epoch)
          logger.add_scalar("loss_G/Cycle", log_loss_G_Cycle/10, i + shift*epoch)
          logger.add_scalar("loss_D/Summarized", log_loss_D_Summarized/10, i + shift*epoch)
          logger.add_image('Real/A', (real_A[0] + 1)/2)
          logger.add_image('Real/B', (real_B[0] + 1)/2)
          logger.add_image('Fake/B', (out_B[0] + 1)/2)
          logger.add_image('Fake/A', (out_A[0] + 1)/2)
          logger.flush()
          if i % 100 == 0:
            print(i)
          log_loss_G_Summarized = 0
          log_loss_G_Identity = 0
          log_loss_G_GAN = 0
          log_loss_G_Cycle = 0
          log_loss_D_Summarized = 0

        log_loss_G_Summarized += loss_G
        log_loss_G_Identity += (loss_identity_A + loss_identity_B)
        log_loss_G_GAN += (loss_GAN_A2B + loss_GAN_B2A)
        log_loss_G_Cycle += (loss_cycle_ABA + loss_cycle_BAB)
        log_loss_D_Summarized += (loss_D_A + loss_D_B)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), f'{base}outputls/netG_A2B{epoch%2}.pth')
    torch.save(netG_B2A.state_dict(), f'{base}outputls/netG_B2A{epoch%2}.pth')
    torch.save(netD_A.state_dict(), f'{base}outputls/netD_A{epoch%2}.pth')
    torch.save(netD_B.state_dict(), f'{base}outputls/netD_B{epoch%2}.pth')
###################################

