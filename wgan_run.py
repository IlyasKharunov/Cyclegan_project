#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from torch.utils.tensorboard import SummaryWriter 
import re
import itertools
import timeit

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
from collections import namedtuple

from code.utils import ReplayBuffer
from code.utils import LambdaLR
from code.utils import weights_init_normal
from code.datasetsdrive import ImageDataset
from code.cyclegan_model import *
from collections import OrderedDict


# In[ ]:


Params = namedtuple('Params', ['startepoch', 'n_epochs', 
                               'batchSize', 'dataroot', 
                               'lr', 'decay_epoch', 
                               'size', 'input_nc', 
                               'output_nc', 'cuda', 
                               'n_cpu', 'resume', 
                               'gpu_ids'])


# In[ ]:


opt = Params(startepoch = 105, n_epochs = 200, 
             batchSize = 16, dataroot = 'database', 
             lr = 0.0002, decay_epoch = 100, 
             size = (180,324), input_nc = 3, 
             output_nc = 3, cuda = True, 
             n_cpu = 0, resume = True, 
             gpu_ids = [0,1,2,3])


# In[ ]:


#init network
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)


# In[ ]:


#init weights
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
    state_dicts.append(torch.load(f'{base}output_bestw/netD_A0.pth'))
    state_dicts.append(torch.load(f'{base}output_bestw/netD_B0.pth'))
    state_dicts.append(torch.load(f'{base}output_bestw/netG_A2B0.pth'))
    state_dicts.append(torch.load(f'{base}output_bestw/netG_B2A0.pth'))
    # create new OrderedDict that does not contain `module.`
    new_state_dicts = [OrderedDict() for i in range(4)]
    for i in range(len(state_dicts)):
        for k, v in state_dicts[i].items():
            name = k[7:] # remove `module.`
            new_state_dicts[i][name] = v
    # load params
    netD_A.load_state_dict(new_state_dicts[0])
    netD_B.load_state_dict(new_state_dicts[1])
    netG_A2B.load_state_dict(new_state_dicts[2])
    netG_B2A.load_state_dict(new_state_dicts[3])
    del state_dicts
    del new_state_dicts


# In[ ]:


#transfer to cuda device
netG_A2B.to(opt.gpu_ids[0])
netG_B2A.to(opt.gpu_ids[0])
netD_A.to(opt.gpu_ids[0])
netD_B.to(opt.gpu_ids[0])
netG_A2B = torch.nn.DataParallel(netG_A2B, opt.gpu_ids)
netG_B2A = torch.nn.DataParallel(netG_B2A, opt.gpu_ids)
netD_A = torch.nn.DataParallel(netD_A, opt.gpu_ids)
netD_B = torch.nn.DataParallel(netD_B, opt.gpu_ids)


# In[ ]:


# Lossess
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
trsfrms = [ transforms.Resize(opt.size, Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

dataset = ImageDataset(opt.dataroot, transforms_=trsfrms, unaligned=True)

dataloader = DataLoader(dataset, batch_size=opt.batchSize, 
                        shuffle=True, num_workers=opt.n_cpu, drop_last=True)


# In[ ]:


# Loss plot
logger = SummaryWriter(filename_suffix='first', log_dir='logs')


# In[ ]:


###### Training ######
log_loss_G_Summarized = 0
log_loss_G_Identity = 0
log_loss_G_GAN = 0
log_loss_G_Cycle = 0
log_loss_D_Summarized = 0
log_loss_D_GAN = 0
log_loss_D_Gradient_penalty = 0

shift = len(dataloader)
for epoch in range(opt.startepoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        #g1 = timeit.default_timer()
        real_A = input_A.copy_(batch['A'])
        real_B = input_B.copy_(batch['B'])

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*25.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*25.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = -torch.mean(pred_fake)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = -torch.mean(pred_fake)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*50.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*50.0

        # Total loss
        #o1 = timeit.default_timer()
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()
        #o2 = timeit.default_timer()
        #print(o1-o2)
        
        g2 = timeit.default_timer()
        #print(g2-g1)
        ###################################
        d1 = timeit.default_timer()
        ###### Discriminator A ######
        out_A = output_A.copy_(fake_A)
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        for _ in range(5):
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            # real loss part of wgan loss
            loss_D_real = -torch.mean(pred_real)

            # Fake loss
            pred_fake = netD_A(fake_A.detach())
            # fake loss part of wgan loss
            loss_D_fake = torch.mean(pred_fake)
            
            # gradient penalty
            eps = torch.rand((1),requires_grad=True, device = 'cuda')
            eps = eps.expand(real_A.size())
            x_tilde=eps*real_A+(1-eps)*fake_A.detach()
            pred_tilde=netD_A(x_tilde)
            gradients = torch.autograd.grad(outputs=pred_tilde, inputs=x_tilde,
                                  grad_outputs=torch.ones(pred_tilde.size(), device = 'cuda'),
                                    create_graph=True, retain_graph=None, only_inputs=True)[0]

            gradients = gradients.view(gradients.size(0), -1)
            D_A_gradient_penalty = 50 * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            # Total loss
            loss_D_A_GAN = loss_D_real + loss_D_fake
            loss_D_A = loss_D_A_GAN + D_A_gradient_penalty
            loss_D_A.backward()

            optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        out_B = output_B.copy_(fake_B)
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        for _ in range(5):
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            # real loss part of wgan loss
            loss_D_real = -torch.mean(pred_real)

            # Fake loss
            pred_fake = netD_B(fake_B.detach())
            # fake loss part of wgan loss
            loss_D_fake = torch.mean(pred_fake)

            eps = torch.rand((1),requires_grad=True, device = 'cuda')
            eps = eps.expand(real_B.size())
            x_tilde=eps*real_B+(1-eps)*fake_B.detach()
            pred_tilde=netD_B(x_tilde)
            gradients = torch.autograd.grad(outputs=pred_tilde, inputs=x_tilde,
                                  grad_outputs=torch.ones(pred_tilde.size(), device = 'cuda'),
                                    create_graph=True, retain_graph=None, only_inputs=True)[0]
            
            gradients = gradients.view(gradients.size(0), -1)
            D_B_gradient_penalty = 50 * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            # Total loss
            loss_D_B_GAN = loss_D_real + loss_D_fake
            loss_D_B = loss_D_B_GAN + D_B_gradient_penalty
            loss_D_B.backward()

            optimizer_D_B.step()
        ###################################
        #d2 = timeit.default_timer()
        #print(d2-d1)
        #l1 = timeit.default_timer()
        # Progress report
        if i % 10 == 0:
          logger.add_scalar("loss_G/Summarized", log_loss_G_Summarized/10, i + shift*epoch)
          logger.add_scalar("loss_G/Identity", log_loss_G_Identity/10, i + shift*epoch)
          logger.add_scalar("loss_G/GAN", log_loss_G_GAN/10, i + shift*epoch)
          logger.add_scalar("loss_G/Cycle", log_loss_G_Cycle/10, i + shift*epoch)
          logger.add_scalar("loss_D/Summarized", log_loss_D_Summarized/10, i + shift*epoch)
          logger.add_scalar("loss_D/GAN", log_loss_D_GAN/10, i + shift*epoch)
          logger.add_scalar("loss_D/Gradient penalty", log_loss_D_Gradient_penalty/10, i + shift*epoch)
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
          log_loss_D_GAN = 0
          log_loss_D_Gradient_penalty = 0

        log_loss_G_Summarized += loss_G
        log_loss_G_Identity += (loss_identity_A + loss_identity_B)
        log_loss_G_GAN += (loss_GAN_A2B + loss_GAN_B2A)
        log_loss_G_Cycle += (loss_cycle_ABA + loss_cycle_BAB)
        log_loss_D_Summarized += (loss_D_A + loss_D_B)
        log_loss_D_GAN += (loss_D_A_GAN + loss_D_B_GAN)
        log_loss_D_Gradient_penalty += (D_B_gradient_penalty + D_A_gradient_penalty)
        
        #l2 = timeit.default_timer()
        #print(t2-t1)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    print(f'{epoch} epoch ended')
    torch.save(netG_A2B.state_dict(), f'{base}outputwgp/netG_A2B{epoch%2}.pth')
    torch.save(netG_B2A.state_dict(), f'{base}outputwgp/netG_B2A{epoch%2}.pth')
    torch.save(netD_A.state_dict(), f'{base}outputwgp/netD_A{epoch%2}.pth')
    torch.save(netD_B.state_dict(), f'{base}outputwgp/netD_B{epoch%2}.pth')
###################################

