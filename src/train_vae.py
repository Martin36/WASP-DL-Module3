import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid

from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import os

import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

import math

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datetime import datetime


dataset_path = 'data'

cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")
batch_size = 80
image_channels = 3
latent_dim = 200
lr = 1e-2
epochs = 200


class PDDLDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.num_images = len([name for name in os.listdir(root_dir)])

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,f"frame_{idx:05d}.png")
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image
    
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x
    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Encoder(nn.Module):

    def __init__(self, image_channels, latent_dim, max_channels,num_layers,c_increase=3):
        super(Encoder, self).__init__()
        convolutions = []
        channels = image_channels
        for i in range(num_layers):
          new_channels = min(channels + c_increase,max_channels)
          convolutions.append(nn.Conv2d(channels, new_channels, kernel_size=3)),
          convolutions.append(nn.ReLU())
          #convolutions.append(PrintLayer())
          channels = new_channels
        flattened_dim = channels*(56-2*num_layers)*(56-2*num_layers) #final_channels*image_channels*final_width*final_height
        print(f"final image size: {56-2*num_layers}")
        print(f"flattened dim: {flattened_dim}")
        print(f"max channels: {channels}")
        #print(f"final channels {channels}")
        self.encoder = nn.Sequential(*convolutions)
        self.flatten = Flatten()
        self.FC_input = nn.Linear(flattened_dim, 3*latent_dim)
        self.FC_mean  = nn.Linear(3*latent_dim, latent_dim)
        self.FC_var   = nn.Linear(3*latent_dim, latent_dim)

        self.training = True

    def forward(self, x):
        h = self.encoder(x)
        h = self.flatten(h)
        h_ = self.FC_input(h)
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance
                                                       #             (i.e., parateters of simple tractable normal distribution "q"

        return mean, log_var
    
class UnFlatten(nn.Module):
    def __init__(self, channels, height, width):
      super(UnFlatten, self).__init__()
      self.channels = channels
      self.height = height
      self.width = width

    def forward(self, input):
        return input.view(-1, self.channels, self.height, self.width)

class Decoder(nn.Module):
    def __init__(self, image_channels, latent_dim, flattened_dim, inner_image, max_channels, num_layers,c_decrease=3):
        super(Decoder, self).__init__()
        #print(f"decoder max channels: {max_channels}")
        self.FC_hidden = nn.Linear(latent_dim, flattened_dim)
        channels = max_channels
        self.unflatten = nn.Sequential(UnFlatten(channels,inner_image,inner_image),nn.ReLU())

        deconvs = [] #[PrintLayer()]

        for i in range(num_layers-1):
          new_channels = max(channels - c_decrease, image_channels)
          deconvs.append(nn.ConvTranspose2d(channels,new_channels, kernel_size=3)),
          deconvs.append(nn.ReLU()),
          #deconvs.append(PrintLayer()),
          channels = new_channels
        new_channels = max(channels - c_decrease, image_channels)
        deconvs.append(nn.ConvTranspose2d(channels,new_channels, kernel_size=3)),
        deconvs.append(nn.Sigmoid())
        self.decoder = nn.Sequential(
            *deconvs
        )

    def forward(self, x):
        #print(x.shape)
        x = self.FC_hidden(x)
        #print(x.shape)
        x = self.unflatten(x)
        x_hat = self.decoder(x)
        return x_hat
    
class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon
        z = mean + var*epsilon                          # reparameterization trick
        return z


    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var
    
    
mnist_transform = transforms.Compose([
      transforms.ToPILImage(),
      #transforms.Grayscale(),
      transforms.RGB(),
      #transforms.Resize((64,64)),
      transforms.ToImage(),
      transforms.ToDtype(torch.float32, scale=True),
])

#kwargs = {'num_workers': 1, 'pin_memory': True}

train_dataset = PDDLDataset(dataset_path, transform=mnist_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)#, **kwargs)

encoder = Encoder(image_channels,latent_dim,max_channels=12,num_layers=12)
decoder = Decoder(image_channels,latent_dim,12*32*32,32,max_channels=12,num_layers=12,c_decrease=1)

model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss, KLD


optimizer = AdamW(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, 'min',patience=5,cooldown=5,factor=0.6)

print("Start training VAE...")
model.train()
print(lr)
beta_start = 0.0
beta_end = 1.0
beta = beta_start
beta_anneal_epochs = 0.7*epochs
for epoch in range(epochs):
    o_repr_loss = 0
    o_kld_loss = 0
    for batch_idx, x in enumerate(train_loader):
        #print(f"Batch {batch_idx}: Inpute shape: {x.shape}")
        #x = x.view(-1, x_dim).to(DEVICE)
        x = x.to(DEVICE)
        optimizer.zero_grad()
        x_hat, mean, log_var = model(x)
        repr,kld = loss_function(x, x_hat, mean, log_var)
        o_repr_loss += repr.item()
        o_kld_loss += kld.item()
        loss = repr+kld*beta
        loss.backward()
        optimizer.step()
    scheduler.step((o_repr_loss + o_kld_loss) / (batch_idx*batch_size))
    if epoch < beta_anneal_epochs:
        beta = beta_start + (beta_end - beta_start) * (epoch / beta_anneal_epochs)
    else:
        beta = beta_end
    print(f"\tEpoch: {epoch + 1}\tAverage Loss: {o_kld_loss / (batch_idx*batch_size):.2f},{o_repr_loss / (batch_idx*batch_size):.2f} \t lr: {scheduler.get_last_lr()} \t kld_scale: {beta:.2f}")

print("Finish!!")
final_loss = (o_kld_loss + o_repr_loss) / (batch_idx*batch_size)
loss_str = f"{final_loss:.4f}".replace('.', 'p')
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"model_{timestamp}_loss_{loss_str}.pth"
torch.save({"state_dict": model.state_dict()}, filename)

