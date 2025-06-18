import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from skimage import io

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
          convolutions.append(nn.BatchNorm2d(new_channels))
          #convolutions.append(PrintLayer())
          channels = new_channels
        flattened_dim = channels*(56-2*num_layers)*(56-2*num_layers) #final_channels*image_channels*final_width*final_height
        #print(f"final channels {channels}")
        self.encoder = nn.Sequential(*convolutions)
        self.flatten = Flatten()
        self.FC_input = nn.Linear(flattened_dim, 3*latent_dim)
        self.FC_mean  = nn.Linear(3*latent_dim, latent_dim)
        self.FC_var   = nn.Linear(3*latent_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        h = self.flatten(h)
        h = self.FC_input(h)
        mean     = self.FC_mean(h)
        log_var  = self.FC_var(h)                     # encoder produces mean and log of variance
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
          deconvs.append(nn.ConvTranspose2d(channels,new_channels, kernel_size=3))
          deconvs.append(nn.ReLU())
          deconvs.append(nn.BatchNorm2d(new_channels))
          #deconvs.append(PrintLayer()),
          channels = new_channels
        new_channels = max(channels - c_decrease, image_channels)
        deconvs.append(nn.ConvTranspose2d(channels,new_channels, kernel_size=3))
        deconvs.append(nn.Sigmoid())
        self.decoder = nn.Sequential(
            *deconvs
        )

    def forward(self, x):
        #print(x.shape)
        x = self.FC_hidden(x)
        x = self.unflatten(x)
        x_hat = self.decoder(x)
        return x_hat

class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.softplus = nn.Softplus()

    def reparameterization(self, mean, log_var,eps=1e-8):
        scale = self.softplus(log_var) + eps
        scale_tril = torch.diag_embed(scale)
        return torch.distributions.MultivariateNormal(mean, scale_tril=scale_tril).rsample()


    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, log_var) # takes exponential function (log var -> var)
        x_hat = self.decoder(z)

        return x_hat, mean, log_var
