import matplotlib.pyplot as plt
import torch
from vae import Model,Encoder,Decoder,PDDLDataset
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from operator import add
import numpy as np

path = "src/model_2025-06-19_13-36-52_loss_1835p3415.pth"
dataset_path = 'data'
DEVICE = "cpu"
model_dict = torch.load(path,map_location=torch.device('cpu'))

grayscale = model_dict["grayscale"]
image_channels = 1 if grayscale else 3
latent_dim = model_dict["latent_dim"]
batch_size = model_dict["batch_size"]
max_channels = model_dict["max_channels"]
num_layers = model_dict["num_layers"]
latent_image_size = model_dict["latent_image_size"]
repr_loss = model_dict["repr_loss"]
kld_loss = model_dict["kld_loss"]

encoder = Encoder(image_channels,latent_dim,max_channels=max_channels,num_layers=num_layers)
decoder = Decoder(image_channels,latent_dim,max_channels*latent_image_size*latent_image_size,latent_image_size,max_channels=max_channels,num_layers=num_layers)

mnist_transform = transforms.Compose([
      transforms.ToPILImage(),
      #transforms.Grayscale(),
      transforms.RGB(),
      transforms.ToImage(),
      transforms.ToDtype(torch.float32, scale=True),
])

mnist_transform_grayscale = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Grayscale(),
      transforms.ToImage(),
      transforms.ToDtype(torch.float32, scale=True),
])
train_dataset = PDDLDataset(dataset_path, transform=mnist_transform_grayscale) if grayscale else PDDLDataset(dataset_path, transform=mnist_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)#, **kwargs)

model = Model(encoder=encoder, decoder=decoder).to("cpu")
model.load_state_dict(torch.load(path,map_location=torch.device('cpu'))["state_dict"])
model.eval()


def plot_training():
  plt.plot(repr_loss,label="repr loss")
  plt.plot(kld_loss,label="kld loss")
  plt.plot(list(map(add,repr_loss,kld_loss)),label="overall loss")
  plt.xlabel("epochs")
  plt.ylabel("loss")
  plt.yscale('log')
  plt.legend()
  name = "vae_learning_gray.png" if grayscale else "vae_learning.png"
  plt.savefig(name,dpi=300)
  #plt.show()

plot_training()


def show_images_with_ground_truth(x, x_hat, idx=[4,14,36,49]):
    fig, axes = plt.subplots(3, len(idx), figsize=(len(idx)* 2, 4))
    for i,id in enumerate(idx):
      ax_orig = axes[0,i]
      if grayscale:
        ax_orig.imshow(x[id].cpu().numpy().squeeze())
      else:
        ax_orig.imshow(x[id].cpu().permute(1,2,0).numpy())
      ax_orig.set_title("Original")
      ax_orig.axis('off')

      ax_recon = axes[1,i]
      if grayscale:
        ax_recon.imshow(x_hat[id].cpu().numpy().squeeze())
      else:
        ax_recon.imshow(x_hat[id].cpu().permute(1,2,0).numpy())
      ax_recon.set_title("Reconstructed")
      ax_recon.axis('off')

      ax_diff = axes[2,i]
      if grayscale:
        ax_diff.imshow(np.abs(x[id].cpu().numpy().squeeze() - x_hat[id].cpu().numpy().squeeze()))
      else:
        ax_diff.imshow(np.abs(x[id].cpu().permute(1,2,0).numpy() - x_hat[id].cpu().permute(1,2,0).numpy()))
      ax_diff.set_title("Difference")
      ax_diff.axis('off')
    plt.tight_layout()
    name = "vae_reconstructions_gray.png" if grayscale else "vae_reconstructions.png"
    plt.savefig(name,dpi=300)
    #plt.show()

def show_images(x_hat,name,idx=[4,12,14,36,49,55]):
  fig, axes = plt.subplots(1,len(idx),figsize=(len(idx)*3,4))
  for i,id in enumerate(idx):
    ax = axes[i]
    ax.imshow(x_hat[id].cpu().permute(1,2,0).numpy())
    ax.axis('off')
  plt.tight_layout()
  plt.savefig(name,dpi=300)
  #plt.show()

def show_images_2rows(x_hat,name,idx=[4,7,12,14,24,29,36,49,55,78]):
  width = len(idx)//2
  fig, axes = plt.subplots(2,width,figsize=(width*2,4))
  for i,id in enumerate(idx):
    ax = axes[i//width,i%width]
    ax.imshow(x_hat[id].cpu().permute(1,2,0).numpy())
    ax.axis('off')
  plt.tight_layout()
  plt.savefig(name,dpi=300)

model.eval()
encoder.eval()
decoder.eval()

with torch.no_grad():
    x = next(iter(train_loader)).to(DEVICE)
    x_hat, _, _ = model(x)
    z,_ = encoder(x)

show_images_with_ground_truth(x, x_hat)
with torch.no_grad():
  noise = torch.randn(batch_size, latent_dim).to(DEVICE)
  generated_images = decoder(noise)
name = "vae_generations_gray.png" if grayscale else "vae_generations.png"
show_images_2rows(generated_images,name)

def interpolate_latent_space(z1, z2,name, n_steps=10):

    with torch.no_grad():
        # Perform linear interpolation in the latent space
        interpolated_latents = torch.zeros(n_steps, *z1.shape).to(DEVICE)
        for i in range(n_steps):
            alpha = i / (n_steps - 1) # Alpha goes from 0 to 1
            value = (1 - alpha) * z1 + alpha * z2
            interpolated_latents[i] = value

        #print(f"Interpolated latents shape: {interpolated_latents.shape}")

        # Decode the interpolated latent vectors back into images
        interpolated_images = decoder(interpolated_latents)

        # Visualize the interpolated images
        show_images(interpolated_images,name, idx=list(range(n_steps)))
name = "vae_interpolation_neighbours_gray.png" if grayscale else "vae_interpolation_neighbours.png"
interpolate_latent_space(z[0],z[1], name, n_steps=6)
name = "vae_interpolation_far_gray.png" if grayscale else "vae_interpolation_far.png"
interpolate_latent_space(z[28],z[75],name,n_steps=6)
