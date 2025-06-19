import matplotlib.pyplot as plt
import torch
from vae import Model,Encoder,Decoder,PDDLDataset
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms

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


def show_images_with_ground_truth(x, x_hat, idx=[4,14,36,49]):
    fig, axes = plt.subplots(2, len(idx), figsize=(len(idx)* 2, 4))
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
    plt.tight_layout()
    plt.show()

def show_images(x_hat,idx=[4,14,36,49]):
  fig, axes = plt.subplots(1,len(idx),figsize=(len(idx)*3,4))
  for i,id in enumerate(idx):
    ax = axes[i]
    ax.imshow(x_hat[id].cpu().permute(1,2,0).numpy())
    ax.axis('off')
  plt.show()

model.eval()

with torch.no_grad():
    x = next(iter(train_loader)).to(DEVICE)
    x_hat, _, _ = model(x)
    z,_ = encoder(x)

show_images_with_ground_truth(x, x_hat)
with torch.no_grad():
  noise = torch.randn(batch_size, latent_dim).to(DEVICE)
  generated_images = decoder(noise)
show_images(generated_images)

def interpolate_latent_space(z1, z2, n_steps=10):

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
        show_images(interpolated_images, idx=list(range(n_steps)))

interpolate_latent_space(z[0],z[1],n_steps=6)
interpolate_latent_space(z[28],z[75],n_steps=6)
