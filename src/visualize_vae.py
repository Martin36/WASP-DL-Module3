import matplotlib.pyplot as plt
import torch
from vae import Model,Encoder,Decoder,PDDLDataset
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms


image_channels = 3
latent_dim = 200
batch_size = 80

path = "src/model_2025-06-18_15-59-45_loss_4495p3166.pth"
dataset_path = 'data'

encoder = Encoder(image_channels,latent_dim,max_channels=12,num_layers=4)
decoder = Decoder(image_channels,latent_dim,12*48*48,48,max_channels=12,num_layers=4)

mnist_transform = transforms.Compose([
      transforms.ToPILImage(),
      #transforms.Grayscale(),
      transforms.RGB(),
      #transforms.Resize((64,64)),
      transforms.ToImage(),
      transforms.ToDtype(torch.float32, scale=True),
])
train_dataset = PDDLDataset(dataset_path, transform=mnist_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

model = Model(encoder=encoder, decoder=decoder).to("cpu")
model.load_state_dict(torch.load(path,map_location=torch.device('cpu'))["state_dict"])
model.eval()


def show_images_with_ground_truth(x, x_hat, idx=[4,14,36,49]):
    fig, axes = plt.subplots(2, len(idx), figsize=(len(idx)* 2, 4))
    for i,id in enumerate(idx):
      ax_orig = axes[0,i]
      ax_orig.imshow(x[id].cpu().permute(1,2,0).numpy())
      ax_orig.set_title("Original")
      ax_orig.axis('off')

      ax_recon = axes[1,i]
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
    x = next(iter(train_loader)).to("cpu")
    x_hat, _, _ = model(x)

show_images_with_ground_truth(x, x_hat)
with torch.no_grad():
  noise = torch.randn(batch_size, latent_dim).to("cpu")
  generated_images = decoder(noise)
show_images(generated_images)
