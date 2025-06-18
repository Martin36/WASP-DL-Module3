import argparse
import os
import torch
from einops import rearrange
from timm.utils.model_ema import ModelEmaV3
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import SokobanDataset
from diffusion_model import UNET, DDPMScheduler

arg_parser = argparse.ArgumentParser(description="Test a saved model")
arg_parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
arg_parser.add_argument("--data_folder", type=str, required=True, help="Directory containing the data")
arg_parser.add_argument("--output_folder", type=str, default=None, help="Directory to save the output images")
arg_parser.add_argument("--num_time_steps", type=int, default=1000, help="Number of time steps for the diffusion model")
arg_parser.add_argument("--ema_decay", type=float, default=0.9999, help="Decay rate for the EMA model")
arg_parser.add_argument("--grayscale", action='store_true', help="Use grayscale images (default is RGB)")
args = arg_parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

def display_reverse(images: list, output_folder: str | None = None, idx: int = 0):
  _, axes = plt.subplots(1, 10, figsize=(10,1))
  for i, ax in enumerate(axes.flat):
    x = images[i].squeeze(0)
    x = rearrange(x, 'c h w -> h w c')
    x = x.numpy()
    ax.imshow(x)
    ax.axis('off')
  plt.savefig(f'{output_folder}/reverse_images_{idx}.png') if output_folder is not None else plt.show()

def inference(checkpoint_path: str,
              data_folder: str,
              num_time_steps: int = 1000,
              ema_decay: float = 0.9999,
              output_folder: str | None = None,
              grayscale: bool = False):
    if output_folder is not None and not os.path.exists(output_folder):
      os.makedirs(output_folder)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    train_dataset = SokobanDataset(data_folder=data_folder, grayscale=grayscale)
    input_channels = train_dataset[0].shape[0]
    input_dimensions = train_dataset[0].shape

    if DEVICE == "cuda":
      model = UNET(input_channels=input_channels).cuda()
    else:
      model = UNET(input_channels=input_channels)
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPMScheduler(num_time_steps=num_time_steps)
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]
    images = []

    with torch.no_grad():
      model = ema.module.eval()
      for i in range(10):
        plt.close()
        z = torch.randn(1, *input_dimensions)
        pbar = tqdm(total=num_time_steps, desc=f"Generating image {i+1}/10")
        for t in reversed(range(1, num_time_steps)):
          pbar.update(1)
          t = [t]
          temp = (scheduler.beta[t] / ((torch.sqrt(1 - scheduler.alpha[t])) * (torch.sqrt(1 - scheduler.beta[t]))))
          if DEVICE == "cuda":
            z = (1 / (torch.sqrt(1 - scheduler.beta[t]))) * z - (temp * model(z.cuda(), t).cpu())
          else:
            z = (1 / (torch.sqrt(1 - scheduler.beta[t]))) * z - (temp * model(z, t).cpu())
          if t[0] in times:
            images.append(z)
          e = torch.randn(*z.shape)
          z = z + (e * torch.sqrt(scheduler.beta[t]))
        temp = scheduler.beta[0] / ((torch.sqrt(1 - scheduler.alpha[0])) * (torch.sqrt(1 - scheduler.beta[0])))
        if DEVICE == "cuda":
          x = (1 / (torch.sqrt(1 - scheduler.beta[0]))) * z - (temp * model(z.cuda(), [0]).cpu())
        else:
          x = (1 / (torch.sqrt(1 - scheduler.beta[0]))) * z - (temp * model(z, [0]).cpu())

        images.append(x)
        x = rearrange(x.squeeze(0), 'c h w -> h w c').detach()
        x = x.numpy()
        plt.imshow(x)
        plt.savefig(f'{output_folder}/image_{i}.png') if output_folder is not None else plt.show()
        display_reverse(images, output_folder=output_folder, idx=i)
        images = []

inference(args.checkpoint_path,
          args.data_folder,
          num_time_steps=args.num_time_steps,
          ema_decay=args.ema_decay,
          output_folder=args.output_folder,
          grayscale=args.grayscale)
