import argparse
import random
import torch
import torch.optim as optim
import torch.nn as nn
import os

from tqdm import tqdm
from timm.utils.model_ema import ModelEmaV3
from torch.utils.data import DataLoader 
from dataset import SokobanDataset
from diffusion_model import UNET, DDPMScheduler
from utils import set_seed

arg_parser = argparse.ArgumentParser(description="Train a model using generated Sokoban images")
arg_parser.add_argument("--data_folder", type=str, required=True, help="Directory containing the generated Sokoban images")
arg_parser.add_argument("--output_folder", type=str, required=True, help="Directory to save the model checkpoints")
arg_parser.add_argument("--model", type=str, choices=["diffusion"], required=True, help="Model to train")
arg_parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
arg_parser.add_argument("--num_time_steps", type=int, default=1000, help="Number of time steps for the diffusion model")
arg_parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs for training")
arg_parser.add_argument("--seed", type=int, default=-1, help="Random seed for reproducibility")
arg_parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for the optimizer")
arg_parser.add_argument("--ema_decay", type=float, default=0.9999, help="Decay rate for the EMA model")
args = arg_parser.parse_args()

class Args:
  def __init__(self, args: argparse.Namespace):
    self.data_folder = args.data_folder
    self.output_folder = args.output_folder
    self.model = args.model
    self.batch_size = args.batch_size
    self.num_time_steps = args.num_time_steps
    self.num_epochs = args.num_epochs
    self.seed = args.seed
    self.lr = args.lr
    self.ema_decay = args.ema_decay

args = Args(args)

class Trainer:
  def __init__(self, args: Args, checkpoint_path: str | None = None):
    self._args = args
    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self._checkpoint_path = checkpoint_path
    
    set_seed(random.randint(0, 2**32-1)) if args.seed == -1 else set_seed(args.seed)

  def train(self):

    # Create output directory if it doesn't exist
    if not os.path.exists(self._args.output_folder):
      os.makedirs(self._args.output_folder)

    train_dataset = SokobanDataset(data_folder=self._args.data_folder)
    train_loader = DataLoader(train_dataset, batch_size=self._args.batch_size, shuffle=True, drop_last=True, num_workers=4)

    # train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    # train_loader = DataLoader(train_dataset, batch_size=self._args.batch_size, shuffle=True, drop_last=True, num_workers=4)

    input_channels = train_dataset[0].shape[0]
    scheduler = DDPMScheduler(num_time_steps=self._args.num_time_steps)
    model = UNET(input_channels=input_channels).to(self._device)
    optimizer = optim.Adam(model.parameters(), lr=self._args.lr)
    ema = ModelEmaV3(model, decay=self._args.ema_decay)
    if self._checkpoint_path is not None:
      checkpoint = torch.load(self._checkpoint_path)
      model.load_state_dict(checkpoint['weights'])
      ema.load_state_dict(checkpoint['ema'])
      optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.MSELoss(reduction='mean')
    best_loss = float('inf')

    for i in range(self._args.num_epochs):
      total_loss = 0
      for x in tqdm(train_loader, desc=f"Epoch {i+1}/{self._args.num_epochs}"):
        x = x.to(self._device)
        # x = F.pad(x, (2, 2, 2, 2))
        t = torch.randint(0, self._args.num_time_steps, (self._args.batch_size,))
        e = torch.randn_like(x, requires_grad=False)
        a = scheduler.alpha[t].view(self._args.batch_size, 1, 1, 1).to(self._device)
        x = torch.sqrt(a) * x + torch.sqrt(1 - a) * e
        output = model(x, t)
        optimizer.zero_grad()
        loss = criterion(output, e)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        ema.update(model)
      epoch_loss = total_loss / (len(train_loader) / self._args.batch_size)
      print(f'Epoch {i+1} | Loss {epoch_loss:.5f}')
      if epoch_loss < best_loss:
        best_loss = epoch_loss
        print(f'New best loss: {best_loss:.5f}, saving model...')
        # Save the model with the best loss
        torch.save(model.state_dict(), f'{self._args.output_folder}/best_model.pth')
      # print(f'Epoch {i+1} | Loss {total_loss / (60000 / self._args.batch_size):.5f}')
      # Save the checkpoint after each epoch
      torch.save(checkpoint, f'{self._args.output_folder}/latest.pth')

    checkpoint = {
      'weights': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'ema': ema.state_dict()
    }
    torch.save(checkpoint, 'checkpoints/ddpm_checkpoint')

trainer = Trainer(args)
trainer.train()