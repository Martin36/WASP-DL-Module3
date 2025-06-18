import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import torchvision.transforms.v2 as transforms

from vae import PDDLDataset,Encoder,Decoder,Model

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau,LambdaLR

from datetime import datetime


dataset_path = 'data'

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")
batch_size = 80
image_channels = 3
latent_dim = 200
lr = 5e-3
epochs = 200


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

encoder = Encoder(image_channels,latent_dim,max_channels=12,num_layers=4)
decoder = Decoder(image_channels,latent_dim,12*48*48,48,max_channels=12,num_layers=4)

model = Model(encoder=encoder, decoder=decoder).to(DEVICE)

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss, KLD


optimizer = AdamW(model.parameters(), lr=lr)
plat_scheduler = ReduceLROnPlateau(optimizer, 'min',patience=5,cooldown=5,factor=0.6)

warmup_steps=50
def warmup_lambda(step):
        if step < warmup_steps:
            # Linear warmup: increases from 0 to 1 over warmup_steps
            return float(step) / float(max(1, warmup_steps))
        else:
            # After warmup, keep factor at 1.0 (or let subsequent scheduler take over)
            return 1.0

warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

print("Start training VAE...")
model.train()
beta_start = 0.0001
beta_end = 1.0
beta = beta_start
beta_anneal_epochs = 0.7*epochs
training_steps= 0
scheduler = warmup_scheduler
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
        training_steps += 1
    if training_steps == warmup_steps:
        scheduler = plat_scheduler
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
