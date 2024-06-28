import os
import argparse
import math
import torch
from datasets import load_dataset
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import *
from utils import setup_seed, FocalFrequencyLoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')
    parser.add_argument('--load_model_path', type=str, default='')
    parser.add_argument('--start_epoch', type=int, default=0)

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    channels = args.channels
    image_size = 64
    patch_size = 8 # how big are the patches you divide the data in
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    ################################# Choose the dataset #################################################################################
    dataset = load_dataset("imagenet-1k")
    train_data = dataset["train"]
    val_data = dataset["validation"]
    
    class ImageNetDataset(Dataset):
        def __init__(self, data, transform=None):
            self.data = data
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            image = self.data[index]['image']
            label = self.data[index]['label']

            if self.transform:
                image = self.transform(image).squeeze()
            
            real_data = image.real
            imag_data = image.imag
            complex_data = torch.stack((real_data, imag_data), dim=0)

            return complex_data
        
    class FourierTransform:
        def __call__(self, image):
            fft_image = torch.fft.fft2(image)
            fft_shifted = torch.fft.fftshift(fft_image)
            
            return fft_shifted

    transform_fft = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        FourierTransform(),
        ])
    train_dataset = ImageNetDataset(train_data, transform=transform_fft)
    val_dataset = ImageNetDataset(val_data, transform=transform_fft)
    val_loader = torch.utils.data.DataLoader(val_dataset, 32, shuffle=False, num_workers=10)

    ##############
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=10)
    writer = SummaryWriter(os.path.join('logs', 'imagenet1k', 'mae-pretrain'))
    if args.load_model_path:
        model = torch.load(args.load_model_path)
        model = model.to(device)
    else:
        model = MAE_ViT(mask_ratio=args.mask_ratio, image_size=image_size, patch_size=patch_size).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    # add focal frequency loss
    focal_freq_loss = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0, ave_spectrum=False, log_matrix=False, batch_matrix=False)

    step_count = 0
    optim.zero_grad()
    for e in range(args.start_epoch, args.total_epoch):
        model.train()
        losses = []
        for img in tqdm(iter(dataloader)):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)

            loss = focal_freq_loss(predicted_img, img)
            # loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio # Old Loss

            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        # with torch.no_grad():
        #     val_img = torch.stack([val_dataset[i][0] for i in range(16)])
        #     val_img = val_img.to(device)
        #     predicted_val_img, mask = model(val_img)
        #     predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
        #     print(val_img.shape, predicted_val_img.shape)
        #     img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
        #     print(img.shape)
        #     img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
        #     writer.add_image('mae_image', (img + 1) / 2, global_step=e)
        with torch.no_grad():
            val_images = []

            # Iterate over the val_loader
            for batch in val_loader:
                val_images.append(batch)
                
                # Break the loop once we have collected 16 images
                if len(val_images) * batch.size(0) >= 16:
                    break
            val_img = torch.cat(val_images, dim=0)[:16]
            print("val:", val_img.shape)
        
            val_img = val_img.to(device)
            predicted_val_img, mask = model(val_img)
            masked = val_img * (1 - mask)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            val_img = torch.complex(val_img[:,0,:,:], val_img[:,1,:,:])
            val_img=torch.fft.ifft2(val_img)
            val_img = torch.abs(val_img)
            predicted_val_img = torch.complex(predicted_val_img[:,0,:,:], predicted_val_img[:,1,:,:])
            predicted_val_img=torch.fft.ifft2(predicted_val_img)
            predicted_val_img = torch.abs(predicted_val_img)

            
            masked = torch.complex(masked[:,0,:,:], masked[:,1,:,:])
            masked=torch.fft.ifft2(masked)
            masked = torch.abs(masked)
            masked = masked.unsqueeze(1)
            predicted_val_img = predicted_val_img.unsqueeze(1)
            val_img = val_img.unsqueeze(1)
            print(val_img.shape, predicted_val_img.shape)
            img = torch.cat([masked, predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            writer.add_image('mae_image', (img + 1) / 2, global_step=e)
        ''' save model '''
        torch.save(model, args.model_path)

        # Command zum Starten:
        # python mae_pretrain.py --load_model_path vit-t-mae.pt --start_epoch <xx>
        # Last Epoch: 29