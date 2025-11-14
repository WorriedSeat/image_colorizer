import os
from PIL import Image
import numpy as np
from skimage import color
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

#Class that represents our dataset
class EuroSATLabDataset(Dataset):
    def __init__(self, root, image_size=64):
        self.paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".jpg")]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        img = img.permute(1,2,0).numpy()  # HWC
        lab = color.rgb2lab(img).astype(np.float32)
        L = lab[:,:,0:1] / 100.0
        ab = lab[:,:,1:] / 128.0
        L = torch.from_numpy(L).permute(2,0,1)
        ab = torch.from_numpy(ab).permute(2,0,1)
        return L, ab

#Function that use dataset class and creates data loaders
def get_dataloaders(path: str='data/prep/EuroSAT_RGB/images', batch_size:int=32, train_part:float=0.8, seed:int=69):
    dataset = EuroSATLabDataset(path)

    # Split on train and test
    train_size = int(train_part * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    print(f"Successfully created data loaders:\n\ttrain: {len(train_loader)}\n\tval: {len(val_loader)}")
    return train_loader, val_loader