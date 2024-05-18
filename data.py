from utils import load_image, load_mask, load_tensor
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class BUSIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, size=256, device='cuda'):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.device = device
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image, mask = load_image(img_path, size=self.size), load_mask(mask_path, size=self.size)

        if self.transform is not None:            
            image = image.transpose(1, 2, 0)
            mask = mask.transpose(1, 2, 0)
            
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

            image = image.transpose(2, 0, 1)
            mask = mask.transpose(2, 0, 1)

        image = torch.tensor(image).to(device=self.device, dtype=torch.float)
        mask = torch.tensor(mask).to(self.device, dtype=torch.float)
        return image, mask