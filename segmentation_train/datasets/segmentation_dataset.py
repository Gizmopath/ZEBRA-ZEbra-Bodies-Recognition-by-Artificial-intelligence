import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms as T

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
resize_size = (512, 512)
output_size = (128, 128)

def to_tensor_and_normalize(img):
    t = TF.to_tensor(img)
    return normalize(t)

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        msk = Image.open(self.mask_paths[idx]).convert("L")
        if self.transform:
            img, msk = self.transform(img, msk)
        else:
            img = TF.resize(img, resize_size)
            msk = TF.resize(msk, output_size, interpolation=TF.InterpolationMode.NEAREST)
            img = to_tensor_and_normalize(img)
            msk = TF.to_tensor(msk)
        return img, msk
