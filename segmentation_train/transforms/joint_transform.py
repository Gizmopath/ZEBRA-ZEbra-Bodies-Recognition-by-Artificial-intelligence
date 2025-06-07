import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision import transforms as T

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
resize_size = (512, 512)
output_size = (128, 128)

def to_tensor_and_normalize(img):
    t = TF.to_tensor(img)
    return normalize(t)

class JointTransform:
    def __init__(self, size=resize_size, mask_size=output_size):
        self.size = size
        self.mask_size = mask_size
        self.color_jitter = T.ColorJitter(brightness=0.1, contrast=0.6, saturation=0.6, hue=0.1)
        self.blur = T.GaussianBlur(kernel_size=(3,3), sigma=(0.2,3.0))

    def __call__(self, img, msk):
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            img = TF.rotate(img, angle)
            msk = TF.rotate(msk, angle)
        if random.random() < 0.5:
            img = TF.hflip(img); msk = TF.hflip(msk)
        if random.random() < 0.5:
            img = TF.vflip(img); msk = TF.vflip(msk)
        if random.random() < 0.7:
            img = self.color_jitter(img)
        if random.random() < 0.3:
            img = self.blur(img)
        img = TF.resize(img, self.size)
        msk = TF.resize(msk, self.mask_size, interpolation=TF.InterpolationMode.NEAREST)
        img = to_tensor_and_normalize(img)
        msk = TF.to_tensor(msk)
        return img, msk
