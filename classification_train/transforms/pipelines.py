import torchvision.transforms as T
import random

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

def get_augmentation_pipeline():
    return T.Compose([
        T.RandomApply([T.Lambda(lambda img: img.rotate(random.choice([90, 180, 270])))], p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.6, saturation=0.6, hue=0.1)], p=0.7),
        T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.2, 3.0))], p=0.3),
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

def get_validation_pipeline():
    return T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
