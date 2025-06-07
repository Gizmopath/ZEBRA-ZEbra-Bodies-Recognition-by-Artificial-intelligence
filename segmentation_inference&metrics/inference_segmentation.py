import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
import pickle
from transformers import SegformerForSemanticSegmentation, SegformerConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

original_folder = "/home/giorgio/Scrivania/Fabry/Test/Segmentation/Original"
mask_folder = "/home/giorgio/Scrivania/Fabry/Test/Segmentation/Mask"
batch_size = 4
num_classes = 1
resize_size = (512, 512)
output_size = (128, 128)

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
            img = TF.to_tensor(img)
            msk = TF.to_tensor(msk)
        return img, msk

def dice_coefficient(pred, target, eps=1e-8):
    intersection = (pred * target).sum()
    return (2. * intersection + eps) / (pred.sum() + target.sum() + eps)

def iou_score(pred, target, eps=1e-8):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)

orig_cases = [os.path.join(original_folder, d) for d in os.listdir(original_folder) if os.path.isdir(os.path.join(original_folder, d))]
image_paths, mask_paths = [], []

for case_path in orig_cases:
    case_name = os.path.basename(case_path)
    for im in os.listdir(case_path):
        if im.endswith('.jpg'):
            image_paths.append(os.path.join(original_folder, case_name, im))
            mask_paths.append(os.path.join(mask_folder, case_name, im.replace('.jpg', '.tif')))

dataset = SegmentationDataset(image_paths, mask_paths)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

config = SegformerConfig.from_pretrained(
    "nvidia/segformer-b4-finetuned-ade-512-512",
    num_labels=1
)
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b4-finetuned-ade-512-512",
    config=config,
    ignore_mismatched_sizes=True
).to(device)

model.load_state_dict(torch.load("segformer_noaug_b4_final_model.pth"))
model.eval()

history = {'test_dice': [], 'test_iou': []}

with torch.no_grad():
    tot_dice = tot_iou = samples = 0
    for imgs, msks in tqdm(loader, desc="Inference"):
        imgs, msks = imgs.to(device), msks.to(device)
        outputs = model(imgs).logits
        outputs = torch.nn.functional.interpolate(outputs, size=msks.shape[2:], mode="bilinear", align_corners=False)

        preds = torch.sigmoid(outputs) > 0.5
        preds_f, msks_f = preds.float(), msks.float()

        dice = dice_coefficient(preds_f, msks_f)
        iou = iou_score(preds_f, msks_f)

        tot_dice += dice.item() * imgs.size(0)
        tot_iou += iou.item() * imgs.size(0)
        samples += imgs.size(0)

    test_dice = tot_dice / samples
    test_iou = tot_iou / samples
    print(f"Test Dice: {test_dice:.4f} IoU: {test_iou:.4f}")

    history['test_dice'].append(test_dice)
    history['test_iou'].append(test_iou)

with open("segformer_b4_inference_history.pkl", 'wb') as f:
    pickle.dump(history, f)
