import os
from glob import glob

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from transformers import SegformerForSemanticSegmentation, SegformerConfig

ORIGINAL_ROOT = "/home/giorgio/Scrivania/Fabry/Test/Segmentation/Original"
MODEL1_PATH   = "glom_segformerb4_best_model_fold4-Copy1.pth"
MODEL2_PATH   = "segformer_b4_final_model.pth"
OUTPUT_SIZE = (128, 128)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_segformer(model_path):
    config = SegformerConfig.from_pretrained(
        "nvidia/segformer-b4-finetuned-ade-512-512",
        num_labels=1
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b4-finetuned-ade-512-512",
        config=config,
        ignore_mismatched_sizes=True
    )
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model

model1 = load_segformer(MODEL1_PATH)
model2 = load_segformer(MODEL2_PATH)

preprocess_img = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
])

def preprocess_for_model(img: Image.Image):
    x = preprocess_img(img).unsqueeze(0).to(DEVICE)
    return x

def infer_mask_area(model, img_tensor):
    with torch.no_grad():
        logits = model(img_tensor).logits
        logits = F.interpolate(logits, size=OUTPUT_SIZE, mode="bilinear", align_corners=False)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).float()
        area = mask.sum().item()
    return area

results = {}

for subdir in sorted(os.listdir(ORIGINAL_ROOT)):
    subpath = os.path.join(ORIGINAL_ROOT, subdir)
    if not os.path.isdir(subpath):
        continue

    ratios = []
    for img_path in glob(os.path.join(subpath, "*.jpg")):
        img = Image.open(img_path).convert("RGB")
        x = preprocess_for_model(img)

        area1 = infer_mask_area(model1, x)
        area2 = infer_mask_area(model2, x)

        if area1 > 0:
            ratios.append(area2 / area1)
        else:
            continue

    if ratios:
        mean_ratio = sum(ratios) / len(ratios)
    else:
        mean_ratio = float('nan')

    results[subdir] = {
        "n_images": len(ratios),
        "mean_ratio": mean_ratio,
        "all_ratios": ratios,
    }

for subdir, stats in results.items():
    print(f"{subdir:30s}  images: {stats['n_images']:3d}  mean(area2/area1) = {stats['mean_ratio']:.4f}")
