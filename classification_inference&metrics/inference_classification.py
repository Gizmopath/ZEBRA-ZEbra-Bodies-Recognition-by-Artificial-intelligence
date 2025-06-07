import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, models

test_root = "/home/giorgio/Scrivania/Fabry/Test/Classification"
model_path = "efficientnet_final_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
num_classes = 2

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

class InferenceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "auc": roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else float('nan'),
        "confusion_matrix": cm
    }

def plot_conf_matrix(cm, title):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(title)
    plt.show()

model = models.efficientnet_b2(pretrained=False)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

def gather_test_images(root_dir):
    image_paths, labels, case_infos = [], [], []
    for class_folder in ["Negative", "Positive"]:
        label = 0 if class_folder == "Negative" else 1
        class_path = os.path.join(root_dir, class_folder)
        for subfolder in os.listdir(class_path):
            subfolder_path = os.path.join(class_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            sub_imgs = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith(".jpg")]
            image_paths.extend(sub_imgs)
            labels.extend([label] * len(sub_imgs))
            case_infos.append((subfolder, label, sub_imgs))
    return image_paths, labels, case_infos

all_paths, all_labels, case_infos = gather_test_images(test_root)
dataset = InferenceDataset(all_paths, all_labels, transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

all_preds = []
with torch.no_grad():
    for imgs, _ in tqdm(loader, desc="Global Inference"):
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())

global_metrics = calculate_metrics(all_labels, all_preds)
print("\nGlobal Metrics:")
for k, v in global_metrics.items():
    if k != "confusion_matrix":
        print(f"{k}: {v:.4f}")
plot_conf_matrix(global_metrics["confusion_matrix"], "Confusion Matrix - All Test Set")

print("\nPer-folder Metrics:")
for subfolder_name, label, img_paths in case_infos:
    if not img_paths:
        continue
    dataset = InferenceDataset(img_paths, [label] * len(img_paths), transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            pred = torch.argmax(outputs, dim=1)
            preds.extend(pred.cpu().numpy())
    metrics = calculate_metrics([label] * len(img_paths), preds)
    print(f"\nMetrics for {subfolder_name} (label {label}):")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")
    plot_conf_matrix(metrics["confusion_matrix"], f"Conf Matrix - {subfolder_name}")
