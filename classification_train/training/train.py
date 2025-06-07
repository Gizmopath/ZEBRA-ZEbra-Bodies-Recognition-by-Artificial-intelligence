import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm

from datasets.custom_dataset import CustomImageDataset
from transforms.pipelines import get_augmentation_pipeline, get_validation_pipeline
from models.efficientnet_b2 import get_efficientnet_b2

def split_data_kfold(cases, labels, n_splits):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = []
    for train_idx, val_idx in kf.split(cases):
        splits.append((
            [cases[i] for i in train_idx],
            [labels[i] for i in train_idx],
            [cases[i] for i in val_idx],
            [labels[i] for i in val_idx]
        ))
    return splits

def train_model(positive_folder, negative_folder, output_dir, num_epochs=40, batch_size=64, learning_rate=1e-4, num_classes=2, patience=5, sched_patience=3, n_splits=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    positive_cases = [os.path.join(positive_folder, d) for d in os.listdir(positive_folder) if os.path.isdir(os.path.join(positive_folder, d))]
    negative_cases = [os.path.join(negative_folder, d) for d in os.listdir(negative_folder) if os.path.isdir(os.path.join(negative_folder, d))]
    all_cases = positive_cases + negative_cases
    case_labels = [1] * len(positive_cases) + [0] * len(negative_cases)

    transform_train = get_augmentation_pipeline()
    transform_val = get_validation_pipeline()

    history = {k: [] for k in [
        "train_loss","train_accuracy","train_precision","train_recall","train_f1","train_auc",
        "val_loss","val_accuracy","val_precision","val_recall","val_f1","val_auc"]}

    splits = split_data_kfold(all_cases, case_labels, n_splits)

    for fold, (train_cases, train_labels_cases, val_cases, val_labels_cases) in enumerate(splits, 1):
        print(f"\n--- Training fold {fold}/{n_splits} ---")

        train_images, train_labels = [], []
        for case, lbl in zip(train_cases, train_labels_cases):
            imgs = [os.path.join(case, f) for f in os.listdir(case) if f.endswith('.jpg')]
            train_images += imgs
            train_labels += [lbl] * len(imgs)

        val_images, val_labels = [], []
        for case, lbl in zip(val_cases, val_labels_cases):
            imgs = [os.path.join(case, f) for f in os.listdir(case) if f.endswith('.jpg')]
            val_images += imgs
            val_labels += [lbl] * len(imgs)

        train_ds = CustomImageDataset(train_images, train_labels, transform=transform_train)
        val_ds = CustomImageDataset(val_images, val_labels, transform=transform_val)

        counts = [train_labels.count(0), train_labels.count(1)]
        weights = [1.0 / c for c in counts]
        sample_w = [weights[lbl] for lbl in train_labels]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = get_efficientnet_b2(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=sched_patience, verbose=True)

        best_f1 = 0.0
        early_stop = 0
        best_model_wts = model.state_dict()

        for epoch in range(1, num_epochs+1):
            print(f"Epoch {epoch}/{num_epochs}")
            for phase in ['train','val']:
                is_train = (phase=='train')
                model.train() if is_train else model.eval()
                loader = train_loader if is_train else val_loader

                running_loss=0.0; running_corrects=0; all_preds=[]; all_labels=[]
                with tqdm(total=len(loader), desc=f"{phase.capitalize()} {epoch}") as pbar:
                    for X, y in loader:
                        X,y = X.to(device), y.to(device)
                        optimizer.zero_grad()
                        with torch.set_grad_enabled(is_train):
                            out = model(X)
                            loss = criterion(out, y)
                            if is_train:
                                loss.backward(); optimizer.step()
                        preds = torch.argmax(out,1)
                        running_loss += loss.item()*X.size(0)
                        running_corrects += (preds==y).sum()
                        all_preds += preds.cpu().tolist()
                        all_labels += y.cpu().tolist()
                        pbar.update(1)

                epoch_loss = running_loss/len(loader.dataset)
                epoch_acc  = running_corrects.double()/len(loader.dataset)
                epoch_f1   = f1_score(all_labels, all_preds, average='macro')
                epoch_prec = precision_score(all_labels, all_preds, average='macro')
                epoch_rec  = recall_score(all_labels, all_preds, average='macro')
                epoch_auc  = roc_auc_score(all_labels, all_preds)

                print(f"{phase} Loss:{epoch_loss:.4f} Acc:{epoch_acc:.4f} F1:{epoch_f1:.4f} AUC:{epoch_auc:.4f}")

                history[f"{phase}_loss"].append(epoch_loss)
                history[f"{phase}_accuracy"].append(epoch_acc.item())
                history[f"{phase}_precision"].append(epoch_prec)
                history[f"{phase}_recall"].append(epoch_rec)
                history[f"{phase}_f1"].append(epoch_f1)
                history[f"{phase}_auc"].append(epoch_auc)

                if phase=='val':
                    scheduler.step(epoch_f1)
                    if epoch_f1>best_f1:
                        best_f1=epoch_f1; best_model_wts=model.state_dict(); early_stop=0
                    else:
                        early_stop+=1
            if early_stop>=patience:
                print("Early stopping")
                break

        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), os.path.join(output_dir, f"efficientnet__best_model_fold{fold}.pth"))
        with open(os.path.join(output_dir, f"efficientnet_history_fold{fold}.pkl"),"wb") as f:
            pickle.dump(history,f)

    print("Training complete.")
