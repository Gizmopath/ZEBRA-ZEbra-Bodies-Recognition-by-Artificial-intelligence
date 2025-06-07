# 🦓 ZEBRA: AI for Early Detection of Fabry Nephropathy

**ZEBRA** is a deep learning pipeline designed to support pathologists in the **early detection of Fabry nephropathy** — a rare, often silent genetic disease that can severely damage kidneys if left untreated.

This repository provides **classification and segmentation tools** trained on high-resolution kidney biopsy images to detect vacuolated (foamy) podocytes — the microscopic hallmark of Fabry disease.

---

## 🧠 Why ZEBRA?

Fabry disease often presents subtle histological signs, especially in women or in late-onset forms. This project proposes a computational pipeline to:

- **Automatically detect vacuolated podocytes**
- **Quantify disease burden with the ZEBRA score**
- **Assist diagnosis where specialized expertise may be lacking**

> 🧬 Digital pathology is the ideal solution to scale Fabry diagnostics using central expertise + AI-based automation.

---

## 🗃️ Dataset Overview

| Split       | Center      | Cases/Slides      | Stains       | Scanners      | Foamy Podocytes | Count |
|-------------|-------------|-------------------|--------------|---------------|------------------|--------|
| Train/Val   | DIPLOMAT    | FN(8/15), Ctrl(14/29) | H&E, PAS     | MIDI II, S60   | Present/Absent   | 479 |
|             | Bologna     | FN(16/51), Ctrl(18/52) | H&E, PAS     | MIDI II        | Present/Absent   | 679 |
|             | Naples      | FN(6/6)              | H&E, PAS     | MIDI II, S60   | Present          | 119 |
| Test        | Florence    | FN(7/13), Ctrl(8/16)  | H&E, PAS     | KF-PRO, S60    | Present/Absent   | 487 |

---

## 🧪 Model Performance

### 📊 Classification Results

| Model       | Aug | Val F1 | Test F1 |
|-------------|-----|--------|---------|
| ResNet      | ✗   | 73     | 47      |
| ResNet      | ✓   | 78     | 73      |
| EfficientNet| ✗   | 74     | 43      |
| EfficientNet| ✓   | 76     | 79      |
| DenseNet    | ✗   | 75     | 36      |
| DenseNet    | ✓   | 77     | 70      |
| Swin-T      | ✗   | 80     | 55      |
| Swin-T      | ✓   | 81     | 78      |
| UNI2        | ✗   | 80     | 65      |
| UNI2        | ✓   | 79     | 67      |

### 🧩 Segmentation Results (Dice / IoU)

| Model        | Aug | Val Dice | Test Dice |
|--------------|-----|----------|-----------|
| UNet         | ✗   | 18       | 16        |
| DeepLab      | ✗   | 33       | 24        |
| SegFormer    | ✗   | 52       | 46        |
| SegFormer    | ✓   | 38       | 25        |
| Glomeruli (SF)| ✗  | 95       | 94        |

---

## 🛠️ How to Use

1. **Annotate glomeruli in QuPath**  
   Use the `QuPath_annotation.groovy` script to place square annotations on glomeruli (number of squares can be adjusted).

2. **Extract glomerular tiles**  
   Use `QuPath_extraction.groovy` to export the ROIs for downstream processing.

3. **Run inference**  
   - Classification: use scripts in `inference_classification/`
   - Segmentation: use scripts in `inference_segmentation/`
   - Compare models (optional): see `compare_segformer_models.py`

4. **Calculate ZEBRA score**  
   Use the `ZEBRA_score/` folder to compute the percentage of vacuolated podocytes per glomerulus.

---

## 🧪 Outputs

- ✅ Pretrained models included
- 📈 Pickle logs of training histories
- 📊 Visual confusion matrices and metric summaries

---

## 🖼️ Visual Examples

### 🔍 Classification Example  
![Classification Preview](<INSERT_CLASSIFICATION_IMAGE_URL>)

### 🧠 Segmentation Example  
![Segmentation Preview](<INSERT_SEGMENTATION_IMAGE_URL>)

---

## 📦 Install Requirements

```bash
pip install -r requirements.txt
```

---

## 🧑‍💻 Authors & Contributors

Project developed as part of collaborative research on AI for rare kidney diseases. Contact for more info.

---

**🦓 ZEBRA** — making early Fabry detection smarter, faster, and more accessible.
