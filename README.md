# Deep CORAL for Fabric Defect Detection

A PyTorch implementation of **Deep CORAL: Correlation Alignment for Deep Domain Adaptation** ([ECCV 2016, B. Sun & K. Saenko](https://arxiv.org/abs/1607.01719)), applied to enhance fabric defect detection in heterogeneous manufacturing environments.

> "Deep CORAL can learn a nonlinear transformation that aligns correlations of layer activations in deep neural networks (Deep CORAL)."

---

## 🧠 Project Overview

This project adapts the Deep CORAL method for **unsupervised domain adaptation** in the textile industry. Specifically, it improves **fabric defect detection** by bridging the domain gap between **grayscale** and **color** image datasets.

🌐 中文介紹: [ssarcandy.tw](https://ssarcandy.tw/2017/10/31/deep-coral/)

---

## 📊 Dataset Description

The dataset consists of four defect classes:

- `hole`
- `knot`
- `line`
- `stain`

Each class exists in:
- **Source domain**: Grayscale images
- **Target domain**: Color images

### Dataset Split (7:2:1 ratio)
| Set        | Hole | Knot | Line | Stain | Total |
|------------|------|------|------|--------|--------|
| Train      | 378  | 440  | 272  | 198    | 1288   |
| Validation |  94  | 110  |  68  |  50    | 322    |
| Test       |   3  |   3  |   3  |   3    | 12     |
| **Total**  |      |      |      |        | 1622   |

---

## 🛠️ Methodology

### 1. Preprocessing

- **Image size**: 227x227  
- **Normalization**:
  - Color: `mean=[0.4411, 0.4729, 0.5579]`, `std=[0.1818, 0.1699, 0.1836]`
  - Grayscale: `mean=[0.4790]*3`, `std=[0.1645]*3`

### 2. Data Augmentation

- Random crop
- Flip
- Color jitter (for target domain)

### 3. Architecture

- **Backbone**: AlexNet
- **Adaptation**: Deep CORAL (aligns 2nd-order statistics of features between domains)

### 4. Training

- **Loss**: Classification loss + λ * CORAL loss
- **Optimizer**: SGD with momentum
- **Epochs**: 10
- **Learning Rate**: `1e-4`, 10× higher for final FC layers

---

## 📈 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score (per class)
- Confusion Matrix

---

## 🚀 Results

| Domain  | Class | Precision | Recall | F1-Score | Accuracy |
|---------|-------|-----------|--------|----------|----------|
| Source  | All   | > 0.97    | > 0.93 | > 0.96   | 97.5%    |
| Target  | All   | ~ 0.71–0.95 | ~ 0.62–0.98 | ~ 0.66–0.97 | 86.6%    |
| Source-only Baseline | — | — | — | — | **73.4%** |

> 🔍 Deep CORAL significantly improves target domain accuracy **without labeled target data**.

---

## 📦 Repository Structure

