# Benchmarking Machine Learning Approaches for Polarization Mapping in Ferroelectrics Using 4D-STEM

This repository contains the implementation and benchmarking suite for automating polarization direction detection in ferroelectric (KNN) materials using 4D-STEM diffraction patterns.

## 🔬 Overview

Four-dimensional scanning transmission electron microscopy (4D-STEM) provides atomic-scale insights, but extracting polarization directions is computationally challenging. This project benchmarks several machine learning approaches to bridge the "domain gap" between synthetic training data and experimental reality.

**Key Features:**

* **Models:** ResNet, VGG, Custom CNN, and PCA-informed k-NN.
* **Training Paradigms:** Standard Classification, Regression, and Prototype Representation Learning.

---

## 🛠 Installation & Setup

* **Prerequisites**: Python 3.11+

1. **Clone the Repository:**
```bash
git clone https://github.com/matejMartinc/4D-STEM-polarization-mapping
cd 4D-STEM-polarization-mapping
pip install -r requirements.txt
```


2. **Prepare Data:**
* Create a `data` folder in the root directory.
* Download the dataset from [KT-Cloud](https://kt-cloud.ijs.si/index.php/s/yKqWe26tAF5tgC5).
* Unzip the contents into the `data` folder.



### Directory Structure

Your `data/` directory should look like this:

```text
data/
├── KNN_New_Simulations/         # Training datasets (RM, RU, MU, LU, etc.)
├── KNN_Experimental/            # Experimental test sets (1E-RM to 5E-RM/RD)
├── LU_4DSTEM/                   # Synthetic test sets (1S-LU, 2S-LU)
└── ... (other synthetic models)

```

> **Note:** For a full mapping of directory names to the Paper IDs (e.g., `KNNsc_40MX_0V_a` ➔ `1E-RM`), please refer to the [Dataset Mapping Table](https://www.google.com/search?q=%23dataset-mapping-table) below.

---

## 🚀 Running the Models

### 1. Neural Classification (CNNs)

Train and evaluate the Deep Learning models (ResNet, VGG, Custom CNN):

```bash
python neural_classification.py

```

* **Filtering:** To disable data filtering, uncomment `training_files = None` in the script.
* **Augmentation:** To disable, comment out lines 173-176 in `preprocessing.py`.

### 2. PCA-informed k-NN

Train and evaluate the PCA-based baseline:

```bash
python pca_classification.py

```

* **Augmentation:** To disable, comment out lines 120-123 in `pca_classification.py`.

---

## 📊 Visualization & Evaluation

| Script | Description |
| --- | --- |
| `make_latex_table.py` | Generates the LaTeX results table used in the paper. |
| `visualize_df_labels_in_a_structure.py` | Visualizes polarization classes as a color-coded map for a 128x128 scan area. |
| `visualize_results.py` | Generates prediction distribution plots across all classes. |
| `get_global_center_of_mass.py` | Calculates the global Center of Mass (CoM) for diffraction patterns. |
| `magnitude_dp_filtering.py` | Identifies and filters patterns with below-average magnitude. |

---

## 📝 Dataset Mapping Table

| Directory Name | Paper ID | Description |
| --- | --- | --- |
| `KNN_New_Simulations/KNN_Right_Mid_tiff` | **RM** | Synthetic Training Set |
| `KNNsc_40MX_0V_a` | **1E-RM** | Experimental Test Set |
| `LU_4DSTEM_20nm_NoTDS` | **1S-LU-6x6-20nm** | Synthetic Test Set |
| `Simulated_anomaly` | **Sim Anomaly** | Defect Detection Set |

---

## 🎓 Citation

If you use this code or dataset in your research, please cite:
*TODO*

---





 