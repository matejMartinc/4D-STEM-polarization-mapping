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
├── KNN_New_Simulations/         # Training datasets (KNN_Right_Mid_tiff, KNN_Right_Up_tiff, etc.)
├── KNN_Experimental/            # Experimental test sets (KNNsc_40MX_0V_a, KNNsc_40MX_0V_b, etc.)
├── LU_4DSTEM/                   # Synthetic test sets (LU_4DSTEM_20nm_NoTDS, LU_4DSTEM_50nm_NoTD, etc.)
└── ... (other synthetic models)

```

> **Note:** For a full mapping of directory names to the Paper IDs (e.g., `KNNsc_40MX_0V_a` ➔ `1E-RM`), please refer to the Dataset Mapping Table below.

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

* **Filtering:** To disable data filtering, uncomment `training_files = None` in the script.
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

These tables provide the direct translation between the directory names found in the `data/` folder and the identifiers used throughout the paper.

#### 1. Synthetic Training & Base Simulation Sets

These datasets represent the 8 primary polarization directions used for model training and initial validation.

| Local Directory Path | Paper ID | Description / Direction |
| --- | --- | --- |
| `data/KNN_New_Simulations/KNN_Right_Mid_tiff` | **RM** | Right-Mid (0°) |
| `data/KNN_New_Simulations/KNN_Right_Up_tiff` | **RU** | Right-Up (45°) |
| `data/KNN_New_Simulations/KNN_Mid_Up_tiff` | **MU** | Mid-Up (90°) |
| `data/KNN_New_Simulations/KNN_Left_Up_tiff` | **LU** | Left-Up (135°) |
| `data/KNN_New_Simulations/KNN_Left_Mid_tiff` | **LM** | Left-Mid (180°) |
| `data/KNN_New_Simulations/KNN_Left_Down_tiff` | **LD** | Left-Down (225°) |
| `data/KNN_New_Simulations/KNN_Mid_Down_tiff` | **MD** | Mid-Down (270°) |
| `data/KNN_New_Simulations/KNN_Right_Down_tiff` | **RD** | Right-Down (315°) |

#### 2. Synthetic Test Sets (Varying Geometry & Thickness)

Used to evaluate the robustness of models against changes in specimen thickness and supercell size.

| Local Directory Path | Paper ID | Specifications |
| --- | --- | --- |
| `data/LU_4DSTEM/LU_4DSTEM_20nm_NoTDS` | **1S-LU-6x6-20nm** | 20nm thick, 6x6 unit cells |
| `data/LU_4DSTEM/LU_4DSTEM_50nm_NoTD` | **2S-LU-6x6-50nm** | 50nm thick, 6x6 unit cells |
| `data/KNN_Right_Mid_20nm_24mrad16383` | **3S-RM-2x2-20nm** | 20nm thick, 2x2 unit cells |
| `data/RD_KNN_Simpl_50nm_24mrad_NoTDS16383` | **4S-RD-6x6-50nm** | 50nm thick, 6x6 unit cells |
| `data/RU_KNN_Simpl_20nm_24mrad_NoTDS16383` | **5S-RU-6x6-20nm** | 20nm thick, 6x6 unit cells |
| `data/LD_KNN_Simpl_20nm_24mrad_NoTDS` | **6S-LD-4x4-20nm** | 20nm thick, 4x4 unit cells |
| `data/KNN_Simpl_Right_Down_20nm` | **7S-RD-2x2-20nm** | 20nm thick, 2x2 unit cells |
| `data/KNN_Experimental/Simulated_anomaly` | **Sim Anomaly** | Structural Defect Simulation |

#### 3. Experimental Data & Anomaly Detection

Real-world 4D-STEM data.

| Local Directory Path | Paper ID | Notes |
| --- | --- | --- |
| `data/KNN_Experimental/KNNsc_40MX_0V_a` | **1E-RM** | Experimental (Right-Mid) |
| `data/KNN_Experimental/KNNsc_40MX_0V_b` | **2E-RM** | Experimental (Right-Mid) |
| `data/KNN_Experimental/KNNsc_40MX_0V_c` | **3E-RM** | Experimental (Right-Mid) |
| `data/KNN_Experimental/KNN_Janina_a` | **4E-RM/RD** | Experimental (Mixed RM/RD) |
| `data/KNN_Experimental/KNNsc_tripod_tiff` | **5E-RM/RD** | Experimental (Mixed RM/RD) |


## 🎓 Citation

If you use this code or dataset in your research, please cite:
*TODO*

---





 