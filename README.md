# Music Genre Classification: Visual vs. Mathematical Approaches

### **Project Overview**

This project explores two distinct methodologies for automated music genre classification using the **GTZAN** dataset. We aimed to determine whether modern **Deep Learning on Spectrograms (Visual)** outperforms traditional **Feature Extraction (Mathematical)** techniques.

  * **Objective:** Classify 30-second audio clips into 10 genres (Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock).
  * **Key Findings:** Mathematical feature extraction (MLP/XGBoost) proved more robust (\~72% accuracy) than visual deep learning (\~65%) for this specific dataset size, primarily due to data scarcity and domain gaps in transfer learning.

### **The Two Approaches**

#### **Path A: Visual (Deep Learning)**

We treated audio analysis as an Image Classification problem by converting audio clips into **Mel-Spectrograms**.

  * **Models Tested:** Custom CNN (Baseline), VGG16 (Transfer Learning), ResNet50, EfficientNetB0.
  * **Best Result:** **VGG16** achieved **65%** accuracy, successfully transferring knowledge from ImageNet to audio textures.

#### **Path B: Mathematical (Feature Extraction)**

We treated audio analysis as a Tabular Classification problem by extracting 58 statistical features using `Librosa`.

  * **Features:** MFCCs, Spectral Centroid, Zero Crossing Rate, Tempo, Rolloff.
  * **Models Tested:** KNN, XGBoost, Multi-Layer Perceptron (MLP).
  * **Best Result:** **MLP** achieved **71.9%** accuracy, proving to be the most effective model.

### **Repository Structure**

```text
├── final_project.ipynb       # Main Jupyter Notebook (Training & Demo)
├── features_3_sec.csv        # Tabular dataset (Audio Features)
├── images_original/          # Folder containing Mel-Spectrogram images (Required for Part 1)
├── genres_original/          # Folder containing raw audio files (Required for Demo)
├── demo_songs/               # Folder containing random raw audio files to test
└── README.md                 # Project Documentation
```

### **Installation & Requirements**

To run this project, install the following dependencies:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn librosa xgboost
```

### **How to Run**

1.  **Dataset Setup:**

      * Download the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).
      * Place the `images_original` folder and `features_3_sec.csv` in the root directory.

2.  **Training:**

      * Open `final_project.ipynb`.
      * Run all cells to train the models. The notebook is divided into **Part 1 (Visual)** and **Part 2 (Mathematical)**.

### **Final Results Leaderboard**

| Rank | Model | Type | Test Accuracy |
| :--- | :--- | :--- | :--- |
| **1** | **MLP (Neural Net)** | Mathematical | **71.57%** |
| 2 | XGBoost | Mathematical | 69.60% |
| 3 | KNN | Mathematical | 67.37% |
| 4 | VGG16 | Visual (Transfer) | 65.35% |
| 5 | Custom CNN | Visual (Baseline) | 11.88% |

### **Authors**

  * Ethan
  * Lucas

-----

*COE379 Final Project - Fall 2025*