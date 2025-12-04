# 🧠 Crowd Trajectory for Safe Evacuation Using DL – Deep Learning Project

This repository contains the code, data preprocessing pipeline, trained models for our Deep Learning course project on **human motion trajectory prediction** using several neural network architectures. The project compares MLP, CNN, GRU, Bi-GRU, and Bi-GRU with emotional features for multi-step future \((x, y)\) trajectory prediction.

---

## 📁 Repository Structure
```bash
├── preprocessing/
│   └── Preprocessing_5featuresXYEmotionVal_100runs.ipynb
│   └── evacuation_data.zip
│   └── evacuation_data(1).zip
│
├── notebook/
    ├── MLP_notebook.ipynb
    ├── CNN_notebook.ipynb
    ├── GRU_notebook.ipynb
    ├── BiGRU_notebook.ipynb
    ├── BiGRU_3features_notebook.ipynb
    └── Linear_baseline_notebook.ipynb
    └── evac_P8F12_P5features_100run.npz
```

---

## 📦 Data Preprocessing 

The raw dataset was originally provided as many CSV files inside two ZIP archives stored on Google Drive.  
Each ZIP archive includes 50 unique CSV files that include one whole evacuation scenario results.
During preprocessing, both ZIP files were extracted, and all CSV files were loaded and merged into a single dataset.

From the raw \((x, y)\) coordinates, we computed additional features \((v_x, v_y)\) and the `emotion_val` signal, producing **5 features per timestep**.  
Using an 8-step past and 12-step future sliding window, we generated input sequences **X** and targets **Y**, then saved them into a single compressed **NPZ file** to enable fast and consistent loading in all model notebooks.

Although the NPZ file contains **5 features**, different models used only the required subset:
- **2 features:** \((x, y)\) for MLP, CNN, GRU, and Bi-GRU
- **3 features:** \((x, y, emotion\_val)\) for the Bi-GRU with emotional input

This consolidated NPZ format ensures efficient and standardized data handling for all experiments.

---

## 🧪 Models Implemented

All models use **8 past frames** as input to predict **12 future frames**.

### Baseline
- **Linear Regression** — simple reference linear baseline using only \((x, y)\) of the starting point and endpoint of the trajectory

### Deep Learning Models

| Model | Input Features | Description |
|-------|----------------|-------------|
| **MLP** | x, y | Fully connected feedforward baseline |
| **CNN** | x, y | 2D convolution model |
| **GRU** | x, y | Recurrent sequence model |
| **Bi-GRU (XY-only)** | x, y | Bidirectional GRU with richer temporal representation |
| **Bi-GRU (3 features)** | x, y, emotion\_val | Adds emotional context as an additional modality |

Each notebook includes training code, evaluation metrics, and trajectory visualizations.

---

## 📊 Evaluation

Models were compared using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Average Displacement Error (ADE)
- Final Displacement Error (FDE)
- Predicted vs. ground-truth trajectory plots
- Prediction stability/smoothness
- Impact of adding the emotional signal

The Bi-GRU with emotional features showed the strongest performance.

---

