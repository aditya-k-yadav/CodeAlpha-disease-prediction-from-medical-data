# DiseasePrediction

** Disease Prediction from Medical Data**
Machine Learning Internship

## 📌 Objective
Predict whether a patient's tumor is **malignant** or **benign** based on diagnostic measurements taken from biopsy images (the same kind of structured medical data task as heart disease / diabetes prediction).

## 🧠 Approach
- Uses the **Breast Cancer Wisconsin (Diagnostic) Dataset** — built directly into scikit-learn (`load_breast_cancer`), so there's nothing to manually download.
- 569 patient records, 30 numeric diagnostic features (radius, texture, perimeter, area, smoothness, etc.).
- Trains and compares 3 classification models:
  - Logistic Regression
  - SVM (RBF kernel)
  - Random Forest
- Evaluates each model with **Accuracy, Precision, Recall, F1-Score, ROC-AUC, and 5-fold cross-validation**.
- Produces visualizations: model comparison chart, confusion matrix, top feature importances.

> Want to use a different disease dataset instead (e.g. UCI Heart Disease or Diabetes)? Replace `load_data()` in `disease_prediction.py` with `pd.read_csv("your_file.csv")` — the rest of the pipeline works unchanged.

## 📂 Project Structure
```
DiseasePrediction/
│
├── disease_prediction.py           # Main script
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── medical_data.csv                 # Dataset export (created after running)
├── model_comparison_results.csv     # Metrics table (created after running)
├── model_comparison.png             # Bar chart of all metrics (created after running)
├── confusion_matrix.png             # Confusion matrix for best model (created after running)
└── feature_importance.png           # Top 15 feature importances (created after running)
```

---

## 🛠 Setup Instructions (Anaconda + VS Code)

### 1. Install prerequisites
- [Anaconda](https://www.anaconda.com/download)
- [VS Code](https://code.visualstudio.com/)
- VS Code extensions: **Python** (by Microsoft) and **Jupyter** (optional)

### 2. Create and activate a conda environment
*(Skip this step if you already created `codealpha-ml`  — just reuse it.)*
```bash
conda create -n codealpha-ml python=3.10 -y
conda activate codealpha-ml
```

### 3. Get the project files
Copy the `DiseasePrediction` folder anywhere on your computer.

### 4. Install dependencies
```bash
cd DiseasePrediction
pip install -r requirements.txt
```

### 5. Open the project in VS Code
```bash
code .
```
Press `Ctrl+Shift+P` → **"Python: Select Interpreter"** → choose the `codealpha-ml` conda environment.

### 6. Run the script
```bash
python disease_prediction.py
```

All metrics print to the terminal; CSV/PNG outputs appear in the project folder.

---


---

## 📊 Sample Results (Breast Cancer Wisconsin dataset)

| Model                | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------------------------|----------|-----------|--------|----------|---------|
| Logistic Regression    | ~0.98    | ~0.99     | ~0.99  | ~0.99    | ~0.995  |
| SVM (RBF kernel)        | ~0.98    | ~0.99     | ~0.99  | ~0.99    | ~0.995  |
| Random Forest            | ~0.95    | ~0.96     | ~0.96  | ~0.96    | ~0.994  |

(Exact numbers may vary slightly depending on library versions.)


## 📜 License

MIT

---

## ⭐ Support

If you found this project useful:

* Give it a ⭐ on GitHub
