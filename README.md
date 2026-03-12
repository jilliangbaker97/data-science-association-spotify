# 🎵 Can Audio Features Alone Predict Commercial Success?

### A Data Science Case Study in Music Analytics

[![Open In Colab](https://drive.google.com/file/d/1bA0MqDryif_2KnSvUecU0k10Xt1-O_Bd/view?usp=sharing)]

> **The Question:** Spotify encodes every track with an acoustic fingerprint — danceability, energy, valence, tempo, and more. But does any of that actually predict whether a song becomes a _hit_?

In this notebook, we build a complete machine learning pipeline to find out — using nothing but audio features.

---

## 📋 What's in this repo

```
spotify-hit-predictor/
├── spotify_hit_prediction_student.ipynb   ← The main notebook (start here)
├── data/
│   └── README.md                          ← How to get the dataset
├── requirements.txt                       ← For local Python installs
└── README.md                              ← You're reading it
```

---

## 🚀 Quickstart (Google Colab — recommended)

1. Click the **Open in Colab** badge above
2. Click **File → Save a copy in Drive** so your changes are saved
3. Follow the dataset instructions in the first notebook cell
4. Click **Runtime → Run all** and follow along

> ⚠️ **Don't skip the "Save a copy" step!** Without it you're editing a read-only version and your work won't persist.

---

## 📦 Dataset Setup

The notebook requires `spotify_cleaned_data.csv`. It is **not included in this repo** (file size). There are two ways to load it:

### Option A — Upload manually (simplest for one-off use)

Run this cell at the top of the notebook:

```python
from google.colab import files
uploaded = files.upload()  # select spotify_cleaned_data.csv from your computer
```

### Option B — Load from shared Google Drive (recommended for groups)

1. Upload `spotify_cleaned_data.csv` to a shared Google Drive folder
2. Run this at the top of the notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
# Then update the read_csv line to:
df = pd.read_csv('/content/drive/MyDrive/YOUR_FOLDER/spotify_cleaned_data.csv')
```

---

## 🗺️ Notebook Structure

| Section                   | What you'll do                                                                     |
| ------------------------- | ---------------------------------------------------------------------------------- |
| **0. Setup**              | Load libraries                                                                     |
| **1. Load & Explore**     | Understand the dataset's shape, types, and missing values                          |
| **2. Define "Hit"**       | Engineer a binary target variable from stream counts                               |
| **3. EDA**                | Visualise stream distributions, feature correlations, and hit vs. non-hit profiles |
| **4. Preprocessing**      | Select features, split data, scale for Logistic Regression                         |
| **5. Model Training**     | Train Logistic Regression and Random Forest classifiers                            |
| **6. Evaluation**         | Accuracy, confusion matrices, and ROC-AUC curves                                   |
| **7. Feature Importance** | Which audio features drive hit prediction?                                         |
| **8. Conclusions**        | Answer the question — and discuss what comes next                                  |
| **🚀 Your Turn**          | Extension challenges across beginner / intermediate / advanced levels              |

---

## 🛠️ Running Locally (optional)

If you prefer to run the notebook in your own environment:

```bash
git clone https://github.com/YOUR-USERNAME/spotify-hit-predictor.git
cd spotify-hit-predictor
pip install -r requirements.txt
jupyter notebook spotify_hit_prediction_student.ipynb
```

Place `spotify_cleaned_data.csv` in the root folder before running.

---

## 📚 Key Concepts Covered

- Binary classification pipeline end-to-end
- Target variable engineering from continuous data
- Exploratory data analysis (distributions, correlation heatmaps)
- Data leakage — what it is and how to avoid it
- Feature scaling with `StandardScaler`
- Logistic Regression vs. Random Forest — trade-offs
- Evaluating imbalanced classifiers (why accuracy alone is misleading)
- ROC-AUC interpretation
- Random Forest feature importance

---

## 💬 Discussion Prompts

The notebook includes built-in discussion prompts throughout. Key questions to explore with your group:

1. Is stream count the right definition of "commercial success"?
2. Why does 80% accuracy not actually mean our model is good?
3. Which type of error matters more — missing a hit, or over-promoting a flop?
4. Does the top-ranked feature surprise you?
5. Could a hit-prediction model disadvantage certain genres or independent artists?

---

_Built for a group data science session · 20,592 Spotify tracks · scikit-learn pipeline_
