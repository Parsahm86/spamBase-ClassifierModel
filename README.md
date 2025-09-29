# 📧 Spam Detection Using Machine Learning 🚀

This project uses machine learning to **detect spam emails** from the **Spambase dataset**. It covers **data loading, exploration, feature selection, model comparison, training, and evaluation** with interpretable visualizations.

---

## 🗂 Dataset
- **Spambase** (UCI ML Repository)  
- 57 features extracted from email content (word frequencies, character frequencies, capital letters, etc.)  
- Target: `spam_class` (1 = spam, 0 = non-spam)

---

## 🔍 Project Workflow

### 1️⃣ Data Loading
- Load dataset using `pandas`.
- Extract feature names from `spambase.names`.

### 2️⃣ Data Exploration
- Check missing values and dataset info.
- Compute correlation of features with `spam_class`.
- Identify top correlated features for initial experiments.

### 3️⃣ 🎯 Feature Selection
- Create 3 feature sets:
  1. Top 5 features
  2. Top 10 features
  3. All 57 features
- Reduces noise, speeds up training, and improves interpretability.

### 4️⃣ Train-Test Split
- Split data: 70% train / 30% test using `train_test_split`.

### 5️⃣ Model Comparison 🤖
- Classifiers tested:
  - 🌳 **Decision Tree** (tune `max_depth`)
  - 🌲 **Random Forest**
  - 👥 **K-Nearest Neighbors (KNN)** (tune `n_neighbors`)
  - 📈 **Logistic Regression** (baseline)
- **Random Forest** gave the best accuracy and was selected for final analysis.

### 6️⃣ Random Forest Training
- Train the model on selected features.
- Predict on test set.

### 7️⃣ 📊 Model Evaluation
- **Confusion Matrix**: Visualize TP, TN, FP, FN.  
- **Feature Importance**: Identify which features most influence spam prediction.

---

## 🛠 Libraries Used
- `numpy`, `pandas` for data manipulation  
- `matplotlib`, `seaborn` for visualization  
- `scikit-learn` for ML models, metrics, and train-test splitting  

---

## ⚡ How to Run
1. Clone the repository.  
2. Place `spambase.data` and `spambase.names` in the project folder.  
3. Install dependencies:  
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
