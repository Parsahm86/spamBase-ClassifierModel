# add package 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rcParams

from scipy.stats import randint, uniform

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,  ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# ----------------------------

# normal font for matplotlib
rcParams['font.family'] = 'DejaVu Sans'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42


# ----------------------------> load data

# print(df)
# print(df.info())
# print(df.isnull().sum())


with open("spambase.names", "r") as namesFile:
    namesInfo = namesFile.readlines()

# for index, line in enumerate(namesInfo):
#     print(index, "\t", line)


featuresList = namesInfo[33:]
featureNames = [name[: name.index(":")] for name in featuresList]
featureNames.append("spam_class")


data = pd.read_csv("spambase.data", names=featureNames)

# ----------------------------> correlation

correlation = data.corr()

spam_corr = correlation["spam_class"].abs()
# spam_corr = sorted(spam_corr)
spam_corr = spam_corr.sort_values(ascending=False)[1:]

# ---------------------------> creat X, y

features_5 = list(spam_corr[:5].keys())
features_10 = list(spam_corr[:10].keys())
X = data[features_5]
X2 = data[features_10]
X3 = data.iloc[:,:-1]
y = data["spam_class"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X3, y, test_size=.3, random_state=42)

# --------------------------------> Comparison cls models

# accuracies5 = {}

# def best_model_accuracy(model_class, param_name=None, param_range=None,\
#                         X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
    
#     best_param = None
#     best_acc = -1
    
#     if param_name and param_range:
#         for val in param_range:
#             model = model_class(**{param_name: val})
#             model.fit(X_train, y_train)
#             preds = model.predict(X_test)
#             acc = accuracy_score(y_test, preds)
#             if acc > best_acc:
#                 best_acc = acc
#                 best_param = val
#         return best_param, best_acc
#     else:
#         model = model_class()
#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)
#         acc = accuracy_score(y_test, preds)
#         return None, acc

# # 1️⃣ Decision Tree (جستجوی max_depth)
# best_depth, best_acc = best_model_accuracy(DecisionTreeClassifier, "max_depth", range(1,50))
# accuracies5["Decision Tree"] = best_acc
# print(f"Decision Tree: best depth = {best_depth}, accuracy = {best_acc}")

# # 2️⃣ Random Forest
# _, acc_rf = best_model_accuracy(RandomForestClassifier)
# accuracies5["Random Forest"] = acc_rf
# print(f"Random Forest: accuracy = {acc_rf}")

# # 3️⃣ K-Neighbors (جستجوی n_neighbors)
# best_k, best_acc_knn = best_model_accuracy(KNeighborsClassifier, "n_neighbors", range(1,50))
# accuracies5["K-Neighbors"] = best_acc_knn
# print(f"K-Nearest Neighbors: best k = {best_k}, accuracy = {best_acc_knn}")

# # 4️⃣ Logistic Regression
# _, acc_lr = best_model_accuracy(LogisticRegression)
# accuracies5["Logistic Regression"] = acc_lr
# print(f"Logistic Regression: accuracy = {acc_lr}")

# # -------------------------------> plot

# plt.figure(figsize = (12, 5))
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
# plt.bar(accuracies5.keys(), accuracies5.values(), color=colors)
# plt.title(f"Classification Accuracy of Machine Learning Models Trained on {X2.shape[1]} Features")

# for index, key in enumerate(accuracies5):
#     plt.text(index, accuracies5[key] + 0.01, "{0:.4f}".format(accuracies5[key]), 
#              ha='center', fontsize=10)

# -------------------------------> run randomForest model

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# # پیش‌بینی روی تست
y_pred = rf_model.predict(X_test)

# -------------------------------> Feature Importance - Random Forest

importances = rf_model.feature_importances_
feat_importances = pd.Series(importances, index=X_train.columns)
feat_importances.sort_values(ascending=False).plot(kind='bar', figsize=(12,6))
plt.title("Feature Importance - Random Forest")
plt.show()

# -------------------------------> confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# # نمایش گرافیکی
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.show()










