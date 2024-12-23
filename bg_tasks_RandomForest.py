import csv
import numpy as np
import scipy
import pandas as pd
import scipy.stats
import sklearn
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

features = pd.read_csv("/home/eynaud/Documents/5A/supervised/TP1/Resource/alt_acsincome_ca_features_85.csv")
labels =  pd.read_csv("/home/eynaud/Documents/5A/supervised/TP1/Resource/alt_acsincome_ca_labels_85.csv")

X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.25, random_state=42)

my_scaler = StandardScaler()
standars_feature = ['AGEP', 'WKHP']
X_train[standars_feature] = my_scaler.fit_transform(X_train[standars_feature])
X_val[standars_feature] = my_scaler.transform(X_val[standars_feature])

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay

rf = RandomForestClassifier()

grid_param_RandomForest = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8]
}
grid_search_RandomForest = GridSearchCV(estimator=rf, param_grid=grid_param_RandomForest, cv=5, verbose=2, n_jobs=-1)
grid_search_RandomForest.fit(X_train, y_train)

print("best params : ", grid_search_RandomForest.best_params_)
print("best precision : ", grid_search_RandomForest.best_score_)

test_score = grid_search_RandomForest.score(X_val, y_val)

print("Précision sur l'ensemble de test : ", test_score)

joblib.dump(grid_search_RandomForest.best_estimator_, 'RandomForest_BestModel.joblib')