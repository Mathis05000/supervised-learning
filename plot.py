import joblib
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

randomForest = joblib.load("RandomForest_BestModel_08219.joblib")
randomForestDropSex = joblib.load("./RandomForest_BestModel_no_sex.joblib")
adaBoost = joblib.load("AdaBoost_BestModel_08197.joblib")
gradientBoosting = joblib.load("GradientBoosting_BestModel_08273.joblib")
stackingClassifier = joblib.load("StackingClassifier_BestModel_08198.joblib")

features = pd.read_csv("/home/eynaud/Documents/5A/supervised/TP1/Resource/alt_acsincome_ca_features_85.csv")
labels = pd.read_csv("/home/eynaud/Documents/5A/supervised/TP1/Resource/alt_acsincome_ca_labels_85.csv") 

features_no_sex = features.drop(columns=["SEX"])
data = pd.concat([features, labels], axis=1)


predictions = {
    "RandomForestDropSex": randomForestDropSex.predict(features_no_sex),
}

def compute_metrics_per_gender(labels, predictions, gender_column, gender_value):
    gender_labels = labels[gender_column == gender_value].values.ravel()
    gender_preds = predictions[gender_column == gender_value]
    
    tn, fp, fn, tp = confusion_matrix(gender_labels, gender_preds).ravel()
    
    positive_rate = (tp + fp) / len(gender_labels)
    
    true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    false_positive_rate = tn / (fp + tn) if (fp + tn) > 0 else 0
    
    return positive_rate, true_positive_rate, false_positive_rate

results = {}

for model_name, preds in predictions.items():
    print(f"\nRésultats pour {model_name}:")
    # Hommes
    male_metrics = compute_metrics_per_gender(labels, preds, data["SEX"], gender_value=1)
    print(f"Hommes - Taux de prédictions positives: {male_metrics[0]:.2f}, "
          f"Taux de vrais positifs: {male_metrics[1]:.2f}, "
          f"Taux de faux positifs: {male_metrics[2]:.2f}")
    
    # Femmes
    female_metrics = compute_metrics_per_gender(labels, preds, data["SEX"], gender_value=2)
    print(f"Femmes - Taux de prédictions positives: {female_metrics[0]:.2f}, "
          f"Taux de vrais positifs: {female_metrics[1]:.2f}, "
          f"Taux de faux positifs: {female_metrics[2]:.2f}")
    
    results[model_name] = {"Hommes": male_metrics, "Femmes": female_metrics}

for metric_index, metric_name in enumerate(["Statistical Parity", "Equal Opportunity", "Predictive Equality"]):
    plt.figure(figsize=(8, 6))
    for model_name, metrics in results.items():
        plt.bar(
            [f"{model_name} - Hommes", f"{model_name} - Femmes"],
            [metrics["Hommes"][metric_index], metrics["Femmes"][metric_index]],
            label=model_name
        )
    plt.title(f"Comparaison {metric_name} par genre")
    plt.ylabel(metric_name)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



predictions = {
    "RandomForest": randomForest.predict(features),
    "AdaBoost": adaBoost.predict(features),
    "GradientBoosting": gradientBoosting.predict(features),
    "StackingClassifier": stackingClassifier.predict(features),
}

model_colors = {
    "Initial": "steelblue",
    "RandomForest": "blue",
    "AdaBoost": "mediumpurple",
    "GradientBoosting": "indigo",
    "StackingClassifier": "darkviolet"
}

feature_columns = features.columns

initial_correlations = []
for feature in feature_columns:
    corr, _ = pearsonr(features[feature], labels.values.ravel())  
    initial_correlations.append(corr)

for model, preds in {"Initial": labels.values.ravel(), **predictions}.items():
    correlations = []
    for feature in feature_columns:
        corr, _ = pearsonr(features[feature], preds)
        correlations.append(corr)
    
    plt.figure(figsize=(8, 6))
    plt.bar(feature_columns, correlations, color=model_colors[model])
    plt.title(f"Corrélation entre features et {'labels' if model == 'Initial' else f'prédictions du modèle {model}'}")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Coefficient de corrélation")
    plt.xlabel("Features")
    plt.tight_layout()
    
    plt.savefig(f"./heatmaps/{model}_correlation_plot.png")
    plt.close()


def compute_feature_importance(model, model_name, X, y, color):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring='accuracy')
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': result.importances_mean
    }).sort_values(by='Importance', ascending=False)
    

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color=color)
    plt.gca().invert_yaxis()
    plt.title(f"Feature Importance ({model_name})")
    plt.xlabel("Mean Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(f"./heatmaps/feature_importance_{model_name}.png")
    plt.close()
    print(f"Feature importance for {model_name} saved.")

compute_feature_importance(randomForest, "RandomForest", features, labels, model_colors["RandomForest"])
compute_feature_importance(adaBoost, "AdaBoost", features, labels, model_colors["AdaBoost"])
compute_feature_importance(gradientBoosting, "GradientBoosting", features, labels, model_colors["GradientBoosting"])
compute_feature_importance(stackingClassifier, "StackingClassifier", features, labels, model_colors["StackingClassifier"])

print("Feature importance calculated and saved for all models with consistent colors.")

