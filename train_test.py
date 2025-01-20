import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from imblearn.under_sampling import RandomUnderSampler

# Preprocessing function
def read_and_preprocess_data():
    data = pd.read_csv("customer_churn.csv")
    data.drop(['Age'], axis=1, inplace=True)
    integer_columns = [column for column in data.columns if column not in ['Status', 'Complains', 'Churn']]
    data[integer_columns] = data[integer_columns].astype('int')
    data['Status'] = data['Status'].map({1: True, 2: False}).astype('bool')
    data['Complains'] = data['Complains'].map({1: True, 0: False}).astype('bool')
    data['Churn'] = data['Churn'].map({1: True, 0: False}).astype('bool')
    X = data.drop(['Churn'], axis=1)
    y = data['Churn']
    return data, X, y

# Load and preprocess data
data, X, y = read_and_preprocess_data()

# Apply pairwise linear feature generation
poly = PolynomialFeatures(interaction_only=True, include_bias=False)
X_pairwise = poly.fit_transform(X)

# Update feature names for interpretability
feature_names = poly.get_feature_names_out(input_features=X.columns)
X_pairwise = pd.DataFrame(X_pairwise, columns=feature_names)

# Apply undersampling to balance the dataset
rus = RandomUnderSampler(random_state=1337)
X_resampled, y_resampled = rus.fit_resample(X_pairwise, y)

# Set up cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)

# LightGBM configuration
lgbm_model = LGBMClassifier(boosting_type='gbdt', n_estimators=30, num_leaves=31, learning_rate=0.2, random_state=1337)

# Hyperparameter search space for GridSearchCV
param_grid_lgbm = {
    'n_estimators': [30],
    'num_leaves': [31],
    'learning_rate': [0.2]
}

# Grid Search for LightGBM
lgbm_grid_search = GridSearchCV(estimator=lgbm_model, param_grid=param_grid_lgbm, cv=cv, scoring='roc_auc')
lgbm_grid_search.fit(X_resampled, y_resampled)

# Best LightGBM model
best_lgbm_model = lgbm_grid_search.best_estimator_

# Model evaluation function
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    roc_auc = roc_auc_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)
    return {'ROC AUC': roc_auc, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'Confusion Matrix': conf_matrix}

# Evaluate the LightGBM model
lgbm_evaluation = evaluate_model(best_lgbm_model, X_pairwise, y)

# Print evaluation metrics
print("LightGBM Evaluation:", lgbm_evaluation)

# Visualization: Metrics Bar Chart
metrics = ['Dokładność (Accuracy)', 'Precyzja (Precision)', 'Czułość (Recall)', 'F1']
values = [
    lgbm_evaluation['Accuracy'],
    lgbm_evaluation['Precision'],
    lgbm_evaluation['Recall'],
    lgbm_evaluation['F1 Score']
]

colors = ['blue', 'green', 'orange', 'red']

plt.figure(figsize=(12, 8), dpi=600)  # Increased figure size and resolution
bar_plot = plt.bar(metrics, values, color=colors)
plt.ylim(0, 1.1)  # Metrics range from 0 to slightly above 1 for spacing
plt.title("Metryki Modelu (Model Metrics)")
plt.ylabel("Wartość (Value)")
plt.xlabel("Metryki (Metrics)")

# Add numeric values on top of the bars
for bar, value in zip(bar_plot, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{value:.2f}',
             ha='center', va='bottom', fontsize=10)

plt.savefig("metrics_bar_chart.png", dpi=600, bbox_inches='tight')  # High resolution and tight layout
plt.close()

# Visualization: Confusion Matrix
plt.figure(figsize=(10, 8), dpi=600)  # Increased figure size and resolution
sns.heatmap(lgbm_evaluation['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Macierz pomyłek (Confusion Matrix)")
plt.ylabel("Rzeczywista wartość (True Label)")
plt.xlabel("Przewidywana wartość (Predicted Label)")
plt.savefig("confusion_matrix.png", dpi=600, bbox_inches='tight')  # High resolution and tight layout
plt.close()

# Visualization: Feature Importance
feature_importance = best_lgbm_model.feature_importances_
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(12, 15), dpi=600)  # Increased vertical size for better readability
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
plt.title("Znaczenie cech (Feature Importance)")
plt.xlabel("Znaczenie (Importance)")
plt.ylabel("Cechy (Features)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=600, bbox_inches='tight')
plt.close()
