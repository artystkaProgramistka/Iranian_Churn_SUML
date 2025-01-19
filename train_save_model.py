import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
import joblib

# Preprocessing function from the script
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

# Apply undersampling to balance the dataset
rus = RandomUnderSampler(random_state=1337)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Set up cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)

# Random Forest configuration
rf_model = RandomForestClassifier(n_estimators=100, max_depth=13, min_samples_leaf=1, criterion='gini', random_state=1337)

# LightGBM configuration
lgbm_model = LGBMClassifier(n_estimators=30, num_leaves=31, learning_rate=0.2, random_state=1337)

# Hyperparameter search space for GridSearchCV
param_grid_rf = {
    'n_estimators': [100],
    'max_depth': [13],
    'min_samples_leaf': [1],
    'criterion': ['gini']
}

param_grid_lgbm = {
    'n_estimators': [30],
    'num_leaves': [31],
    'learning_rate': [0.2]
}

# Grid Search for Random Forest
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=cv, scoring='roc_auc')
rf_grid_search.fit(X_resampled, y_resampled)

# Grid Search for LightGBM
lgbm_grid_search = GridSearchCV(estimator=lgbm_model, param_grid=param_grid_lgbm, cv=cv, scoring='roc_auc')
lgbm_grid_search.fit(X_resampled, y_resampled)

# Best models
best_rf_model = rf_grid_search.best_estimator_
best_lgbm_model = lgbm_grid_search.best_estimator_

joblib.dump(best_rf_model, "best_rf_model.pkl")
joblib.dump(best_lgbm_model, "best_lgbm_model.pkl")

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

# Evaluate the Random Forest model
rf_evaluation = evaluate_model(best_rf_model, X, y)
print("Random Forest Evaluation:", rf_evaluation)

# Evaluate the LightGBM model
lgbm_evaluation = evaluate_model(best_lgbm_model, X, y)
print("LightGBM Evaluation:", lgbm_evaluation)
