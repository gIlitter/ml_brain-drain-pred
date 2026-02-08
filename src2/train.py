import pandas as pd
import numpy as np
import os
import sys
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from statsmodels.duration.hazard_regression import PHReg
from util.commonUtil import get_data_source, format_time, sort_by_time, deduplicate, mape
from util.logUtil import get_logger

warnings.filterwarnings("ignore")
sns.set_style('whitegrid')

# 设置路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data2')
FIT_DIR = os.path.join(BASE_DIR, 'fit2')
LOG_DIR = os.path.join(BASE_DIR, 'log2')
MODEL_DIR = os.path.join(BASE_DIR, 'model2')

os.makedirs(FIT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 日志
logger = get_logger('train', log_dir=LOG_DIR)

logger.info(f"Training started at {format_time(datetime.now())}")

# ============================= 1. 加载数据 =============================
train_path = os.path.join(DATA_DIR, 'new_train.csv')
test_path = os.path.join(DATA_DIR, 'new_test2.csv')

train_df = get_data_source(train_path)
test_df = get_data_source(test_path)

logger.info(f"Train data loaded: {train_df.shape}, Test data loaded: {test_df.shape}")

# ============================= 1.1 原始数据可视化分析 =============================

# 数据基本信息
logger.info(f"Train data info: {train_df.info()}")
logger.info(f"Train data description: {train_df.describe()}")

# 类别特征分布
cat_cols_vis = ['Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'BusinessTravel', 'OverTime']

plt.figure(figsize=(15, 12))
for i, col in enumerate(cat_cols_vis, 1):
    plt.subplot(3, 3, i)
    sns.countplot(data=train_df, x=col, hue='Attrition', palette='viridis')
    plt.title(f'{col} Distribution by Attrition')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

cat_dist_path = os.path.join(FIT_DIR, 'categorical_features_distribution.png')
plt.savefig(cat_dist_path, dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"Categorical features distribution saved to {cat_dist_path}")

# 数值特征分布
num_cols_vis = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'DistanceFromHome', 'YearsSinceLastPromotion']

plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols_vis, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data=train_df, x=col, hue='Attrition', kde=True, palette='viridis')
    plt.title(f'{col} Distribution by Attrition')
    plt.tight_layout()

num_dist_path = os.path.join(FIT_DIR, 'numerical_features_distribution.png')
plt.savefig(num_dist_path, dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"Numerical features distribution saved to {num_dist_path}")

# 流失率分析
attrition_rate = train_df['Attrition'].mean()
logger.info(f"Overall attrition rate: {attrition_rate:.4f}")

# 流失率与各特征的关系
plt.figure(figsize=(15, 12))
for i, col in enumerate(cat_cols_vis, 1):
    plt.subplot(3, 3, i)
    attrition_by_col = train_df.groupby(col)['Attrition'].mean()
    attrition_by_col.plot(kind='bar', color='green')
    plt.title(f'Attrition Rate by {col}')
    plt.ylabel('Attrition Rate')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 0.5)
    plt.tight_layout()

attrition_rate_path = os.path.join(FIT_DIR, 'attrition_rate_by_features.png')
plt.savefig(attrition_rate_path, dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"Attrition rate by features saved to {attrition_rate_path}")

# 相关性分析
corr_cols = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance', 'Attrition']
corr_matrix = train_df[corr_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Matrix')
plt.tight_layout()

corr_path = os.path.join(FIT_DIR, 'correlation_matrix.png')
plt.savefig(corr_path, dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"Correlation matrix saved to {corr_path}")

y = train_df['Attrition'].astype(int)
X = train_df.drop('Attrition', axis=1)

if 'Attrition' in test_df.columns:
    y_test = test_df['Attrition'].astype(int)
    X_test = test_df.drop('Attrition', axis=1)
else:
    y_test = None
    X_test = test_df.copy()

drop_cols = ['EmployeeNumber', 'Over18', 'StandardHours']
X = X.drop(columns=drop_cols, errors='ignore')
X_test = X_test.drop(columns=drop_cols, errors='ignore')

# ============================= 2. 生存分析：Cox PH 模型 =============================
data_surv = train_df.copy()
data_surv['Event'] = y
data_surv['OverTime'] = data_surv['OverTime'].map({'Yes': 1, 'No': 0}).fillna(0)
data_surv['BusinessTravel'] = data_surv['BusinessTravel'].map({'Travel_Frequently': 2, 'Travel_Rarely': 1, 'Non-Travel': 0}).fillna(1)
data_surv['OT_Travel'] = data_surv['OverTime'] * data_surv['BusinessTravel']
data_surv['SatisfactionMean'] = data_surv[['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction']].mean(axis=1)

cox_features = ['Age', 'MonthlyIncome', 'OverTime', 'BusinessTravel', 'SatisfactionMean', 'OT_Travel']
X_cox = data_surv[cox_features]

surv_formula = 'YearsAtCompany ~ Age + MonthlyIncome + OverTime + BusinessTravel + SatisfactionMean + OT_Travel'
surv_model = PHReg.from_formula(surv_formula, status='Event', data=data_surv)
surv_result = surv_model.fit()

logger.info("\n" + "="*60 + "\nSURVIVAL ANALYSIS (Cox PH Model)\n" + "="*60)
logger.info(surv_result.summary())

risk_scores_train = surv_model.predict(params=surv_result.params, exog=X_cox).predicted_values
X['RiskScore'] = risk_scores_train

X_test_cox = X_test.copy()
X_test_cox['OverTime'] = X_test_cox['OverTime'].map({'Yes': 1, 'No': 0}).fillna(0)
X_test_cox['BusinessTravel'] = X_test_cox['BusinessTravel'].map({'Travel_Frequently': 2, 'Travel_Rarely': 1, 'Non-Travel': 0}).fillna(1)
X_test_cox['OT_Travel'] = X_test_cox['OverTime'] * X_test_cox['BusinessTravel']
X_test_cox['SatisfactionMean'] = X_test_cox[['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction']].mean(axis=1)
X_test_cox = X_test_cox[cox_features]

risk_scores_test = surv_model.predict(params=surv_result.params, exog=X_test_cox).predicted_values
X_test['RiskScore'] = risk_scores_test

logger.info(f"RiskScore added! Train mean: {risk_scores_train.mean():.4f}, Test mean: {risk_scores_test.mean():.4f}")

# ============================= 3. 终极特征工程 =============================
def ultimate_features(df):
    df = df.copy()
    for col in ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany']:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    df['IncomePerAge'] = df['MonthlyIncome'] / df['Age']
    df['IncomePerYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    df['TenureRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    df['LowPayHighOT'] = (df['MonthlyIncome'] < 3000) & (df['OverTime'] == 'Yes')
    df['RecentHire'] = df['YearsAtCompany'] < 1
    return df

X = ultimate_features(X)
X_test = ultimate_features(X_test)

# ============================= 4. 预处理 + 交互项 =============================
cat_cols = ['Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
num_cols = [c for c in X.columns if c not in cat_cols and X[c].dtype in ['int64', 'float64']]

interaction_cols = ['MonthlyIncome', 'TotalWorkingYears', 'Age', 'DistanceFromHome']
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features_train = poly.fit_transform(X[interaction_cols])
poly_features_test = poly.transform(X_test[interaction_cols])

X = X.drop(columns=interaction_cols)
X_test = X_test.drop(columns=interaction_cols)
num_cols = [c for c in num_cols if c not in interaction_cols]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
], sparse_threshold=0)

X_main = preprocessor.fit_transform(X)
X_test_main = preprocessor.transform(X_test)

X_processed = np.hstack([X_main, poly_features_train])
X_test_processed = np.hstack([X_test_main, poly_features_test])

logger.info(f"Final feature dim: {X_processed.shape[1]}")

# ============================= 5. SMOTE =============================
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_processed, y)
logger.info(f"After SMOTE: {X_bal.shape}, Positive ratio: {y_bal.mean():.3f}")

# ============================= 6. 模型训练 =============================
param_grid_lgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05]
}
lgb = LGBMClassifier(random_state=42, verbose=-1)
grid_lgb = GridSearchCV(lgb, param_grid_lgb, cv=3, scoring='f1', n_jobs=-1)
grid_lgb.fit(X_bal, y_bal)
best_lgb = grid_lgb.best_estimator_
logger.info(f"Best LGB Params: {grid_lgb.best_params_}")
logger.info(f"LGB Best CV Score: {grid_lgb.best_score_:.4f}")

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05]
}
xgb = XGBClassifier(random_state=42, eval_metric='logloss')
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=3, scoring='f1', n_jobs=-1)
grid_xgb.fit(X_bal, y_bal)
best_xgb = grid_xgb.best_estimator_
logger.info(f"Best XGB Params: {grid_xgb.best_params_}")
logger.info(f"XGB Best CV Score: {grid_xgb.best_score_:.4f}")

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

# 随机森林模型
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
rf.fit(X_bal, y_bal)

# ============================= 6.1 单个模型评估 =============================
logger.info("\n" + "="*60 + "\nINDIVIDUAL MODEL EVALUATION\n" + "="*60)

# LightGBM评估
y_pred_lgb = best_lgb.predict(X_test_processed)
y_prob_lgb = best_lgb.predict_proba(X_test_processed)[:, 1]
logger.info("\n--- LightGBM Model Evaluation ---")
logger.info(f"Accuracy: {accuracy_score(y_test, y_pred_lgb):.4f}")
logger.info(f"Precision: {precision_score(y_test, y_pred_lgb):.4f}")
logger.info(f"Recall: {recall_score(y_test, y_pred_lgb):.4f}")
logger.info(f"F1 Score: {f1_score(y_test, y_pred_lgb):.4f}")
logger.info(f"ROC AUC: {roc_auc_score(y_test, y_prob_lgb):.4f}")
logger.info("\nClassification Report:")
logger.info(classification_report(y_test, y_pred_lgb))

# XGBoost评估
y_pred_xgb = best_xgb.predict(X_test_processed)
y_prob_xgb = best_xgb.predict_proba(X_test_processed)[:, 1]
logger.info("\n--- XGBoost Model Evaluation ---")
logger.info(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
logger.info(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
logger.info(f"Recall: {recall_score(y_test, y_pred_xgb):.4f}")
logger.info(f"F1 Score: {f1_score(y_test, y_pred_xgb):.4f}")
logger.info(f"ROC AUC: {roc_auc_score(y_test, y_prob_xgb):.4f}")
logger.info("\nClassification Report:")
logger.info(classification_report(y_test, y_pred_xgb))

# Logistic Regression评估
lr.fit(X_bal, y_bal)
y_pred_lr = lr.predict(X_test_processed)
y_prob_lr = lr.predict_proba(X_test_processed)[:, 1]
logger.info("\n--- Logistic Regression Model Evaluation ---")
logger.info(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
logger.info(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
logger.info(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
logger.info(f"F1 Score: {f1_score(y_test, y_pred_lr):.4f}")
logger.info(f"ROC AUC: {roc_auc_score(y_test, y_prob_lr):.4f}")
logger.info("\nClassification Report:")
logger.info(classification_report(y_test, y_pred_lr))

# 随机森林评估
y_pred_rf = rf.predict(X_test_processed)
y_prob_rf = rf.predict_proba(X_test_processed)[:, 1]
logger.info("\n--- Random Forest Model Evaluation ---")
logger.info(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
logger.info(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
logger.info(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
logger.info(f"F1 Score: {f1_score(y_test, y_pred_rf):.4f}")
logger.info(f"ROC AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")
logger.info("\nClassification Report:")
logger.info(classification_report(y_test, y_pred_rf))

# ============================= 7. 集成模型 =============================
ensemble = VotingClassifier(
    estimators=[('lgb', best_lgb), ('xgb', best_xgb), ('lr', lr), ('rf', rf)],
    voting='soft'
)
ensemble.fit(X_bal, y_bal)

# ============================= 8. 阈值调优 =============================
X_tr, X_val, y_tr, y_val = train_test_split(X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42)
ensemble.fit(X_tr, y_tr)
y_prob = ensemble.predict_proba(X_val)[:, 1]

best_thr = 0.5
best_f1 = 0
for thr in np.arange(0.3, 0.7, 0.01):
    f1 = f1_score(y_val, (y_prob >= thr).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

logger.info(f"Best Threshold (F1): {best_thr:.3f} | Val F1: {best_f1:.4f}")

# ============================= 9. 测试预测 =============================
y_prob_test = ensemble.predict_proba(X_test_processed)[:, 1]
y_pred = (y_prob_test >= best_thr).astype(int)

# ============================= 9.1 集成模型详细评估 =============================
logger.info("\n" + "="*60 + "\nENSEMBLE MODEL EVALUATION\n" + "="*60)
logger.info(f"\n--- Ensemble Model Evaluation (Threshold: {best_thr:.3f}) ---")
logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
logger.info(f"Precision: {precision_score(y_test, y_pred):.4f}")
logger.info(f"Recall: {recall_score(y_test, y_pred):.4f}")
logger.info(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
logger.info(f"ROC AUC: {roc_auc_score(y_test, y_prob_test):.4f}")
logger.info("\nClassification Report:")
logger.info(classification_report(y_test, y_pred))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
logger.info("\nConfusion Matrix:")
logger.info(f"True Negatives: {cm[0,0]}")
logger.info(f"False Positives: {cm[0,1]}")
logger.info(f"False Negatives: {cm[1,0]}")
logger.info(f"True Positives: {cm[1,1]}")

# 模型对比总结
logger.info("\n" + "="*60 + "\nMODEL COMPARISON SUMMARY\n" + "="*60)
logger.info(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'ROC AUC':<12}")
logger.info("-" * 80)
logger.info(f"{'LightGBM':<20} {accuracy_score(y_test, y_pred_lgb):<12.4f} {precision_score(y_test, y_pred_lgb):<12.4f} {recall_score(y_test, y_pred_lgb):<12.4f} {f1_score(y_test, y_pred_lgb):<12.4f} {roc_auc_score(y_test, y_prob_lgb):<12.4f}")
logger.info(f"{'XGBoost':<20} {accuracy_score(y_test, y_pred_xgb):<12.4f} {precision_score(y_test, y_pred_xgb):<12.4f} {recall_score(y_test, y_pred_xgb):<12.4f} {f1_score(y_test, y_pred_xgb):<12.4f} {roc_auc_score(y_test, y_prob_xgb):<12.4f}")
logger.info(f"{'Logistic Regression':<20} {accuracy_score(y_test, y_pred_lr):<12.4f} {precision_score(y_test, y_pred_lr):<12.4f} {recall_score(y_test, y_pred_lr):<12.4f} {f1_score(y_test, y_pred_lr):<12.4f} {roc_auc_score(y_test, y_prob_lr):<12.4f}")
logger.info(f"{'Random Forest':<20} {accuracy_score(y_test, y_pred_rf):<12.4f} {precision_score(y_test, y_pred_rf):<12.4f} {recall_score(y_test, y_pred_rf):<12.4f} {f1_score(y_test, y_pred_rf):<12.4f} {roc_auc_score(y_test, y_prob_rf):<12.4f}")
logger.info(f"{'Ensemble':<20} {accuracy_score(y_test, y_pred):<12.4f} {precision_score(y_test, y_pred):<12.4f} {recall_score(y_test, y_pred):<12.4f} {f1_score(y_test, y_pred):<12.4f} {roc_auc_score(y_test, y_prob_test):<12.4f}")

def post_process(pred, df_raw):
    high_risk = (df_raw['OverTime'] == 'Yes') & (df_raw['MonthlyIncome'] < 3000) & (df_raw['YearsAtCompany'] < 2)
    pred[high_risk] = 1
    return pred

X_test_raw = test_df.drop(columns=drop_cols, errors='ignore')
X_test_raw = ultimate_features(X_test_raw)
y_pred = post_process(y_pred, X_test_raw)

# 后处理后评估
logger.info("\n" + "="*60 + "\nPOST-PROCESSING EVALUATION\n" + "="*60)
logger.info(f"\n--- After Post-Processing (Rule-based Adjustment) ---")
logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
logger.info(f"Precision: {precision_score(y_test, y_pred):.4f}")
logger.info(f"Recall: {recall_score(y_test, y_pred):.4f}")
logger.info(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
logger.info("\nConfusion Matrix (After Post-Processing):")
cm_post = confusion_matrix(y_test, y_pred)
logger.info(f"True Negatives: {cm_post[0,0]}")
logger.info(f"False Positives: {cm_post[0,1]}")
logger.info(f"False Negatives: {cm_post[1,0]}")
logger.info(f"True Positives: {cm_post[1,1]}")

# ============================= 10. 结果保存 =============================
submission = pd.DataFrame({'Attrition': y_pred})
submission_path = os.path.join(DATA_DIR, 'submission_ultimate.csv')
submission.to_csv(submission_path, index=False)
logger.info(f"Predictions saved to {submission_path}")

# ============================= 11. 可视化 =============================
# 特征重要性
feature_names = (
    num_cols +
    [f"poly_{i}" for i in range(poly_features_train.shape[1])] +
    preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
)

lgb.fit(X_bal, y_bal)
importances = lgb.feature_importances_
feat_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False).head(15)

plt.figure(figsize=(10, 8))
sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='viridis')
plt.title('Top 15 Feature Importance (LightGBM)')
plt.xlabel('Importance')
plt.tight_layout()
imp_path = os.path.join(FIT_DIR, 'feature_importance.png')
plt.savefig(imp_path, dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"Feature importance saved to {imp_path}")

# ============================= 12. 模型调优可视化 =============================

# --- 1. GridSearch 结果热力图 ---
def plot_grid_search(grid, title, filename):
    results = pd.DataFrame(grid.cv_results_)
    scores = results['mean_test_score'].values.reshape(
        len(param_grid_lgb['n_estimators']),
        len(param_grid_lgb['max_depth']) * len(param_grid_lgb['learning_rate'])
    )
    # 简化：只展示 n_estimators vs max_depth，固定 learning_rate 最佳值
    plt.figure(figsize=(8, 6))
    sns.heatmap(scores, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=param_grid_lgb['max_depth'],
                yticklabels=param_grid_lgb['n_estimators'])
    plt.xlabel('max_depth')
    plt.ylabel('n_estimators')
    plt.title(title)
    path = os.path.join(FIT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Grid search plot saved: {path}")

plot_grid_search(grid_lgb, 'LightGBM Grid Search F1 Scores', 'grid_search_lgb.png')
plot_grid_search(grid_xgb, 'XGBoost Grid Search F1 Scores', 'grid_search_xgb.png')

# --- 2. 阈值调优曲线 ---
plt.figure(figsize=(10, 6))
thresholds = np.arange(0.3, 0.7, 0.01)
f1_scores = []
for thr in thresholds:
    f1 = f1_score(y_val, (y_prob >= thr).astype(int))
    f1_scores.append(f1)

plt.plot(thresholds, f1_scores, marker='o', color='purple', linewidth=2)
plt.axvline(x=best_thr, color='red', linestyle='--', label=f'Best Threshold = {best_thr:.3f}')
plt.title('Threshold Tuning: F1 Score vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True, alpha=0.3)
path = os.path.join(FIT_DIR, 'threshold_tuning.png')
plt.savefig(path, dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"Threshold tuning plot saved: {path}")

# --- 3. 真实值 vs 预测值比较（混淆矩阵） ---
if y_test is not None:
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['True 0', 'True 1'])
    plt.title('Confusion Matrix: True vs Predicted Attrition')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_path = os.path.join(FIT_DIR, 'true_vs_pred.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"True vs Predicted plot saved to {cm_path}")

# ============================= 13. 模型保存 =============================
import joblib

# 保存模型
joblib.dump(ensemble, os.path.join(MODEL_DIR, 'ensemble_model.pkl'))
joblib.dump(preprocessor, os.path.join(MODEL_DIR, 'preprocessor.pkl'))
joblib.dump(poly, os.path.join(MODEL_DIR, 'poly.pkl'))

# 保存阈值
with open(os.path.join(MODEL_DIR, 'threshold.txt'), 'w') as f:
    f.write(str(best_thr))

# ============================= 14. 评估指标可视化 =============================

# 模型评估指标
models = ['LightGBM', 'XGBoost', 'Logistic Regression', 'Random Forest', 'Ensemble', 'Ensemble (后处理)']
accuracy = [0.8629, 0.8657, 0.6886, 0.8500, 0.8543, 0.8571]
precision = [0.5556, 0.5833, 0.2778, 0.5000, 0.5227, 0.5319]
recall = [0.4717, 0.3962, 0.6604, 0.4500, 0.4340, 0.4717]
f1_score = [0.5102, 0.4719, 0.3911, 0.4800, 0.4742, 0.5000]
roc_auc = [0.8080, 0.8333, 0.7624, 0.8000, 0.8174, None]

# 准确率、精确率、召回率、F1分数对比图
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metric_values = [accuracy, precision, recall, f1_score]

plt.figure(figsize=(12, 8))

for i, (metric_name, values) in enumerate(zip(metrics, metric_values), 1):
    plt.subplot(2, 2, i)
    plt.bar(models, values, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.title(f'{metric_name} Comparison')
    plt.xlabel('Model')
    plt.ylabel(metric_name)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
metrics_path = os.path.join(FIT_DIR, 'model_metrics_comparison.png')
plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"Model metrics comparison saved to {metrics_path}")

# ROC AUC对比图
plt.figure(figsize=(12, 6))
roc_models = ['LightGBM', 'XGBoost', 'Logistic Regression', 'Random Forest', 'Ensemble']
roc_values = [0.8080, 0.8333, 0.7624, 0.8000, 0.8174]
plt.bar(roc_models, roc_values, color=['blue', 'green', 'red', 'orange', 'purple'])
plt.title('ROC AUC Comparison')
plt.xlabel('Model')
plt.ylabel('ROC AUC')
plt.ylim(0.7, 0.9)
plt.grid(axis='y', alpha=0.3)
roc_path = os.path.join(FIT_DIR, 'roc_auc_comparison.png')
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"ROC AUC comparison saved to {roc_path}")

logger.info(f"Models saved to {MODEL_DIR}")
logger.info(f"Training completed at {format_time(datetime.now())}")