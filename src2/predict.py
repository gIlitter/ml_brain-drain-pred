import pandas as pd
import numpy as np
import joblib
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.commonUtil import get_data_source
from util.logUtil import get_logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data2')
MODEL_DIR = os.path.join(BASE_DIR, 'model2')
LOG_DIR = os.path.join(BASE_DIR, 'log2')

logger = get_logger('predict', log_dir=LOG_DIR)


def load_model():
    model_path = os.path.join(MODEL_DIR, 'ensemble_model.pkl')
    preprocessor_path = os.path.join(MODEL_DIR, 'preprocessor.pkl')
    poly_path = os.path.join(MODEL_DIR, 'poly.pkl')
    threshold_path = os.path.join(MODEL_DIR, 'threshold.txt')

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    poly = joblib.load(poly_path)
    with open(threshold_path, 'r') as f:
        threshold = float(f.read().strip())

    return model, preprocessor, poly, threshold


def predict(test_path):
    model, preprocessor, poly, threshold = load_model()

    test_df = get_data_source(test_path)
    X_test = test_df.copy()

    drop_cols = ['EmployeeNumber', 'Over18', 'StandardHours']
    X_test = X_test.drop(columns=drop_cols, errors='ignore')

    # 计算RiskScore（与训练一致）
    X_test_cox = X_test.copy()
    X_test_cox['OverTime'] = X_test_cox['OverTime'].map({'Yes': 1, 'No': 0}).fillna(0)
    X_test_cox['BusinessTravel'] = X_test_cox['BusinessTravel'].map({'Travel_Frequently': 2, 'Travel_Rarely': 1, 'Non-Travel': 0}).fillna(1)
    X_test_cox['OT_Travel'] = X_test_cox['OverTime'] * X_test_cox['BusinessTravel']
    X_test_cox['SatisfactionMean'] = X_test_cox[['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction']].mean(axis=1)
    
    # 特征工程（与训练一致）
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

    X_test = ultimate_features(X_test)

    # 重新计算RiskScore（简化版本，使用固定系数）
    # 注意：这里使用了训练时的系数，实际应用中应该保存Cox模型
    cox_coeffs = {
        'Age': -0.0585,
        'MonthlyIncome': -0.0002,
        'OverTime': 1.8491,
        'BusinessTravel': 0.5542,
        'SatisfactionMean': -0.5684,
        'OT_Travel': -0.5037
    }
    
    # 计算RiskScore
    X_test['RiskScore'] = (
        X_test_cox['Age'] * cox_coeffs['Age'] +
        X_test_cox['MonthlyIncome'] * cox_coeffs['MonthlyIncome'] +
        X_test_cox['OverTime'] * cox_coeffs['OverTime'] +
        X_test_cox['BusinessTravel'] * cox_coeffs['BusinessTravel'] +
        X_test_cox['SatisfactionMean'] * cox_coeffs['SatisfactionMean'] +
        X_test_cox['OT_Travel'] * cox_coeffs['OT_Travel']
    )

    interaction_cols = ['MonthlyIncome', 'TotalWorkingYears', 'Age', 'DistanceFromHome']

    poly_features = poly.transform(X_test[interaction_cols])
    X_test = X_test.drop(columns=interaction_cols)

    X_main = preprocessor.transform(X_test)
    X_processed = np.hstack([X_main, poly_features])

    prob = model.predict_proba(X_processed)[:, 1]
    pred = (prob >= threshold).astype(int)

    # 后处理
    high_risk = (test_df['OverTime'] == 'Yes') & (test_df['MonthlyIncome'] < 3000) & (test_df['YearsAtCompany'] < 2)
    pred[high_risk] = 1

    return pred


if __name__ == "__main__":
    test_path = os.path.join(DATA_DIR, 'new_test2.csv')
    pred = predict(test_path)
    submission = pd.DataFrame({'Attrition': pred})
    submission.to_csv(os.path.join(DATA_DIR, 'submission_predict.csv'), index=False)
    logger.info("Prediction completed.")