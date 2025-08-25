import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# 유일값 + 저분산 제거
def drop_low_information_features(df):
    nunique = df.nunique()
    to_drop = nunique[nunique == 1].index.tolist()

    selector = VarianceThreshold(threshold=1e-5)
    numeric = df.select_dtypes(include=['float64', 'int64'])
    selector.fit(numeric)
    low_var_cols = numeric.columns[~selector.get_support()]
    to_drop += low_var_cols.tolist()

    return df.drop(columns=list(set(to_drop)))


# 상관계수 기반 제거
def drop_highly_correlated(df, threshold=0.95):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    print(f"⚠️ 상관계수>{threshold}로 제거된 변수 수: {len(to_drop)}")
    return df.drop(columns=to_drop)


# VIF 기반 다중공선성 제거
def remove_multicollinearity(df, threshold=10.0, target_col=None):
    numeric = df.select_dtypes(include=['float64', 'int64'])
    if target_col and target_col in numeric.columns:
        numeric = numeric.drop(columns=[target_col])

    removed_cols = []

    while True:
        vif_dict = {}
        for col in numeric.columns:
            X = numeric.drop(columns=[col])
            y = numeric[col]
            model = LinearRegression().fit(X, y)
            r2 = model.score(X, y)
            vif = 1 / (1 - r2) if r2 < 1 else float('inf')
            vif_dict[col] = vif

        vif_series = pd.Series(vif_dict).sort_values(ascending=False)
        max_vif = vif_series.iloc[0]

        if max_vif > threshold:
            drop_col = vif_series.index[0]
            print(f"⚠️ 컬럼 제거: '{drop_col}' (VIF={max_vif:.2f})")
            removed_cols.append((drop_col, max_vif))
            numeric = numeric.drop(columns=[drop_col])
            df = df.drop(columns=[drop_col])
        else:
            break

    return df


# p-value 기반 제거
def remove_insignificant_features_by_pvalue(df, y_col, alpha=0.05):
    X = df.drop(columns=[y_col]).copy()
    y = df[y_col]

    bool_cols = X.select_dtypes(include='bool').columns
    X[bool_cols] = X[bool_cols].astype(int)
    X = X.select_dtypes(include=['float64', 'int64'])

    while True:
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        pvalues = model.pvalues.drop('const')

        max_p = pvalues.max()
        if max_p > alpha:
            drop_col = pvalues.idxmax()
            print(f"⚠️ 컬럼 제거: '{drop_col}' (p-value={max_p:.4f})")
            X = X.drop(columns=[drop_col])
            df = df.drop(columns=[drop_col])
        else:
            break

    return df
