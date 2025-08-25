import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_extraction import DictVectorizer
from pymatgen.core.composition import Composition


# 컬럼 타입 자동 분리
def split_column_types(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    return numerical_cols, categorical_cols, datetime_cols


# 이상치 제거 (IQR)
def remove_outliers_iqr(df, numerical_cols):
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


# 스케일링
def scale_features(df, numerical_cols, method='standard'):
    scaler = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }[method]
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df


# 범주형 인코딩 (One-hot, 최소값 드랍)
def encode_categorical_onehot_min_drop(df, categorical_cols):
    for col in categorical_cols:
        counts = df[col].value_counts()
        drop_value = counts.idxmin()
        dummies = pd.get_dummies(df[col], prefix=col)
        drop_col = f"{col}_{drop_value}"
        if drop_col in dummies.columns:
            dummies = dummies.drop(columns=[drop_col])
        df = df.drop(columns=[col])
        df = pd.concat([df, dummies], axis=1)
    return df


# 화학식 조성 벡터화
def parse_composition_column(df, column, drop_original=True):
    comp_dicts = []
    for formula in df[column]:
        try:
            if isinstance(formula, str) and formula.lower() in ['other', 'others', 'unknown', 'nan']:
                comp_dict = {}
            else:
                comp = Composition(formula)
                comp_dict = comp.get_el_amt_dict()
        except Exception:
            comp_dict = {}
        comp_dicts.append(comp_dict)

    vec = DictVectorizer(sparse=False)
    comp_array = vec.fit_transform(comp_dicts)
    comp_df = pd.DataFrame(comp_array, columns=[f"{column}_el_{el}" for el in vec.feature_names_])

    if drop_original:
        df = df.drop(columns=[column])

    return pd.concat([df.reset_index(drop=True), comp_df.reset_index(drop=True)], axis=1)
