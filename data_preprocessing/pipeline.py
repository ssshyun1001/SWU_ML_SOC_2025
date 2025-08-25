import pandas as pd
from data_prep import (
    split_column_types,
    encode_categorical_onehot_min_drop,
    scale_features,
    parse_composition_column,
)
from feature_select import (
    drop_low_information_features,
    drop_highly_correlated,
    remove_multicollinearity,
    remove_insignificant_features_by_pvalue,
)

DROP_COLS = ['material_id', 'electrolyte']
ONEHOT_COLS = ['material_structure', 'synthesis_method', 'counter_electrode', 'separator']
FORMULA_COLS = ['Li_source', 'Co_source', 'Mn_source', 'Ni_source']
TARGET = 'state_of_charge'


def preprocess_pipeline(df: pd.DataFrame, scaling_method='standard'):
    print(f"\n🟡 시작: 원본 데이터 shape = {df.shape}")

    # 1) 불필요 컬럼 제거
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # 2) dopant_fraction 처리
    if 'dopant_fraction' in df.columns and df['dopant_fraction'].dtype == 'object':
        df['dopant_fraction'] = pd.to_numeric(df['dopant_fraction'], errors='coerce').fillna(0.0)

    # 3) 타입 분리
    num_cols, cat_cols, dt_cols = split_column_types(df)
    print(f"🔹 수치형 {len(num_cols)}개 | 범주형 {len(cat_cols)}개 | 날짜형 {len(dt_cols)}개")

    # 4) 범주형 인코딩
    df = encode_categorical_onehot_min_drop(df, [c for c in ONEHOT_COLS if c in df.columns])

    # 5) 화학식 조성 벡터화
    for col in FORMULA_COLS:
        if col in df.columns:
            df = parse_composition_column(df, col)

    # 6) low-info 제거
    df = drop_low_information_features(df)

    # 7) bool → int 변환
    bool_cols = df.select_dtypes(include='bool').columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(int)

    # 8) 스케일링 (타깃 제외)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if TARGET in num_cols:
        num_cols.remove(TARGET)
    df = scale_features(df, num_cols, method=scaling_method)

    # 9) 상관관계 & 다중공선성 제거
    df = drop_highly_correlated(df)
    df = remove_multicollinearity(df, threshold=10.0, target_col=TARGET)

    print(f"✅ 최종 전처리 완료: shape = {df.shape}")
    return df
