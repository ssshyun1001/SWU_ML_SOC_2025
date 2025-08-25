from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 입력 파일 (프로젝트 루트/data/dataset_processed.csv)
BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / "data" / "dataset_processed.csv"

# 1) 데이터 로드
df = pd.read_csv(INPUT)

# 2) 타깃/피처 분리 (있을 때만 드랍)
y = df["state_of_charge"]
drop_cols = ["discharge_capacity (mAh/g)", "state_of_charge", "Strain"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])

# 3) 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 5) 평가
pred_tr = model.predict(X_train)
pred_te = model.predict(X_test)

rmse_tr = np.sqrt(mean_squared_error(y_train, pred_tr))
rmse_te = np.sqrt(mean_squared_error(y_test, pred_te))
r2_tr = r2_score(y_train, pred_tr)
r2_te = r2_score(y_test, pred_te)

print("📚 LinearRegression")
print(f"Train: RMSE={rmse_tr:.4f}, R²={r2_tr:.4f}")
print(f"Test : RMSE={rmse_te:.4f}, R²={r2_te:.4f}")