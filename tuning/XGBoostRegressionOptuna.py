import optuna
import time
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def objective(trial):
    trial_start = time.time()  # 🔸 이 trial의 시작 시간

    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 400),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth":        trial.suggest_int("max_depth", 6, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 4),
        "subsample":        trial.suggest_float("subsample", 0.8, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
        "gamma":            trial.suggest_float("gamma", 0.0, 0.3),

        # CPU 설정으로 변경
        "tree_method":      "hist",           # CPU용 히스토그램 방법
        "device":           "cpu",            # 명시적으로 CPU 지정
        "random_state":     42,
        "n_jobs":           -1,               # 모든 CPU 코어 사용
        "eval_metric":      "rmse",
        "objective":        "reg:squarederror",
        "early_stopping_rounds": 20          # 조기 종료를 파라미터에 포함
    }

    model = XGBRegressor(**params)

    # XGBoost 버전에 따른 호환성 처리
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    except TypeError:
        # early_stopping_rounds가 fit()에서 지원되지 않는 경우
        params_without_early_stop = params.copy()
        params_without_early_stop.pop('early_stopping_rounds', None)
        model = XGBRegressor(**params_without_early_stop)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    # 🔸 시간 출력
    trial_time = time.time() - trial_start
    total_elapsed = time.time() - start
    print(f"[Trial {trial.number:2d}] RMSE: {rmse:.5f} | Trial time: {trial_time/60:.2f}분 | Total: {total_elapsed/60:.2f}분 | Params: {trial.params}")

    return rmse

# Optuna 최적화 실행
start = time.time()
study = optuna.create_study(direction="minimize", study_name="xgb_cpu_tuning")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\n" + "="*60)
print("▶ Best params :", study.best_params)
print(f"▶ Best RMSE   : {study.best_value:.5f}")

# 최적 파라미터로 최종 모델 학습
best_params = study.best_params.copy()
best_params.update({
    "tree_method": "hist",
    "device": "cpu",
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "rmse",
    "objective": "reg:squarederror"
})

print("\n▶ 최적 파라미터로 전체 데이터셋에서 재학습 중...")
best_model = XGBRegressor(**best_params)
best_model.fit(X_train_full, y_train_full)

# 테스트 성능 평가
y_pred = best_model.predict(X_test)

print("\n" + "="*60)
print("▶ 최종 테스트 성능:")
print(f"▶ Test RMSE   : {np.sqrt(mean_squared_error(y_test, y_pred)):.5f}")
print(f"▶ Test R²     : {r2_score(y_test, y_pred):.5f}")
print(f"⏱️ 총 경과시간: {time.time() - start:.2f}초")