from pathlib import Path
import pandas as pd
from pipeline import preprocess_pipeline

# 프로젝트 루트: data_preprocessing의 한 단계 위
BASE_DIR = Path(__file__).resolve().parents[1]

# ✅ 실제 파일명에 맞게 절대경로로 지정
DEFAULT_INPUT  = BASE_DIR / "data" / "raw_battery_sample.csv"
DEFAULT_OUTPUT = BASE_DIR / "data" / "dataset_processed.csv"

def run_preprocessing(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT):
    print("🚀 전처리 실행 시작")
    print(f"📁 BASE_DIR  : {BASE_DIR}")
    print(f"📥 입력 파일 : {input_path}")

    if not Path(input_path).exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    df = pd.read_csv(input_path)
    print(f"📂 불러온 데이터 shape: {df.shape}")

    df_processed = preprocess_pipeline(df, scaling_method='standard')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    print(f"✅ 전처리 완료. 결과 저장: {output_path}")

if __name__ == "__main__":
    run_preprocessing()
