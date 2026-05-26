# Supply Chain Uncertainty Forecasting & TSFM Benchmark

이 저장소는 공급망 관리(SCM) 시스템의 핵심 불확실성인 **수요(Demand)**, **재고(Inventory)**, **리드타임(Lead-time)**을 예측하기 위해 기존의 정교한 통계/딥러닝 기준 모델들과 시계열 기초 모델(Time Series Foundation Models, TSFM)을 비교하고 벤치마킹하는 연구 및 개발 워크플로우를 담고 있습니다.

본인의 고유 시계열 데이터(CSV) 파일로도 즉시 학습하고 예측할 수 있도록 일반화된 범용 예측 스크립트(`run_custom_dataset.py`)를 함께 제공합니다.

---

## 1. WSL2 CUDA 및 로컬 가상환경 구축

이 프로젝트는 GPU 가속(CUDA)을 활용한 딥러닝 학습 및 Foundation Model 추론을 지원합니다. WSL2(Windows Subsystem for Linux) 환경 및 리눅스 장비에서 GPU를 온전히 활용할 수 있도록 환경을 다음과 같이 설정합니다.

### 1-1. WSL2 CUDA 라이브러리 경로 매핑 (필요시)
WSL2에서 NVIDIA GPU 노드를 올바르게 바인딩하기 위해 시스템 라이브러리 경로 환경변수를 쉘 설정 파일(`~/.bashrc` 등)에 추가합니다.
```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

### 1-2. 가상환경 구성 및 패키지 설치
Python 3.10 ~ 3.11 환경에서 가상환경을 생성하고 아래 의존성을 순서대로 설치합니다.

```bash
# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/bin/activate  # 또는 source .venv/bin/activate

# 1. 기본 시계열 분석 및 데이터 과학 패키지 설치
python -m pip install numpy pandas scikit-learn statsmodels nbformat

# 2. CUDA 지원 PyTorch 설치 (GPU가 없는 경우 cpu 인덱스 사용)
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Chronos 계열 및 PEFT (LoRA 파인튜닝용) 라이브러리 설치
python -m pip install chronos-forecasting==2.2.2 peft

# 4. IBM Granite TTM 및 Google TimesFM 2.5 설치
python -m pip install granite-tsfm
python -m pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git"
```

---

## 2. 사용자 커스텀 시계열 데이터(CSV)로 학습 및 예측하기

공급망 벤치마크 모델뿐만 아니라, **사용자가 보유한 임의의 시계열 CSV 파일**에 대해서도 빠르게 예측을 수행할 수 있는 `run_custom_dataset.py` 유틸리티를 제공합니다.

### 2-1. 지원 모델 목록
* `arima`: 전통적인 자기회귀 이동평균 통계 모델
* `lstm`: 딥러닝 기반의 대표적인 재귀 신경망(Recurrent Neural Network) 모델
* `tcn`: 인과관계 딜레이트 합성곱(Causal Dilated 1D CNN) 신경망 모델
* `chronosbolt`: Amazon의 제로샷(Zero-shot) 시계열 기초 모델 (최고 성능 수준)

### 2-2. 실행 인자(Argument) 설명
* `--csv-path`: 예측을 수행할 커스텀 CSV 파일의 경로 (필수)
* `--date-col`: 시계열 타임스탬프가 포함된 날짜 열의 이름 (필수)
* `--target-col`: 예측하고자 하는 수량 또는 수치 열의 이름 (필수)
* `--series-col`: 데이터를 구분하는 고유 상품 ID 또는 계정 코드 열 이름 (다중 시계열일 경우 필수)
* `--model`: 사용할 예측 모델 선택 (`arima`, `lstm`, `tcn`, `chronosbolt`)
* `--lookback`: 과거 며칠 동안의 데이터를 입력으로 쓸 것인지 설정 (기본값: `30`일)
* `--horizon`: 향후 며칠 앞의 수요를 예측할 것인지 설정 (기본값: `7`일)
* `--epochs`: 딥러닝 모델 학습을 수행할 최대 에폭 (기본값: `10`)
* `--device`: 학습을 수행할 장치 선택 (`auto`, `cpu`, `cuda`)

### 2-3. 실행 예제

```bash
# 1. 테스트용 모의 시계열 데이터셋 생성 (tmps/dummy_ts.csv 생성됨)
python tmps/generate_dummy_data.py

# 2. LSTM 모델을 사용하여 예측 테스트 (CPU 구동)
python run_custom_dataset.py \
  --csv-path tmps/dummy_ts.csv \
  --date-col date \
  --target-col demand \
  --series-col item_id \
  --model lstm \
  --epochs 5 \
  --device cpu

# 3. Amazon Chronos-Bolt Foundation Model로 제로샷(Zero-shot) 예측 실행 (GPU 구동)
python run_custom_dataset.py \
  --csv-path tmps/dummy_ts.csv \
  --date-col date \
  --target-col demand \
  --series-col item_id \
  --model chronosbolt \
  --device cuda
```

### 2-4. 출력 산출물
실행이 완료되면 테스트 세트에 대한 메트릭과 각 시계열 그룹별 예측값이 자동으로 저장됩니다.
* 예측 결과 CSV: `artifacts/results/custom/<model>_predictions.csv`
* 성능 평가지표 JSON (MAE, MSE, RMSE, R2): `artifacts/results/custom/<model>_metrics.json`

---

## 3. 내장 공급망 데이터 전처리 및 학습

이 저장소에 기본 탑재된 3대 공급망 도메인 데이터를 가공하고 실험을 전체 파이프라인으로 돌리는 가이드입니다.

### 3-1. 원본 데이터 정제 및 SKU 패널 생성
```bash
# DataCo 거래 데이터 정제 (PII 컬럼 제거 및 누수 차단)
python preprocessing.py

# 일별 SKU 단위 수요예측 텐서 파일(.npz) 생성
python data_build.py
```
* **생성 파일:** `data/df_cleaned.csv`, `data/sku_daily.csv` 및 `data/sku_xy_30_7_*.npz`

### 3-2. 공급망 벤치마크 Baseline 모델 학습
```bash
python training.py --dataset demand --model rnn
python training.py --dataset inventory --model lstm
python training.py --dataset leadtime --model tcn
```
한 번에 세 데이터셋 전체 모델을 학습하려면 아래 명령을 실행합니다.
```bash
python training.py --dataset all --model all --epochs 12
```
* 모델 체크포인트는 `artifacts/checkpoints/<dataset>/<model_name>/` 아래에 자동 저장됩니다.

### 3-3. 모든 체크포인트 모델 평가
```bash
python testing.py --dataset all --model all
```
* 평가 결과 메트릭 및 전체 예측값 CSV는 `artifacts/results/baseline/` 경로 하위에 저장됩니다.

### 3-4. TSFM / 시계열 기초 모델 실험
Amazon Chronos, IBM Granite TTM, Google TimesFM의 제로샷 및 파인튜닝 실험 스크립트입니다.
```bash
python foundation_experiments.py --dataset demand --model all --split both
```
개별 Foundation Model만 실행하고자 할 경우:
```bash
python foundation_experiments.py --dataset inventory --model timesfm --split test
python foundation_experiments.py --dataset demand --model chronos2_ft --split test --finetune-mode lora --num-steps 100
```

---

## 4. 전체 파이프라인 원클릭 오케스트레이션

데이터 정제부터 시계열 패널 구축, 품질 점검, 재귀 신경망(RNN/LSTM/GRU/TCN) 모델 학습, 그리고 기초 모델(TSFM) 검증까지 모든 파이프라인을 쉘 스크립트 단 한 번의 실행으로 완수할 수 있습니다.

```bash
# 기본 실행 (CUDA 장치가 감지되면 우선 사용하여 전체 자동 수행)
scripts/run_full_pipeline.sh

# 환경변수를 조절하여 특정 디바이스 및 에폭 지정 실행
DEVICE=cuda BASELINE_EPOCHS=12 TSFM_STEPS=100 scripts/run_full_pipeline.sh
```
* 실행 과정에서 발생하는 디버그 로그와 에러 트레이스는 `test_logs/` 디렉토리 하위에 상세히 누적 보관됩니다.

---

## 5. 프로젝트 소스 코드 구조

* `run_custom_dataset.py`: [NEW] 사용자의 CSV 파일로 시계열 학습/예측을 진행하는 유틸리티
* `preprocessing.py`: DataCo 원본 거래 데이터의 PII 제거 및 노이즈 필터링
* `data_build.py`: 트랜잭션 데이터를 시계열 SKU 패널로 변환하고 캘린더 피처 추가
* `forecasting_data.py`: `demand`, `inventory`, `leadtime` 각 도메인 데이터셋 관리 레지스트리 및 윈도우 생성 클래스
* `model.py`: ARIMA 래퍼 및 RNN/LSTM/GRU/TCN 딥러닝 아키텍처 모델 정의
* `training.py`: 각 도메인 데이터셋별 딥러닝 Baseline 모델 지도 학습
* `testing.py`: 도메인별 학습된 체크포인트 가중치 테스트셋 평가 및 성능 지표 산출
* `foundation_experiments.py`: Chronos 계열, Granite TTM, TimesFM 기초 시계열 모델의 제로샷 추론 및 파인튜닝
* `build_inventory_dataset.py`: Mendeley Retail 인벤토리 원본 데이터 정제 및 일별 재고 패널 변환
* `build_leadtime_dataset.py`: BPI 2019 이벤트 로그로부터 구매주문(PO)-물품입고(GR) 리드타임 패널 데이터 생성
* `configs/experiment_presets.json`: 학습 설정 변수 및 파라미터 최적화 프리셋 정의
* `artifacts/dataset_notes/`: 각 공급망 시계열 벤치마크 데이터셋의 도메인 분석 리포트
