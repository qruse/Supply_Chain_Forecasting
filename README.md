# TSFM 수요예측 워크플로우

이 저장소는 DataCo Supply Chain 데이터셋을 SKU 단위 수요예측 데이터로 정리하고, 기준선 모델을 학습하고 평가하는 흐름을 담고 있습니다.

## 1. 의존성 설치

```bash
python -m pip install numpy pandas scikit-learn statsmodels
```

GPU 학습을 사용할 수 있는 환경이면 CUDA 빌드 PyTorch를 설치하세요.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

GPU가 없는 환경에서는 CPU 빌드를 설치해도 됩니다.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Chronos 계열 실험을 하려면 추가로 설치하세요.

```bash
python -m pip install chronos-forecasting==2.2.2 peft
```

Granite TTM과 TimesFM 2.5 실험을 하려면 추가로 설치하세요.

```bash
python -m pip install granite-tsfm
python -m pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git"
```

## 2. 원본 데이터 전처리

정제된 거래 테이블을 생성합니다.

```bash
python preprocessing.py
```

이 단계에서 하는 일:

- 불필요한 개인정보 컬럼 제거
- `Customer Zipcode` 제거
- 누수 가능성이 있는 배송 관련 컬럼 제거
  - `shipping date (DateOrders)`
  - `Days for shipping (real)`
  - `Late_delivery_risk`
  - `Delivery Status`
- 정제 결과를 `data/df_cleaned.csv`로 저장
- 이후 단계에서 사용할 일별 집계 파일도 함께 생성

## 3. SKU 단위 데이터 생성

SKU별 일 단위 패널과 `30일 입력 -> 7일 예측` 학습 데이터를 생성합니다.

```bash
python data_build.py
```

생성되는 파일:

- `data/sku_daily.csv`
- `data/sku_daily_train.csv`
- `data/sku_daily_val.csv`
- `data/sku_daily_test.csv`
- `data/sku_xy_30_7_train.npz`
- `data/sku_xy_30_7_val.npz`
- `data/sku_xy_30_7_test.npz`

구성:

- 단위: `SKU x day`
- 입력 길이: 30일
- 예측 구간: 7일
- 분할 방식: 날짜 기준으로 연속 구간 분할, 대략 70/15/15
- `X_num`: 일별 숫자형 시계열 입력
- `X_static`: 문자열 범주를 one-hot 인코딩한 고정 입력

## 4. 기준선 모델 학습

기준선 모델을 학습합니다.

```bash
python training.py --model arima
python training.py --dataset demand --model rnn
python training.py --dataset inventory --model lstm
python training.py --dataset leadtime --model tcn
```

한 번에 모두 학습하려면:

```bash
python training.py --dataset all --model all
```

학습 옵션:

- `--dataset`: `demand`, `inventory`, `leadtime`, `all` 중 선택. 기본값 `demand`
- `--epochs`: 기본값 `12`
- `--batch-size`: 기본값 `32`, 더 크게 입력해도 `32`로 제한
- `--device`: `auto`, `cpu`, `cuda` 중 선택. 기본값 `auto`이며 CUDA 사용 가능 시 GPU를 사용
- `--lr`: 학습률, 기본값 `0.001`
- `--hidden-size`: recurrent hidden size, 기본값 `48`
- `--num-layers`: recurrent layer 수, 기본값 `1`
- `--dropout`: dropout 비율, 기본값 `0.10`
- `--patience`: early stopping patience, 기본값 `3`
- `--seed`: 난수 시드

예시:

```bash
python training.py --dataset demand --model lstm --epochs 12 --batch-size 32 --hidden-size 48 --num-layers 1 --dropout 0.1 --patience 3
```

학습 방식:

- 매 epoch마다 validation을 적용
- 매 epoch checkpoint 저장
- 최고 성능 checkpoint는 `best_...` 이름으로 별도 저장
- checkpoint는 `artifacts/checkpoints/<dataset>/<model_name>/` 아래에 저장
- 기본적으로 학습 시작 시 해당 dataset/model의 오래된 epoch/best checkpoint와 history를 정리합니다. 이전 checkpoint를 보존하려면 `--keep-old-checkpoints` 또는 `KEEP_OLD_CHECKPOINTS=1`을 사용합니다.
- `ARIMA`는 단변량 기준선으로 유지
- `RNN`은 가장 기본적인 recurrent baseline
- `TCN`은 causal dilated 1D convolution 기반 neural baseline
- 문자열 one-hot을 외생변수로 쓰는 선형 시계열 모델을 만들고 싶다면, `VARIMA`보다 `ARIMAX` / `SARIMAX`가 더 맞는 표현

## 5. 모든 checkpoint 테스트

저장된 모든 epoch checkpoint와 best checkpoint를 test split에 대해 평가합니다.

```bash
python testing.py --dataset all --model all
```

이 명령은 결과를 출력하고 CSV 파일로도 저장합니다.

baseline 결과는 `artifacts/results/baseline/<dataset>/<model_name>/` 아래에 저장됩니다.

- `artifacts/results/baseline/demand/arima/arima_test_metrics.csv`
- `artifacts/results/baseline/demand/rnn/rnn_test_metrics.csv`
- `artifacts/results/baseline/inventory/lstm/lstm_test_metrics.csv`
- `artifacts/results/baseline/leadtime/tcn/tcn_test_metrics.csv`
- `artifacts/results/baseline/arima/arima_test_sku_metrics.csv`
- `artifacts/results/baseline/rnn/rnn_test_sku_metrics.csv`
- `artifacts/results/baseline/lstm/lstm_test_sku_metrics.csv`
- `artifacts/results/baseline/gru/gru_test_sku_metrics.csv`
- `artifacts/results/baseline/tcn/tcn_test_sku_metrics.csv`
- `artifacts/results/baseline/arima/arima_test_sku_mae.csv`
- `artifacts/results/baseline/rnn/rnn_test_sku_mae.csv`
- `artifacts/results/baseline/lstm/lstm_test_sku_mae.csv`
- `artifacts/results/baseline/gru/gru_test_sku_mae.csv`
- `artifacts/results/baseline/tcn/tcn_test_sku_mae.csv`
- `artifacts/results/baseline/arima/arima_test_forecasts.csv`

test metric에는 `MAE`, `MSE`, `RMSE`, `MAPE`, `R2`가 들어갑니다.
SKU/target별 metric CSV에도 `MAE`, `RMSE`, `MAPE`, `R2`가 저장됩니다. 실제값이 0인 관측치는 MAPE 계산에서 제외하고, 전부 0인 target은 `NaN`으로 둡니다.

## 6. TSFM / Foundation Model 실험

Chronos, Granite TTM, TimesFM 기반 TSFM 실험은 아래 스크립트로 돌립니다.

```bash
python foundation_experiments.py --dataset demand --model all --split both
```

개별 모델만 돌리고 싶으면:

```bash
python foundation_experiments.py --dataset demand --model chronos1 --split test
python foundation_experiments.py --dataset inventory --model chronosbolt --split test
python foundation_experiments.py --dataset leadtime --model chronos2 --split test
python foundation_experiments.py --dataset demand --model chronos2_ft --split test --finetune-mode lora --num-steps 100
python foundation_experiments.py --dataset demand --model ttm --split test
python foundation_experiments.py --dataset demand --model ttm_ft --split test --num-steps 100
python foundation_experiments.py --dataset inventory --model timesfm --split test
```

TSFM/Chronos 실행의 `--dataset`은 `demand`, `inventory`, `leadtime`, `all`을 지원합니다. `--batch-size` 기본값은 `16`이고, 더 크게 입력해도 `16`으로 제한합니다. `--device auto`가 기본값이며 CUDA 사용 가능 시 GPU를 사용합니다.
`--model all`은 세 데이터셋 공통 비교가 가능하도록 Chronos-1, Chronos-Bolt, Chronos-2, Chronos-2 LoRA fine-tuning, TimesFM 2.5 zero-shot을 실행합니다. Granite TTM/TTM fine-tuning은 공통 비교에서 제외했으며, 필요할 때만 `--model ttm` 또는 `--model ttm_ft`로 개별 실행합니다.
`timesfm`은 공식 TimesFM 2.5 torch 패키지 기반 zero-shot 경로입니다. TimesFM 2.5 fine-tuning은 별도 Transformers 2.5 지원 환경이 필요해서 현재 공용 실행 경로에서는 비활성화해 둡니다.
현재 저장소의 기본 실험 프리셋은 `configs/experiment_presets.json`에 정리되어 있으며, 리소스 절약을 위해 작은 모델 크기와 짧은 fine-tuning step을 우선 사용합니다.

출력 위치:

- 예측 결과 CSV: `artifacts/results/chronos/<dataset>/<model_name>/`
- fine-tuned Chronos-2 저장본: `artifacts/foundation_models/`
- 모델별 메트릭: `artifacts/results/chronos/<dataset>/<model_name>/<model_name>_<split>_metrics.csv`
- 모델별 SKU 메트릭: `artifacts/results/chronos/<dataset>/<model_name>/<model_name>_<split>_sku_metrics.csv`
- 모델별 SKU MAE: `artifacts/results/chronos/<dataset>/<model_name>/<model_name>_<split>_sku_mae.csv`

모델별 입력 방식:

- `chronos1`: target만 사용
- `chronosbolt`: target만 사용
- `chronos2`: target + past/future covariates 사용
- `chronos2_ft`: Chronos-2를 현재 데이터셋으로 fine-tuning
- `ttm`: Granite TTM target-only zero-shot, 개별 실행용
- `ttm_ft`: Granite TTM target-only fine-tuning, 개별 실행용
- `timesfm`: TimesFM 2.5 target-only zero-shot

주의:

- Chronos-2는 문자열 범주를 one-hot으로 바꾸지 않고, covariate 타입에 맞춰 그대로 입력할 수 있습니다.
- Chronos-1과 Chronos-Bolt는 target-only zero-shot으로 두는 편이 맞습니다.
- zero-shot과 fine-tuning 결과는 서로 다른 모델 폴더와 CSV로 저장됩니다.
- TSFM test metric에도 `MAE`, `MSE`, `RMSE`, `MAPE`, `R2`가 같이 저장됩니다.

## 7. 코드 구성

- `model.py`: 공통 모델 정의와 ARIMA helper
- `forecasting_data.py`: `demand`, `inventory`, `leadtime` 데이터셋 정의와 공통 window 생성
- `training.py`: ARIMA, RNN, LSTM, GRU, TCN 학습 및 checkpoint 저장
- `testing.py`: 저장된 모든 checkpoint를 test set에서 평가
- `foundation_experiments.py`: Chronos-1, Chronos-Bolt, Chronos-2 zero-shot 및 Chronos-2 fine-tuning

## 8. 전체 파이프라인 스크립트

데이터 처리부터 학습/검증까지 한 번에 실행하려면:

```bash
scripts/run_full_pipeline.sh
```

기본 실행 범위:

- DataCo 수요 데이터 전처리 및 SKU 패널 생성
- Inventory 데이터 생성
- BPI lead-time 데이터 생성
- 데이터 품질 점검
- DataCo 수요예측 baseline 전체 학습: `ARIMA`, `RNN`, `LSTM`, `GRU`, `TCN`
- Inventory baseline 전체 학습: `ARIMA`, `RNN`, `LSTM`, `GRU`, `TCN`
- BPI lead-time baseline 전체 학습: `ARIMA`, `RNN`, `LSTM`, `GRU`, `TCN`
- baseline test 평가
- TSFM 실험: 기본 실행. `TSFM_DATASET=all` 기본값으로 demand, inventory, leadtime을 순회하며 공통 모델 세트를 실행합니다.
  - zero-shot: `Chronos-1`, `Chronos-Bolt`, `Chronos-2`, `TimesFM 2.5`
  - fine-tuning: `Chronos-2 LoRA`
  - 개별 실행용: `Granite TTM`, `Granite TTM fine-tuning`
  - 제외: `TimesFM 2.5 fine-tuning`은 현재 공용 의존성 환경에서 안정적인 기본 경로가 없어 제외

주요 옵션은 환경변수로 조정합니다.

```bash
DEVICE=cuda BASELINE_EPOCHS=12 TSFM_STEPS=100 scripts/run_full_pipeline.sh
RUN_TSFMS=1 TSFM_DATASET=inventory TSFM_MODEL=timesfm TSFM_SPLIT=val scripts/run_full_pipeline.sh
RUN_DATA=0 RUN_BASELINES=1 RUN_TSFMS=0 scripts/run_full_pipeline.sh
FORCE_REBUILD=1 scripts/run_full_pipeline.sh
TSFM_DATASET=leadtime TSFM_MODEL=chronos2 TSFM_SPLIT=val scripts/steps/20_run_tsfm.sh
```

데이터 준비 단계는 필수 산출물과 품질 리포트가 이미 있으면 자동으로 건너뜁니다. 다시 만들고 싶으면 `FORCE_REBUILD=1`을 붙입니다.

단계별 스크립트:

- `scripts/steps/01_prepare_demand_data.sh`
- `scripts/steps/02_prepare_inventory_data.sh`
- `scripts/steps/03_prepare_leadtime_data.sh`
- `scripts/steps/04_validate_data_quality.sh`
- `scripts/steps/10_train_baselines.sh`
- `scripts/steps/11_test_baselines.sh`
- `scripts/steps/20_run_tsfm.sh`

모든 실행 로그는 `test_logs/` 아래에 저장됩니다.

## 9. 현재 모델링 설정

- 예측 대상: SKU별 일 수요
- 입력 길이: 30일
- 예측 길이: 다음 7일
- 손실 함수: MSE
- 숫자형 시계열 입력은 train split 통계로 표준화
- 문자열 범주 입력은 one-hot 인코딩 후 RNN/LSTM/GRU/TCN의 static input으로 사용
- categorical embedding은 아직 사용하지 않음

## 10. 참고

- `EDA.ipynb`에는 수요 집중도, 간헐수요, SKU별 zero-share 분석이 들어 있습니다.
- 현재 파이프라인은 행 단위 예측이 아니라 계층형 SKU 수요예측에 맞춰져 있습니다.
