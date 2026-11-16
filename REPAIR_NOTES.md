# 코드 수정 및 검증 기록

기준 커밋: `b5afb1cdb0159f72e04474aea6bb483add47de36`

이 변경은 입력 검증과 재현 절차를 개선하는 수정이다.
기존 측정 데이터, 저장 모델, 평가·예측 결과는 변경하지 않았다.

## 수정 내용

- CSV의 중간 빈 셀을 제거하지 않고 오류로 처리한다. 좌표·스펙트럼은 같은 파서로 읽는다.
- 입력·모델·template의 파수축을 검사한다. 같은 채널 수라도 파수 값이 다르면 거부한다.
- 기준 band 누락, 비정상 수치, 빈 입력과 정규화 불가능한 결과를 명시적으로 처리한다.
- 모든 pixel이 포화된 경우에도 학습 제외 규칙을 적용한다. 남는 pixel이 없으면 중단한다.
- 새 학습 캐시는 annotation, 원본, 코드, 설정, 환경 및 캐시 자체의 체크섬을 확인한다.
- 기존 배포 테이블은 정확한 파일과 annotation일 때만 `--allow-legacy-cache`로 사용한다.
  이 경로는 새 전처리로 원본을 다시 처리했다고 간주하지 않는다.
- 재학습 가중치·파수축·설정·출처를 공용 NPZ 모델 형식으로 저장하고 읽는다.
- `bundle_model.py`로 raw gate template을 모델에 포함할 수 있다. 묶음 생성 후에는
  새 CSV 예측에 원본 template 폴더가 필요하지 않다.
- `predict.py --model`과 `--output-dir`을 추가했다. 결과는 새 디렉터리에 저장한다.
- 동명 입력을 구분하고, batch 중 일부 파일 실패 시 상태·오류를 표에 남긴다.
- 평가 분모를 실제 개수로 생성하고 빈 집계는 N/A로 표시한다.
- 실행 출처와 버전·체크섬을 manifest에 남기며 README를 갱신했다.

## 실행 예

원본 없이 기존 전처리 테이블로 재학습·기존 가중치 비교:

```bash
python train.py --allow-legacy-cache --compare --output results/model/refit_portable.npz
```

원본 template이 있을 때 한 번 모델 묶음 준비:

```bash
export SERS_DATA=/path/to/data
python bundle_model.py --model results/model/refit_portable.npz --output results/model/reader_bundle.npz
```

Windows PowerShell에서는 환경변수를 다음처럼 설정한다.

```powershell
$env:SERS_DATA = 'D:\path\to\data'
```

묶음 모델로 새 CSV 예측:

```bash
python predict.py path/to/map.csv --model results/model/reader_bundle.npz --output-dir results/predict_runs/run01
```

`run01`은 존재하지 않는 새 디렉터리여야 한다. `--output-dir`을 생략하면 고유 실행 폴더를 만든다.
실제 원본 데이터로 새 전처리 캐시를 만들려면 SERS_DATA를 설정하고 legacy 옵션 없이 학습한다.

## 검증

검증 환경: Windows, Python 3.12.14, NumPy 2.5.1, SciPy 1.18.0, scikit-learn 1.9.0.
저장소 환경 파일의 Python 3.13 환경 자체는 이번 작업에서 생성하지 않았다.

```bash
python -W error::ResourceWarning -m unittest discover -s tests -v
```

자동 검사 16개 통과. 입력 손상·축 불일치·캐시 갱신·포화 pixel·영 스펙트럼·모델 저장과
읽기·template 묶음·원본 없는 합성 map 예측·파일명 충돌·부분 실패·동적 집계를 검사했다.

실제 포함된 학습 테이블(95 map, 9,498 spectrum, 1,290 channel)로 재학습 및 비교:

| 항목 | 기존 모델과의 최대 절대 차이 |
|---|---:|
| 조성 가중치·절편 | 약 4.00 × 10⁻¹⁵ |
| 농도 가중치·절편 | 약 1.11 × 10⁻¹⁵ |
| 학습 pixel의 조성 share | 약 1.85 × 10⁻¹¹ |

이 검사는 포함된 전처리 테이블에서의 재학습 확인이다. 원본 측정 CSV에 새 전처리를
적용한 결과나 새로운 평가 점수를 확인한 것은 아니다.

## 남은 외부 데이터 의존성

원본 Raman map과 raw gate template map은 저장소에 없다. 따라서 실제 측정 자료의 전체
재전처리·평가와 실제 template을 포함한 배포 모델 생성은 아직 수행할 수 없다.
원본의 위치를 SERS_DATA로 지정하면 새 코드 경로로 실행할 수 있다. 합성 데이터로는
template 생성 → 묶음 저장 → 원본 접근 없는 예측까지 검사했다.

구형 .dlm은 pickle이므로 저장소에 포함된 파일의 체크섬을 검증한 후에만 읽는다.
다른 구형 모델을 옮길 때는 그 모델의 실제 기준 파수축을 명시하여 변환해야 한다.
