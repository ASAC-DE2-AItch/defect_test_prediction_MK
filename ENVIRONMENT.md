# 가상환경 구축 & 사용 가이드

> SK Hynix Defect Test Prediction 프로젝트 실행 환경 정리
> (Windows / OneDrive 한글 경로 기준)

---

## 1. 사전 요구사항

| 항목 | 내용 |
|------|------|
| OS | Windows (PowerShell 또는 cmd) |
| Python | **3.10 ~ 3.12 권장** (⚠️ 3.13/3.14는 lightgbm·torch wheel 부재/커널 오류 위험 → 사용 금지) |
| 경로 | OneDrive 한글 경로 사용 시 **Windows Long Path 활성화 필수** |
| 데이터 | `문제1(하)/`, `문제1_하_answer/` 폴더가 프로젝트 루트에 있어야 함 |

### Windows Long Path 활성화 (한글·긴 경로 오류 방지)

관리자 PowerShell에서 1회 실행:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

---

## 2. 가상환경 구축

### 2.1 생성 & 활성화

```powershell
# 프로젝트 루트에서 — Python 버전을 명시해 생성 (3.11 권장)
py -0                       # 설치된 Python 버전 확인
py -3.11 -m venv venv       # 3.11 없으면 -3.12 또는 -3.10

# 활성화 (PowerShell)
.\venv\Scripts\Activate.ps1
# 활성화 (cmd)
venv\Scripts\activate

# 버전 확인 (3.10~3.12 인지 반드시 확인)
python --version
```

> ⚠️ `python -m venv venv`로 만들면 시스템 기본 파이썬(3.14 등 최신)이 잡힐 수 있습니다. 반드시 `py -3.11`처럼 **버전을 지정**하세요.

활성화되면 프롬프트 앞에 `(venv)`가 표시됩니다.

### 2.2 패키지 설치

> ⚠️ **회사 보안정책 주의**: `pip.exe` 직접 실행이 차단될 수 있습니다.
> 반드시 **`python -m pip`** 형태로 실행하세요.

```powershell
python -m pip install -r requirements.txt --break-system-packages
```

`requirements.txt` 포함 패키지: pandas, numpy, scipy, scikit-learn, lightgbm, matplotlib, seaborn, jupyter, optuna, torch.

### 2.3 PyTorch (v6/v7 시퀀스·딥러닝 노트북 전용)

`requirements.txt`에 `torch`가 포함되어 있지만, GPU 사용 시 CUDA 빌드를 별도로 설치하는 것이 좋습니다.

```powershell
# CPU 버전 (기본, 소규모라 CPU로도 수 분 내 학습)
python -m pip install torch --break-system-packages

# GPU(CUDA) 버전은 https://pytorch.org 에서 환경에 맞는 명령 확인
```

> v1~v5, v7(LightGBM)만 쓸 경우 torch는 없어도 됩니다.

---

## 3. Jupyter 커널 등록

노트북(.ipynb)을 venv 환경에서 실행하려면 커널로 등록해야 합니다.

```powershell
python -m ipykernel install --user --name venv --display-name "venv"
```

등록 후 VS Code 또는 Jupyter에서 커널을 **"venv"**로 선택합니다.

---

## 4. 사용 방법

### 4.1 노트북 실행 (권장)

VS Code에서 `.ipynb`를 열고 커널을 `venv`로 선택 → 위에서부터 셀 실행(Shift + Enter).

| 노트북 | 위치 | 데이터 경로 | 비고 |
|--------|------|-------------|------|
| `modeling_baseline.ipynb` | 루트 | `문제1(하)/` | 기준 모델(v1 정리본) |
| `modeling_v2~v7.ipynb` | 각 서브폴더 | `../문제1(하)/` | **서브폴더 안에서 실행** (상대경로) |
| `EDA/01_EDA.ipynb` | `EDA/` | `../문제1(하)/` | 탐색적 분석 |

> ⚠️ v2~v7 노트북은 반드시 **해당 서브폴더 안에서** 실행해야 상대경로(`../문제1(하)`)가 맞습니다. 결과는 각 폴더의 `outputs/`에 저장됩니다.

### 4.2 CLI 파이프라인 (train.py)

baseline과 동일한 파이프라인을 커맨드라인으로 실행:

```powershell
python train.py                        # 기본 파라미터 학습
python train.py --tune                 # Optuna 100 trials 튜닝 후 학습
python train.py --tune --n_trials 200  # trial 수 지정
```

출력: `outputs/results.json`, `outputs/valid_Y_submit.csv`, `outputs/test_Y_submit.csv`

---

## 5. 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `pip`이 실행 안 됨 | 회사 보안정책이 pip.exe 차단 | `python -m pip ...` 사용 |
| 패키지 설치 권한 오류 | 시스템 파이썬 보호 | `--break-system-packages` 플래그 추가 |
| 한글·긴 경로 파일 오류 | Long Path 미활성 | 1장 Long Path 활성화 |
| Jupyter 커널이 venv를 못 찾음 | 커널 미등록 | 3장 ipykernel 등록 |
| 커널이 계속 죽거나 멈춤 | jupyter 프로세스 충돌 | `taskkill /F /IM jupyter* /T` 후 VS Code 재시작 |
| `Failed to start the Kernel ... listen EFAULT: bad address :::9001` | 커널 소켓 실행 실패 (Python 버전 불일치/pyzmq) | ① `taskkill /F /IM jupyter* /T` 후 VS Code 재시작 → ② venv를 3.11/3.12로 **재생성** → ③ `python -m pip install --upgrade ipykernel pyzmq --break-system-packages` |
| 커널 이름이 Python 3.13/3.14 | venv가 최신 파이썬으로 생성됨 | `py -3.11 -m venv venv`로 재생성 후 커널 재등록 |
| v2~v7에서 데이터 못 찾음 | 루트에서 실행함 | 해당 서브폴더 안에서 실행 |
| `No module named torch` | torch 미설치 | 2.3장 torch 설치 (v6/v7만 필요) |

---

## 6. 요약 (빠른 시작)

```powershell
# 1) 가상환경
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2) 패키지
python -m pip install -r requirements.txt --break-system-packages

# 3) Jupyter 커널 등록
python -m ipykernel install --user --name venv --display-name "venv"

# 4) 실행 — 노트북(커널 venv 선택) 또는
python train.py --tune
```
