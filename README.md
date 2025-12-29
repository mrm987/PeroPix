# PeroPix

NAI API + Local Diffusers 이미지 생성기 (Tauri 앱)

## 빠른 시작 (개발 모드)

Tauri 빌드 없이 바로 사용하려면:

```bash
# 1. Python 의존성 설치
pip install -r requirements.txt

# 2. 백엔드 실행
python backend.py

# 3. index.html 더블클릭 또는 브라우저에서 열기
```

## Tauri 앱 빌드

### 필수 설치

**Windows:**
```powershell
# Rust 설치
winget install Rustlang.Rust.MSVC

# Node.js 설치
winget install OpenJS.NodeJS.LTS

# Visual Studio Build Tools (C++ 포함)
winget install Microsoft.VisualStudio.2022.BuildTools
```

**Mac:**
```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Node.js
brew install node

# Xcode Command Line Tools
xcode-select --install
```

**Linux:**
```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Node.js
sudo apt install nodejs npm

# 의존성
sudo apt install libwebkit2gtk-4.0-dev build-essential curl wget libssl-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev
```

### Python 백엔드 패키징

Tauri에서 Python을 sidecar로 포함하려면 PyInstaller로 빌드:

```bash
# PyInstaller 설치
pip install pyinstaller

# 빌드 (Windows)
pyinstaller --onefile --name backend backend.py
# → dist/backend.exe 생성

# 빌드 (Mac/Linux)
pyinstaller --onefile --name backend backend.py
# → dist/backend 생성

# Tauri가 인식할 위치로 복사
# Windows:
copy dist\backend.exe src-tauri\backend-x86_64-pc-windows-msvc.exe

# Mac (Intel):
cp dist/backend src-tauri/backend-x86_64-apple-darwin

# Mac (Apple Silicon):
cp dist/backend src-tauri/backend-aarch64-apple-darwin

# Linux:
cp dist/backend src-tauri/backend-x86_64-unknown-linux-gnu
```

### Tauri 빌드

```bash
# 의존성 설치
npm install

# 개발 모드 (핫 리로드)
npm run dev

# 프로덕션 빌드
npm run build
# → src-tauri/target/release/bundle/ 에 설치 파일 생성
```

## 폴더 구조

```
nai_generator_app/
├── index.html          # UI
├── backend.py          # Python 백엔드
├── config.json         # 설정 (자동 생성)
├── models/             # (자동 생성)
│   ├── checkpoints/    # SD 모델 (.safetensors, .ckpt)
│   └── loras/          # LoRA 파일 (.safetensors)
├── outputs/            # 생성 이미지 (자동 생성)
├── src-tauri/          # Tauri 앱 설정 (빌드용)
├── requirements.txt
├── run_backend.bat     # Windows 실행
├── run_backend.sh      # Mac/Linux 실행
└── README.md
```

## 기능

### NAI
- V4.5 Full / V4 Curated / V3 모델
- UC Preset (Heavy/Light/Human Focus/None)
- Quality Tags 자동 추가
- SMEA/SMEA+DYN

### Local
- SD 1.5 / SDXL 자동 감지
- 다중 LoRA (스케일 조절)
- 모델 캐싱 (첫 로드만 느림)

### 공통
- 멀티 프롬프트 대량 생성
- 실시간 스트리밍 프리뷰
- 이미지별 랜덤 시드
- 라이트박스 뷰어
- 설정 저장/불러오기

## 단축키

| 키 | 동작 |
|---|------|
| Ctrl+Enter | 생성 |
| Esc | 라이트박스/설정 닫기 |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | 헬스체크 |
| `/api/config` | GET/POST | 설정 조회/저장 |
| `/api/models` | GET | 모델 목록 |
| `/api/loras` | GET | LoRA 목록 |
| `/api/generate` | POST | 단일 생성 |
| `/api/generate/multi` | POST | 멀티 생성 (SSE) |
| `/api/cache/clear` | POST | 모델 캐시 클리어 |
| `/api/status` | GET | 상태 확인 |

## 트러블슈팅

### "Backend not connected"
- Python 백엔드가 실행 중인지 확인
- 포트 8765가 사용 중인지 확인: `netstat -an | findstr 8765`

### NAI 토큰 설정
1. https://novelai.net 로그인
2. 우측 상단 → Account Settings
3. Get Persistent API Token 클릭
4. 앱 Settings에서 토큰 입력

### CUDA 오류 (Local)
- PyTorch CUDA 버전 확인: `python -c "import torch; print(torch.cuda.is_available())"`
- GPU 드라이버 업데이트
- CUDA 11.8+ 필요

### 모델 로드 실패
- 모델 파일이 손상되지 않았는지 확인
- SDXL 모델은 VRAM 8GB 이상 필요
