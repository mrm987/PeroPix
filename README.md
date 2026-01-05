# PeroPix

챗봇 제작을 위한 최적의 이미지 생성 앱.
- 감정/H 에셋 일괄 생성
- 원클릭 자동 검열
- NAI & 로컬 통합 지원 (로컬 지원 예정)

## 다운로드

[Releases](https://github.com/mrm987/PeroPix/releases)에서 최신 버전 다운로드

## 설치

### Windows
1. `PeroPix-Windows.zip` 다운로드 후 압축 해제
2. `PeroPix.bat` 실행

### macOS
1. `PeroPix-macOS.zip` 다운로드 후 압축 해제
2. `PeroPix.command` 실행 (처음 실행 시 우클릭 → 열기)

## 시작하기

1. Settings에서 NAI API Token 입력
2. 프롬프트 입력
3. Queue 버튼 클릭

## 주요 기능

### 이미지 생성
- NovelAI v4/v4.5 모델 지원
- 멀티 슬롯 시스템 - 여러 이미지를 동시에 큐에 넣어 생성
- 실시간 생성 진행 상황 표시
- SMEA / Variety+ 옵션

### Character Reference
- 참조 이미지로 캐릭터 일관성 유지
- Fidelity / Style Aware 조절
- NAI 웹과 동일한 결과

### Vibe Transfer
- 참조 이미지의 스타일/분위기 적용
- Information Extracted / Strength 조절
- 바이브 캐시로 Anlas 절약

### Inpaint / Img2Img
- 이미지 일부 수정 (인페인트)
- 이미지 기반 생성 (img2img)
- 마스크 에디터 내장

### 프롬프트 관리
- **Character Prompts** - 캐릭터별 분리 프롬프트 (최대 6개)
- **Quality Tags / UC Preset** 자동 적용
- **프롬프트 프리셋** 저장/불러오기
- **가중치 하이라이팅** - NAI/Local 문법 실시간 색상 표시
  - NAI: `{강화}`, `[약화]`, `1.2::태그::`
  - Local: `(강화)`, `[약화]`, `(태그:1.5)`
  - 가중치 수준에 따라 골드/블루 색상 강조
- **스마트 태그 자동완성**
  - 242만+ Danbooru 태그 데이터베이스
  - 스페이스 입력 시 자동으로 언더바로 변환 검색
  - 연속 스페이스 2번 입력으로 자동완성 종료

### 갤러리
- 생성된 이미지 브라우징 및 관리
- 폴더 분류
- 메타데이터에서 설정 복원
- 다양한 저장 포맷 (PNG/JPG/WebP)

### Local 생성 (선택)
- SDXL + LoRA 지원 (Windows, NVIDIA GPU 필요)
- 첫 실행 시 Install 버튼으로 설치

## 폴더 구조

```
PeroPix/
├── gallery/          # 저장된 이미지
├── outputs/          # 생성된 이미지
├── prompts/          # 프롬프트 프리셋
├── presets/          # 생성 프리셋
├── vibe_cache/       # 바이브 캐시
└── models/           # Local 생성용 모델
    ├── checkpoints/
    └── loras/
```

## 시스템 요구사항

### NAI 생성
- Windows 10/11 또는 macOS
- 인터넷 연결
- NovelAI 구독 및 API 토큰

### Local 생성
- Windows 10/11
- NVIDIA GPU (8GB+ VRAM 권장)

## Credits

### Tag Autocomplete
- **Danbooru Tag Database** (2026-01-01)
  - Source: [DraconicDragon/dbr-e621-lists-archive](https://github.com/DraconicDragon/dbr-e621-lists-archive)
  - License: Public Domain
  - 242만+ 태그 데이터로 프롬프트 입력 시 자동완성 지원

### Censoring Model
- **YOLOv8 Nudenet Detector**
  - Model: `yolov8x-seg_nudenet_dataset-detector_best.pt`
  - Purpose: 자동 이미지 검열 기능
  - NAI 생성 이미지 자동 탐지 및 모자이크 처리

## License

MIT
