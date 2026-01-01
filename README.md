# PeroPix

NAI API + Local Diffusers 이미지 생성기

## 다운로드

[Releases](https://github.com/mrm987/PeroPix/releases)에서 최신 버전 다운로드

## 설치

1. `PeroPix-Windows.zip` 다운로드
2. 압축 해제
3. `PeroPix.exe` 실행

## 사용법

### NAI 생성
1. Settings → NAI API Token 입력
2. NAI 탭에서 프롬프트 입력
3. Queue 버튼 클릭

### Local 생성 (GPU 필요)
1. Local 탭 클릭 → Install 버튼 (~3GB 추가 다운로드)
2. `models/checkpoints/`에 SDXL 모델 배치
3. 모델 선택 후 Generate

## 주요 기능

### 이미지 생성
- **NAI API** - NovelAI v4/v4.5 모델 지원
- **Local 생성** - SDXL + LoRA 지원
- **2-Pass Upscale** - 업스케일 모델로 고해상도 이미지 생성
- **SMEA / DYN** - NAI 고급 샘플링 옵션

### 멀티 슬롯 시스템
- **슬롯 모드** - 여러 프롬프트를 동시에 큐에 넣어 생성
- **슬롯 너비 조절** - 슬롯 좌우 모서리 드래그로 크기 조절
- **실시간 미리보기** - SSE로 생성 진행상황 및 결과 즉시 표시

### Vibe Transfer
- **바이브 이미지** - 참조 이미지의 스타일/분위기 적용
- **Information Extracted** - 바이브 강도 조절 (0.0 ~ 1.0)
- **Reference Strength** - 레퍼런스 강도 조절 (0.0 ~ 1.0)

### 갤러리
- **갤러리 모드** - 저장된 이미지 브라우징 및 관리
- **이미지 카드** - 썸네일, 시드, 프롬프트/시드/설정 적용 버튼
- **파일명 수정** - 갤러리에서 바로 이미지 이름 변경
- **폴더 열기** - 갤러리 폴더 바로가기
- **메타데이터 적용** - 캐릭터 프롬프트 포함 전체 설정 복원

### 이미지 관리
- **이미지 드롭** - 이미지를 드롭하여 설정 불러오기
- **설정 불러오기** - 미리보기에서 클릭 한 번으로 메타데이터 복원
- **PNG 메타데이터** - 모든 생성 설정이 이미지에 저장됨 (캐릭터 프롬프트 포함)

### 프롬프트 관리
- **Character Prompts** - 캐릭터별 분리 프롬프트 (최대 6개)
- **Negative Prompt** - 제외할 요소 지정
- **Quality Tags** - 품질 태그 자동 추가 옵션
- **UC Preset** - Undesired Content 프리셋
- **프리셋 저장/불러오기** - 프롬프트 프리셋 관리

### UI 커스터마이징
- **사이드바 크기 조절** - 드래그로 설정 패널 너비 조절
- **슬롯 크기 조절** - 드래그로 슬롯 너비 조절 (150px ~ 1200px)
- **설정 저장/복원** - 모든 설정이 자동 저장됨
- **다크 테마** - Deep Navy 컬러 스킴

## 폴더 구조

```
PeroPix/
├── PeroPix.exe
├── backend.py
├── index.html
├── python_env/       # Python 환경 (포함)
├── prompts/          # 프롬프트 프리셋
├── vibes/            # Vibe Transfer 이미지
├── gallery/          # 갤러리 저장 이미지
├── models/
│   ├── checkpoints/  # SDXL 모델 (.safetensors)
│   ├── loras/        # LoRA 파일
│   └── upscale_models/
└── outputs/          # 생성된 이미지
```

## 시스템 요구사항

### NAI 생성
- Windows 10/11
- 인터넷 연결
- NAI 구독 및 API 토큰

### Local 생성
- Windows 10/11
- NVIDIA GPU (CUDA)
- 8GB+ VRAM 권장

## License

MIT
