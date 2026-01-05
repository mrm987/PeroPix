# PeroPix

챗봇 캐릭터 에셋을 빠르고 쉽게 만드는 이미지 생성 앱

## 왜 PeroPix인가?

### 🚀 대량 생산에 최적화
- **멀티 슬롯 시스템**: 감정 표현 20장, H씬 30장을 한 번에 큐에 넣고 자동 생성
- **캐릭터별 프롬프트**: 6개 캐릭터를 각각 다른 프롬프트로 동시 관리
- **프리셋 시스템**: 자주 쓰는 설정을 저장하고 원클릭으로 불러오기

### 🔒 자동 검열로 시간 절약
- 생성된 이미지를 AI가 자동으로 분류
- 검열 필요한 부분만 모아서 한 번에 처리
- 수백 장의 에셋도 빠르게 마무리

### 💡 편리한 프롬프트 작업
- **스마트 태그 자동완성**: 242만 Danbooru 태그 데이터베이스로 원하는 태그를 바로 찾기
- **가중치 색상 표시**: 강조/약화 문법을 실시간으로 시각화
- 메타데이터에서 프롬프트 복원 - 마음에 드는 이미지 설정을 바로 재사용

### ✨ NovelAI 웹 기능 완벽 지원
- Character Reference, Vibe Transfer, Inpaint 등 NAI 웹의 모든 기능 사용 가능
- 설정값과 결과물이 NAI 웹과 100% 동일

## 다운로드

[Releases](https://github.com/mrm987/PeroPix/releases)에서 최신 버전 다운로드

**Windows**: `PeroPix-Windows.zip` 압축 해제 후 `PeroPix.bat` 실행
**macOS**: `PeroPix-macOS.zip` 압축 해제 후 `PeroPix.command` 실행

## 빠른 시작

1. Settings에서 NAI API Token 입력
2. 프롬프트 입력하고 Queue 버튼 클릭
3. 생성된 이미지 확인 및 저장

## 시스템 요구사항

**기본**: Windows 10/11 또는 macOS + NovelAI 구독
**로컬 생성** (선택): NVIDIA GPU 8GB+ VRAM

## 폴더 구조

생성된 파일은 각 폴더에 자동 저장됩니다:
- `gallery/` - 갤러리에 저장한 이미지
- `outputs/` - 생성 직후 이미지
- `presets/` - 생성 설정 프리셋
- `prompts/` - 프롬프트 프리셋

## Credits

- **Danbooru Tag Database** (2026-01-01) - [DraconicDragon/dbr-e621-lists-archive](https://github.com/DraconicDragon/dbr-e621-lists-archive)
- **YOLOv8 Nudenet Detector** - 자동 검열 모델

---

MIT License
