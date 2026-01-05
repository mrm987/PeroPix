# PeroPix

> 챗봇 캐릭터 에셋 200장, 하루 만에 끝내는 이미지 생성 도구

## 당신의 시간은 비쌉니다

NovelAI 웹에서 감정 표현 20장 만드는 데 몇 시간 걸리셨나요?
프롬프트 복붙하고, 생성 누르고, 다운로드하고, 또 복붙하고...
그러다 H씬 100장 만들어야 한다는 걸 깨달았을 때의 그 막막함.

**PeroPix는 그 반복을 없앱니다.**

## 무엇이 다른가

### 슬롯 모드: 대량 생산의 핵심
NAI 웹은 한 번에 하나씩 생성하고 다운로드.
PeroPix는 **슬롯 8개에 각각 다른 프롬프트를 넣고 Queue 한 번**.

- 슬롯 1: `happy, smile`
- 슬롯 2: `sad, tears`
- 슬롯 3: `angry, shouting`
- ...
- 슬롯 8: `sleeping, zzz`

각 슬롯마다 생성 횟수 설정 가능. 감정 표현 8종류 × 각 5장 = 40장을 한 번에 큐에 넣습니다.
아침에 큐 채워놓고 출근하면, 퇴근할 땐 200장이 폴더에 있습니다.

프리셋 저장으로 "감정팩", "H씬팩" 같은 설정을 원클릭 불러오기.

### 검열 모드: 자동 감지 + 원클릭 수정
생성된 수백 장을 일일이 열어볼 필요 없습니다.

1. **자동 감지**: YOLO 모델이 이미지를 스캔해서 `uncensored/` 폴더로 분류
2. **한눈에 확인**: 검열 필요한 이미지만 모아서 보여줌
3. **원클릭 모자이크**: 클릭 한 번으로 자동 모자이크 → `censored/` 폴더로 이동

AI 감지가 완벽하지 않아도 괜찮습니다.
잘못 분류된 건 버튼 클릭 한 번으로 폴더 이동.
수백 장 작업해도 5분이면 끝납니다.

### NAI + 로컬, 한 앱에서
대부분의 툴은 둘 중 하나만 지원합니다.
PeroPix는 NAI와 로컬(Stable Diffusion) 둘 다 지원하고, 탭 전환으로 즉시 전환.

- **NovelAI**: Character Reference, Vibe Transfer, Inpaint 등 웹과 100% 동일
- **로컬 생성**: SDXL + LoRA, NVIDIA GPU에서 무제한 생성 (선택 사항)

Anlas 아끼고 싶으면 로컬로, 퀄리티 원하면 NAI로.
같은 슬롯/검열 워크플로우를 양쪽 다 사용할 수 있습니다.

태그 자동완성(242만 Danbooru DB)도 있습니다.

## 설치

[Releases](https://github.com/mrm987/PeroPix/releases)에서 ZIP 다운로드 → 압축 해제 → 실행

- **Windows**: `PeroPix.bat`
- **macOS**: `PeroPix.command`

첫 실행 시 자동으로 환경 설정됩니다.

## 사용법

```
1. Settings → NAI API Token 입력
2. 프롬프트 작성
3. Queue 클릭
```

생성된 파일은 `outputs/` 폴더에 자동 저장됩니다.

## 시스템 요구사항

NovelAI 구독만 있으면 됩니다. (Windows 10/11 또는 macOS)
로컬 생성을 원하면 NVIDIA GPU 8GB 이상 필요.

## 크레딧

태그 자동완성: [Danbooru Tag Database](https://github.com/DraconicDragon/dbr-e621-lists-archive) (2026-01-01)
자동 검열: YOLOv8 Nudenet Detector

---

MIT License
