# PeroPix

> 챗봇 캐릭터 에셋 수백 장, 하루 만에 끝내는 이미지 생성 도구

## 에셋 생성/검열 작업 시간을 획기적으로 줄여보세요

챗봇 제작에서 가장 오래걸리는 부분이 **이미지 생성**이라고 생각하는데요.

특히 캐릭터 별로 감정, H 에셋을 뽑고 검열하는데 상당한 시간이 들죠.

그래서 만들었습니다.

**PeroPix**는 세팅된 감정/H 세트를 한번에 생성하고, 검열도 자동으로 진행합니다.

## 핵심 피쳐

### 슬롯 모드: 대량 생산의 핵심
**무한히 추가 가능한 슬롯 각각 다른 프롬프트를 넣고 한 번에 생성.**

- 슬롯 1: `happy, smile`
- 슬롯 2: `sad, tears`
- 슬롯 3: `angry, shouting`
- ...

각 슬롯에 감정별로 프롬프트를 설정한 후 생성하면...!

하나의 캐릭터의 수십가지 에셋이 한 번에 생성됩니다.

### 검열 모드: 자동 감지 + 원클릭 수정
생성된 수백 장을 일일이 열어볼 필요 없습니다.

폴더에 검열할 이미지 모두 넣고 > [자동 검열] 버튼 누르면 > 검열 완료!

- **원클릭 자동 검열**: YOLO 모델이 이미지를 스캔해서 자동으로 검열 후 저장.
- **간단한 추가 수정**: 전체 리스트를 확인하며 자동으로 감지되지 않은 부분도 편리하게 수정할 수 있습니다.

### NAI + 로컬, 한 앱에서
PeroPix는 NAI와 로컬(Stable Diffusion) 둘 다 지원하고, 탭 전환으로 즉시 전환.

- **NovelAI**: Character Reference, Vibe Transfer, Inpaint 등 웹과 100% 동일한 결과물.
- **로컬 생성**: SDXL + LoRA, 업스케일 지원.

같은 슬롯/검열 워크플로우를 양쪽 다 사용할 수 있습니다.

### 기타 기능
- 태그 자동완성(242만 Danbooru DB).

## 설치

[Releases](https://github.com/mrm987/PeroPix/releases)에서 ZIP 다운로드 → 압축 해제 → 실행 파일 실행

**실행 파일**
- **Windows**: `PeroPix.bat`
- **macOS**: `PeroPix.command`

## 사용법

```
1. Settings → NAI API Token 입력
2. 프롬프트 작성
3. Queue 클릭
```

생성된 파일은 `outputs/` 폴더에 자동 저장됩니다.

## 시스템 요구사항

NovelAI 구독만 있으면 됩니다. (Windows 10/11 또는 macOS)

로컬 생성을 원하면 NVIDIA GPU 12GB 이상이 권장됩니다.

## 크레딧

태그 자동완성: [Danbooru Tag Database](https://github.com/DraconicDragon/dbr-e621-lists-archive) (2026-01-01)

자동 검열: YOLOv8 Nudenet Detector

---

MIT License
