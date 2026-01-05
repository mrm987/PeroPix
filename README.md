# PeroPix

> 챗봇 캐릭터 에셋 200장, 하루 만에 끝내는 이미지 생성 도구

## 당신의 시간은 비쌉니다

NovelAI 웹에서 감정 표현 20장 만드는 데 몇 시간 걸리셨나요?
프롬프트 복붙하고, 생성 누르고, 다운로드하고, 또 복붙하고...
그러다 H씬 100장 만들어야 한다는 걸 깨달았을 때의 그 막막함.

**PeroPix는 그 반복을 없앱니다.**

## 무엇이 다른가

### 큐 시스템으로 대량 생산
NAI 웹은 한 번에 하나씩. PeroPix는 슬롯 8개에 프롬프트 넣고 Queue 누르면 끝.
아침에 큐 채워놓고 출근하면, 퇴근할 땐 200장이 폴더에 있습니다.

### 6개 캐릭터를 동시에
메인 히로인, 서브 캐릭터들 각자 다른 프롬프트.
탭 전환 없이 한 화면에서 전부 관리하고, 한 번에 생성 시작.

### AI가 검열합니다
생성된 수백 장을 일일이 확인할 필요 없습니다.
YOLO 모델이 자동으로 분류해서, 검열 필요한 것만 모아줍니다.
나머지는 그냥 쓰면 됩니다.

### 프롬프트 작업이 편합니다
- 242만 Danbooru 태그 자동완성: `blond` 입력하면 `blonde_hair` 바로 추천
- 가중치 실시간 색상 표시: `{강조}` 골드색, `[약화]` 블루색으로 한눈에
- 메타데이터 원클릭 복원: 마음에 드는 이미지 클릭하면 설정 전부 불러오기

### NovelAI 웹과 100% 동일
Character Reference, Vibe Transfer, Inpaint 전부 지원.
같은 프롬프트 넣으면 같은 결과 나옵니다.

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
