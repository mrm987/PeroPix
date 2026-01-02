# PeroPix 개발 노트

## 프로젝트 구조
- `index.html` - 프론트엔드 (HTML + CSS + JavaScript 단일 파일)
- `backend.py` - FastAPI 백엔드

---

## NAI (NovelAI) API

### API 문서
- 공식 문서: https://docs.novelai.net/
- 비공식 API 문서: https://api.novelai.net/docs

### 주요 엔드포인트
- 이미지 생성: `POST https://image.novelai.net/ai/generate-image`
- 구독 정보: `GET https://api.novelai.net/user/subscription`

### 구독 Tier
```
0: Paper (Free)
1: Tablet
2: Scroll
3: Opus
```
- Opus 확인: `tier >= 3`

### Anlas 비용 계산

#### Opus 무료 조건 (공식)
- 1024×1024 픽셀 이하 (약 1MP)
- 28 steps 이하
- 단일 이미지 생성
- 다른 이미지를 base로 사용하지 않음 (img2img, inpaint 제외)

#### 기본 비용 공식
```
비용 = ceil(메가픽셀 × 20)
     = ceil(pixels / 1048576 × 20)
```
- Steps 보정: 28 초과시 `base_cost × (steps / 28)`

#### NAI 비용 공식 (역산)
```python
# 기본 비용: ceil(megapixels × 20)
base_cost = math.ceil(pixels / 1048576 * 20)

# Steps 보정 (28 초과시)
if steps > 28:
    base_cost = int(base_cost * (steps / 28))

# 검증 데이터 (Opus, 28 steps)
# 1152×1152 (1.27MP) → 26 Anlas ✓
# 1280×1280 (1.56MP) → 32 Anlas ✓
# 1920×1080 (1.98MP) → 40 Anlas ✓
# 1472×1472 (2.07MP) → 42 Anlas ✓
# 1536×1536 (2.25MP) → 45 Anlas ✓
```

#### Vibe Transfer (V4/V4.5)
- 인코딩: 2 Anlas/vibe (일회성, 캐시됨)
- Information Extracted 값 변경 시 재인코딩 필요
- 4개 초과 시: 추가 vibe당 +2 Anlas
- 최대 16개 사용 가능

#### Character Reference (V4.5 전용)
- +5 Anlas/이미지 (Opus/일반 동일)
- Vibe Transfer와 동시 사용 불가
- 최대 6개 이미지 사용 가능

#### 비용 표시 형식
```
// 일반: 총비용 (개별 × 슬롯 × 횟수)
"15 (5 × 3슬롯 × 1회)"

// Vibe: 인코딩 비용만 표시 (일회성)
"4 (바이브 2개)"
```

### subscription API 응답 구조
```json
{
  "tier": 3,
  "active": true,
  "trainingStepsLeft": {
    "fixedTrainingStepsLeft": 10000,  // 구독 Anlas
    "purchasedTrainingSteps": 5000     // 구매 Anlas
  }
}
```

### Inpaint / Img2Img

#### Action 타입
- `action: "generate"` - 일반 txt2img
- `action: "img2img"` - 이미지 기반 생성 (strength, noise 사용)
- `action: "infill"` - 인페인트 (strength만 사용, noise 없음)

#### 인페인트 전용 모델 (중요!)
V4/V4.5는 인페인트 시 별도 모델 필요. 일반 모델로 `infill` action 사용 시 에러 발생.

```
일반 모델                          → 인페인트 모델
nai-diffusion-4-5-full            → nai-diffusion-4-5-full-inpainting
nai-diffusion-4-5-curated         → nai-diffusion-4-5-curated-inpainting
nai-diffusion-4-full              → nai-diffusion-4-full-inpainting
nai-diffusion-4-curated-preview   → nai-diffusion-4-curated-inpainting
nai-diffusion-3                   → nai-diffusion-3-inpainting
```

#### 마스크 형식
- 검정(black) = 유지할 영역
- 흰색(white) = 인페인트할 영역

#### 참고 자료
- novelai-python SDK: https://github.com/LlmKira/novelai-python
- ComfyUI NAI Generator: https://github.com/bedovyy/ComfyUI_NAIDGenerator

### V4 모델에서 제거된 옵션
- `dynamic_thresholding` (Decrisper) - V4에서 효과 없음, 항상 False
- `uncond_scale` - V4에서 제거됨, 항상 1.0

---

## 해결한 문제들

### 1. JavaScript 변수 중복 선언 오류
**문제**: `images.forEach(img => {...})` 안에서 `const img = ...` 선언시 충돌
**해결**: 내부 변수명을 `imgEl` 등으로 변경

### 2. 이미지 드래그 동작
**문제**: `<img>`의 기본 드래그가 부모 드래그를 방해
**해결**: `imgEl.draggable = false` 설정 (preventDefault 대신)

### 3. CSS overflow와 외부 요소
**문제**: `overflow: hidden` 부모 안에서 `right: -8px` 요소가 잘림
**해결**: 부모를 `overflow: visible`로 변경, 내부 요소에 개별 overflow 설정

### 4. 드롭다운 클리핑
**문제**: `.collapsible-content`의 `overflow: hidden`이 드롭다운 잘림
**해결**: `#charactersContent:not(.collapsed)` 에만 `overflow: visible` 적용

### 5. 큐 진행률 동기화
**문제**: 빠르게 여러번 클릭시 진행률 표시 오류
**해결**:
- `totalImages`와 `currentIndex` 독립적으로 동기화
- 시드는 큐 추가 전에 즉시 갱신 (중복 방지)

### 6. 갤러리 폴더 이동 후 삭제 실패
**문제**: 슬롯에서 갤러리 등록 후 폴더 이동하면 삭제 불가
**해결**: 백엔드에서 폴더 미지정시 전체 폴더 검색

### 7. 슬롯 이미지/정보 동시 표시
**문제**: 이미지보다 하단 info bar가 먼저 표시됨
**해결**: `img.onload` 콜백에서 카드 삽입

---

## 주요 데이터 구조

### 슬롯 이미지 데이터 (card._imageData)
```javascript
{
  image: "base64...",           // 이미지 데이터
  image_path: "/path/to/file",  // 파일 경로
  filename: "image.png",
  metadata: { ... },
  galleryFilename: "saved.png", // 갤러리 저장시
  galleryFolder: "folder"       // 갤러리 폴더
}
```

### Vibe 데이터
```javascript
{
  image: "base64...",
  strength: 0.6,
  info_extracted: 1.0,
  name: "vibe_name"
}
```

### 갤러리 폴더 구조
```
gallery/
├── image1.png          (루트)
├── image2.png
├── 캐릭터A/
│   ├── char1.png
│   └── char2.png
└── 배경/
    └── bg1.png
```

---

## 프론트엔드 주요 변수

```javascript
currentProvider      // 'nai' | 'local'
currentMode          // 'slot' | 'gallery'
currentGalleryFolder // 현재 갤러리 폴더 ('' = 루트)
isOpusTier           // Opus 구독 여부
vibeList             // Vibe Transfer 목록
charRefData          // Character Reference 데이터
```

---

## 백엔드 주요 경로

```python
GALLERY_DIR = APP_DIR / "gallery"
OUTPUT_DIR  = APP_DIR / "outputs"
CONFIG_FILE = APP_DIR / "peropix_config.json"
```

---

## 디버깅 팁

### 콘솔 로그 확인
- `[Generate] Job xxx started - N image(s), WxH, steps`
- `[Generate] Image N/M completed - Xs - filename`
- `[Generate] Job xxx finished - N image(s) in Xs`
- `[WS] Client connected/disconnected`
- `[NAI] Vibe cache hit/miss`

### 일반적인 문제
1. **슬롯 안보임**: JavaScript 오류 → 콘솔 확인
2. **API 실패**: 네트워크 탭에서 요청/응답 확인
3. **스타일 깨짐**: CSS overflow, z-index 확인

---

## 최근 추가된 기능

### Vibe Cache Viewer
- 갤러리 모드에 `vibe` 탭 추가 (gallery 왼쪽, 이중 구분선)
- `vibe_cache` 폴더의 캐시된 바이브 파일 표시
- 버튼: `🎨 Vibe` (바이브 적용), `🗑️` (삭제)
- 적용 시 사전 인코딩된 데이터 사용 (Anlas 무료)

### Wheel Navigation
- 라이트박스에서 마우스 휠로 이전/다음 이미지 탐색
- 슬롯 모드, 갤러리, 바이브 캐시 모두 지원

### 설정 적용 모달 개선
- `전체 적용` / `프롬프트만` 선택 가능
- 프롬프트만: prompt, negative, character prompts, seed만 적용

### Save Options
- 저장 포맷 선택: PNG / JPG / WebP
- JPG Quality 설정
- 메타데이터 제거 옵션

---

## 계획된 기능

### Phase 1: Inpaint / Img2Img

#### 마스크 모달
```
갤러리/슬롯에서 [Inpaint] 버튼
        ↓
┌─ Mask Editor Modal ─────────────┐
│  [Canvas + Mask Layer]          │
│  [Brush] [Eraser] [Clear]       │
│  Size: ━━●━━                    │
│  [Cancel]  [Apply to Generate]  │
└─────────────────────────────────┘
        ↓
Base Image 섹션에 이미지+마스크 설정
        ↓
슬롯 모드에서 생성
```

#### Base Image 설정 (Generation Settings 내)
```javascript
baseImageSettings = {
    enabled: true,
    image: base64,           // 원본 이미지
    mask: base64 | null,     // 마스크 (inpaint용)
    mode: 'inpaint',         // 'img2img' | 'inpaint'
    strength: 0.5,           // 변형 강도
    noise: 0.0               // 노이즈
}
```

#### NAI API 변경사항
- `action: "generate"` → txt2img (현재)
- `action: "img2img"` → 이미지 기반 생성
- `action: "infill"` → 인페인트
- 추가 파라미터: `image`, `mask`, `strength`, `noise`

### Phase 2: Censor Mode

#### 구조
```
[Slot Mode] [Gallery Mode] [Censor Mode]
```

#### 워크플로우
```
1. 폴더 선택 (Source / Output)
2. [Run Auto Censor] → 일괄 처리
3. Review Grid (썸네일 + 상태)
   - ✓ OK / ⚠️ 확인필요
4. 이미지 클릭 → Quick Editor
   - 간단한 도형/브러시 도구
   - [Save & Next]로 빠른 작업
5. [Export All] → 승인된 것만 저장
```

#### Quick Editor 도구
- 사각형 (흰/검/모자이크)
- 브러시
- 이동/크기 조절

---

## API 엔드포인트

### Vibe Cache API
```
GET  /api/vibe-cache              - 캐시 목록
GET  /api/vibe-cache/{filename}   - 상세 정보 (vibe_data 포함)
DELETE /api/vibe-cache/{filename} - 캐시 삭제
```

### 향후 추가 예정
```
POST /api/generate/edit           - img2img / inpaint 생성
POST /api/censor/auto             - 자동 검열 실행
POST /api/censor/save             - 검열 결과 저장
```

---

## Vibe 데이터 구조 (확장)

```javascript
{
  image: "base64...",
  strength: 0.6,
  info_extracted: 1.0,
  name: "vibe_name",
  encoded: "base64..."  // 사전 인코딩된 데이터 (캐시에서 로드 시)
}
```

- `encoded` 필드가 있으면 재인코딩 없이 바로 사용
- Anlas 비용 계산 시 encoded가 있으면 캐시된 것으로 처리
