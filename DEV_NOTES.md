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
- RGBA PNG, alpha=255 (완전 불투명)
- 반드시 순수 흑백 (0 또는 255만, 회색 금지)
- 8x8 픽셀 그리드 기반 브러시 (NAI 웹과 동일)

#### NAI 웹 인페인트 파라미터 (2025-01 캡처)
```json
{
  "action": "infill",
  "model": "nai-diffusion-4-5-full-inpainting",
  "parameters": {
    "add_original_image": false,
    "image_format": "png",
    "inpaintImg2ImgStrength": 1,
    "legacy": false,
    "legacy_v3_extend": false,
    "noise": 0,
    "strength": 0.7,
    "image": "<base64 PNG>",
    "mask": "<base64 PNG, 순수 흑백>"
  }
}
```

**주의사항:**
- `add_original_image: false` (true로 하면 seam 발생)
- `inpaintImg2ImgStrength: 1` (고정값)
- `noise: 0` (삭제가 아니라 0으로 설정)
- `img2img` 중첩 객체 **사용 안함**
- 바이브/캐릭터레퍼런스는 인페인트에서 **미지원** (UI에만 표시, 실제 적용 안됨)

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

### 8. NAI 웹과 바이브 생성 결과 불일치
**문제**: 동일한 설정/바이브로 생성해도 NAI 웹과 PeroPix 결과물이 다름
**원인**: `/ai/encode-vibe` API payload 구조가 NAI 웹과 달랐음
- NAI 웹: `{ image, information_extracted, model }` (최상위 레벨)
- PeroPix: `{ image, model, parameters: { information_extracted } }` (잘못된 구조)
**해결**:
1. payload 구조를 NAI 웹과 동일하게 수정
2. RGBA PNG 이미지도 원본 그대로 유지 (불필요한 RGB 변환 제거)
```python
# encode-vibe payload (NAI 웹과 동일)
payload = {
    "image": image_base64,
    "information_extracted": info_val,
    "model": model
}
```

### 9. Quality Tags / UC Preset이 결과에 영향 없음
**문제**: `qualityToggle`과 `ucPreset` 옵션을 켜거나 꺼도 생성 결과가 동일함
**원인**: NAI 서버가 V4.5에서 이 파라미터를 처리하지 않음. NAI 웹은 클라이언트에서 직접 태그를 추가함.
**해결**: 클라이언트에서 직접 프롬프트에 태그 추가

```python
# V4.5 Quality Tags (프롬프트 끝에 추가)
V45_QUALITY_TAGS = ", very aesthetic, masterpiece, no text"

# V4.5 UC Presets (네거티브 프롬프트 앞에 추가)
V45_UC_PRESETS = {
    "Heavy": "nsfw, lowres, artistic error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, dithering, halftone, screentone, multiple views, logo, too many watermarks, negative space, blank page",
    "Light": "nsfw, lowres, artistic error, scan artifacts, worst quality, bad quality, jpeg artifacts, multiple views, very displeasing, too many watermarks, negative space, blank page",
    "Furry Focus": "nsfw, {worst quality}, distracting watermark, unfinished, bad quality, ...",
    "Human Focus": "... + @_@, mismatched pupils, glowing eyes, bad anatomy",
}
```

**참고**:
- `ucPreset`과 `qualityToggle` 파라미터는 여전히 NAI API에 전송됨 (메타데이터용)
- NAI 이미지 임포트 시 중복 방지를 위해 기존 태그를 자동으로 제거함 (`normalizeMetadata`)
- 순수 NAI 이미지 임포트 시 경고 메시지 표시

### 10. 인페인트 회색 경계선 아티팩트
**문제**: 인페인트 마스크 경계에 회색 선이 생김
**원인**: 마스크 에디터 캔버스가 **축소된 디스플레이 크기**로 그려진 후, Apply 시 원본 크기로 **업스케일**됨. 이 과정에서 미세한 보간 아티팩트 발생.

```
문제 상황:
1. 원본 이미지: 1216×832
2. 디스플레이 축소: 800×548 (화면에 맞춤)
3. 캔버스 크기: 800×548 (축소된 크기로 설정됨) ← 문제!
4. 마스크 그리기: 800×548 해상도로 그림
5. Apply: 1216×832로 업스케일 ← 아티팩트 발생!
```

**해결**:
1. 캔버스 실제 크기 = 원본 이미지 크기 (1216×832)
2. CSS `style.width/height`로 디스플레이만 축소 (800×548)
3. 마우스 좌표를 CSS 스케일 비율로 보정
4. Apply 시 업스케일 불필요 (이미 원본 크기)

```javascript
// 캔버스 초기화
canvas.width = originalWidth;      // 실제 해상도 = 원본
canvas.height = originalHeight;
canvas.style.width = displayWidth + 'px';   // CSS로 축소 표시
canvas.style.height = displayHeight + 'px';

// 마우스 좌표 변환 (디스플레이 → 캔버스)
const canvasX = displayX * (canvas.width / rect.width);
const canvasY = displayY * (canvas.height / rect.height);
```

---

## 주요 데이터 구조

### 통합 이미지 메타데이터 (PNG Comment 필드)

NAI 웹과 100% 호환되는 형식. PeroPix 전용 설정은 `peropix` 확장 필드에 저장.

```json
{
  // === NAI 표준 필드 ===
  "prompt": "1girl, ...",
  "uc": "lowres, bad anatomy, ...",     // negative_prompt
  "steps": 28,
  "width": 1216,
  "height": 832,
  "scale": 5.0,                          // cfg
  "seed": 123456789,
  "sampler": "k_euler_ancestral",
  "noise_schedule": "karras",            // scheduler
  "sm": false,                           // SMEA
  "sm_dyn": false,                       // SMEA+DYN
  "ucPreset": 0,                         // uc_preset (0=Heavy, 1=Light, 2=Human Focus, 3=None)
  "qualityToggle": true,                 // quality_tags
  "cfg_rescale": 0.0,
  "request_type": "nai-diffusion-4-5-full",  // nai_model
  "v4_prompt": {...},                    // V4 캐릭터 프롬프트 구조
  "v4_negative_prompt": {...},

  // === PeroPix 확장 필드 ===
  "peropix": {
    "version": 1,
    "provider": "nai",                   // 'nai' | 'local'
    "character_prompts": ["girl, ..."],  // 캐릭터별 프롬프트
    "variety_plus": false,
    "furry_mode": false,
    "local_model": "",                   // local provider용 모델명
    "vibe_transfer": [                   // 바이브 설정 (이미지 제외)
      {"strength": 0.6, "info_extracted": 1.0, "name": "vibe_name"}
    ]
  }
}
```

#### 필드 매핑 (NAI ↔ 앱 내부)
| NAI 필드 | 앱 내부 필드 |
|----------|-------------|
| `uc` | `negative_prompt` |
| `scale` | `cfg` |
| `noise_schedule` | `scheduler` |
| `request_type` | `nai_model` |
| `ucPreset` | `uc_preset` |
| `qualityToggle` | `quality_tags` |
| `sm` / `sm_dyn` | `smea` |

#### normalizeMetadata() 함수
NAI 형식 메타데이터를 앱 내부 형식으로 변환하는 중앙화된 함수.
모든 메타데이터 표시/적용 시 이 함수를 통해 정규화해야 함.
```javascript
// 사용 예시
const normalized = normalizeMetadata(naiMetadata);
applyMetadataSettings(normalized);
```

#### 메타데이터 복원 가능 여부
| 설정 | NAI 원본 | PeroPix 생성 | 비고 |
|------|:--------:|:------------:|------|
| 프롬프트/네거티브 | ✓ | ✓ | |
| 시드/크기/스텝/CFG | ✓ | ✓ | |
| 샘플러/스케줄러 | ✓ | ✓ | |
| SMEA/Variety+/Furry | ✓ | ✓ | |
| UC Preset/Quality Tags | ✓ | ✓ | |
| 캐릭터 프롬프트 | ✓ | ✓ | v4_prompt에서 추출 |
| 모델명 | △ | ✓ | NAI는 내부 타입명일 수 있음 |
| **바이브 설정** | ✗ | △ | 캐시에서 이름 매칭 필요 |
| **캐릭터 레퍼런스** | ✗ | ✗ | 이미지 데이터 필요 |
| **베이스 이미지/마스크** | ✗ | ✗ | 이미지 데이터 필요 |

**원칙**: 이미지 데이터가 필요한 설정은 메타데이터에 저장/복원 불가 (파일 크기)

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
