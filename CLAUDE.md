# PeroPix

NovelAI 이미지 생성 데스크톱 클라이언트 (Windows/macOS)

## 프로젝트 구조

```
index.html    - 프론트엔드 (HTML + CSS + JavaScript 단일 파일)
backend.py    - FastAPI 백엔드
```

### 의존성
- `requirements-core.txt` - 릴리즈용 (GitHub Actions 빌드)
- `requirements.txt` - 개발용 (torch/diffusers 포함)

### 빌드
```
.github/workflows/
├── build.yml         # Windows (PeroPix-Windows.zip)
└── build-macos.yml   # macOS (PeroPix-macOS.zip)
```

## NAI API

### 엔드포인트
- 이미지 생성: `POST https://image.novelai.net/ai/generate-image`
- 구독 정보: `GET https://api.novelai.net/user/subscription`
- Vibe 인코딩: `POST https://image.novelai.net/ai/encode-vibe`

### 참고 자료
- **NAIS2**: https://github.com/sunanakgo/NAIS2 (가장 정확한 참고)

### Action 타입
| Action | 용도 | 모델 |
|--------|------|------|
| `generate` | txt2img | 일반 |
| `img2img` | 이미지 기반 생성 | 일반 |
| `infill` | 인페인트 | `-inpainting` 접미사 필요 |

### 인페인트 모델 매핑
```
nai-diffusion-4-5-full → nai-diffusion-4-5-full-inpainting
nai-diffusion-4-5-curated → nai-diffusion-4-5-curated-inpainting
```

### 마스크 형식
- 검정(0) = 유지, 흰색(255) = 인페인트
- 순수 흑백만 (회색 금지)
- RGBA PNG, alpha=255

## Character Reference (V4.5)

**핵심**: 프론트엔드에서 Canvas로 처리해야 NAI 웹과 동일한 결과

```javascript
// Canvas로 리사이즈/패딩 후 JPEG 95% 출력
// 캔버스 크기: 1472x1472 (정사각형), 1536x1024 (가로), 1024x1536 (세로)
```

### API 파라미터
```javascript
director_reference_images: [processedImageBase64],  // JPEG 95%
director_reference_information_extracted: [1.0],
director_reference_strength_values: [1.0],
director_reference_secondary_strength_values: [1.0 - fidelity],  // 반전!
```

**주의**: `fidelity` UI 값이 반전됨 (UI 1.0 → API secondary 0.0)

## Vibe Transfer

```javascript
reference_image_multiple: [encodedVibeData],
reference_information_extracted_multiple: [1.0],
reference_strength_multiple: [0.6]
```

## 메타데이터 필드 매핑

| NAI | 앱 내부 |
|-----|---------|
| `uc` | `negative_prompt` |
| `scale` | `cfg` |
| `noise_schedule` | `scheduler` |
| `request_type` | `nai_model` |
| `sm` / `sm_dyn` | `smea` |

## 주요 변수

### 프론트엔드
```javascript
currentProvider      // 'nai' | 'local'
currentMode          // 'slot' | 'gallery'
vibeList             // Vibe Transfer 목록
charRefData          // { image, processedImage, fidelity, style_aware }
```

### 백엔드 경로
```python
GALLERY_DIR = APP_DIR / "gallery"
OUTPUT_DIR  = APP_DIR / "outputs"
```

## V4 모델 주의사항

- `dynamic_thresholding` - 항상 False
- `uncond_scale` - 항상 1.0
- Quality Tags / UC Preset은 클라이언트에서 직접 프롬프트에 추가
