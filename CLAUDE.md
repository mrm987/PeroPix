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

---

## 개발 방법론 교훈

### 1. 포팅할 때는 원본과 1:1 비교부터
다른 프로젝트 코드를 포팅할 때:
- **먼저 원본이 정상 동작하는지 확인**
- 원본과 포팅 코드의 **중간값을 단계별로 비교**
- "대충 비슷하겠지" 추측 금지 → 실제 텐서 값 출력해서 검증

### 2. 성능 문제는 측정 먼저
"느리다"는 증상만으로 원인 추측하지 말 것:
- 각 단계별 소요 시간 측정 (CLIP, UNet step, VAE 등)
- VRAM 사용량 변화 추적
- GPU 사용률 vs VRAM 사용량 동시 확인

### 3. 가설 목록 만들고 하나씩 검증
문제 원인이 불명확할 때:
- 가능한 원인들을 **모두 나열**
- 영향도/검증 난이도로 우선순위 정렬
- **한 번에 하나씩만** 변경하며 테스트

### 4. 디버그 코드가 문제를 만들 수 있다
조사를 위해 추가한 코드가 오히려 증상을 악화시킬 수 있음:
- print()가 GPU 동기화 유발 → 18배 느려짐
- 로깅 추가 후 "더 느려졌다"면 로깅 자체가 원인

### 5. 키/이름 매핑 문제는 양쪽 실제 값 출력
LoRA 적용 안됨, state_dict 로드 실패 등:
- 한쪽 키만 보지 말고 **양쪽 키를 모두 출력**해서 비교
- "매칭될 것 같은데"가 아니라 **실제로 매칭되는지** 확인

### 6. 원본 프로젝트의 "왜"를 이해하기
ComfyUI가 왜 이렇게 구현했는지 이해 없이 복사하면:
- 필요 없는 코드까지 가져오거나
- 핵심 로직을 빠뜨림
- 특히 키 변환, 스케일링 같은 미묘한 차이 놓침
