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

## 코딩 원칙

### 코드 변경 전 필수
1. **영역 파악 먼저** - 관련 파일 읽고, 연결 관계 추적, 유사 기능 존재 여부 확인
2. **"이게 뭘 망가뜨릴 수 있지?"** - 확신 없으면 조사 먼저

### 구현 시
- **작성 전 검색** - 기존 함수/유틸리티/패턴 있는지 확인, 없을 때만 새로 작성
- **중복 금지** - 비슷한 로직 두 번 쓰게 되면 → 공유 함수로 추출

### 디버깅 시 (추측 금지, 조사 먼저)
1. **실제 흐름 추적** - 버그로 이어지는 코드 경로 읽기, 로깅으로 실제 동작 확인
2. **참조 문서 확인** - 라이브러리 문제면 공식 문서, API 에러면 실제 응답 메시지
3. **그 다음에 가설** - 감이 아닌 증거 기반으로

### 위험 신호 (멈추고 재고)
- 복붙하려 함 → 함수로 추출
- 왜 되는지 모르겠는데 고쳐짐 → 더 깊이 조사
- 임시 우회책 추가하려 함 → 제대로 된 해결책 먼저 고려
- 변경 내용 설명 못함 → 커밋 금지

---

## 프로젝트별 교훈

### 포팅 작업
- **원본 정상 동작 먼저 확인**
- 원본과 포팅 코드의 **중간값 단계별 비교**
- "대충 비슷하겠지" 추측 금지 → 실제 값 출력해서 검증

### 성능 문제
- "느리다"는 증상만으로 추측하지 말 것
- 각 단계별 소요 시간 측정 (CLIP, UNet step, VAE 등)
- print()가 GPU 동기화 유발 → 18배 느려질 수 있음

### 키/이름 매핑 문제
- 한쪽 키만 보지 말고 **양쪽 키를 모두 출력**해서 비교
- "매칭될 것 같은데"가 아니라 **실제로 매칭되는지** 확인

### 외부 코드 참조
- ComfyUI 등 원본 프로젝트의 **"왜"를 이해**하고 가져올 것
- 이해 없이 복사하면 핵심 로직 빠뜨리거나 불필요한 코드 포함됨
