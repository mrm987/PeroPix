# PeroPix 개발 노트

## 프로젝트 구조

```
index.html    - 프론트엔드 (HTML + CSS + JavaScript 단일 파일)
backend.py    - FastAPI 백엔드
```

### 의존성
- `requirements-core.txt` - 릴리즈용 (GitHub Actions 빌드)
- `requirements.txt` - 개발용 (torch/diffusers 포함, 런타임 자동 설치)

### 빌드
```
.github/workflows/
├── build.yml         # Windows (PeroPix-Windows.zip)
└── build-macos.yml   # macOS (PeroPix-macOS.zip)
```

---

## NAI API 레퍼런스

### 엔드포인트
- 이미지 생성: `POST https://image.novelai.net/ai/generate-image`
- 구독 정보: `GET https://api.novelai.net/user/subscription`

### 참고 자료
- **NAIS2**: https://github.com/sunanakgo/NAIS2 (가장 정확한 참고)
- 비공식 API 문서: https://api.novelai.net/docs

### 구독 Tier
| Tier | 이름 |
|------|------|
| 0 | Paper (Free) |
| 1 | Tablet |
| 2 | Scroll |
| 3 | Opus |

### Anlas 비용
```python
# 기본 비용
base_cost = ceil(pixels / 1048576 * 20)

# Steps 보정 (28 초과시)
if steps > 28:
    base_cost = int(base_cost * (steps / 28))
```

**Opus 무료 조건**: 1024×1024 이하, 28 steps 이하, 단일 이미지, txt2img만

**추가 비용**:
- Vibe Transfer: 2 Anlas/vibe (인코딩, 캐시됨)
- Character Reference: +5 Anlas/이미지

---

## Action 타입

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

---

## Character Reference (V4.5)

**핵심**: 프론트엔드에서 브라우저 Canvas로 처리해야 NAI 웹과 동일한 결과

### 프론트엔드 처리 (NAIS2/NAI 웹 방식)
```javascript
// Canvas로 리사이즈/패딩 후 JPEG 95% 출력
function processCharacterReferenceImage(base64Image) {
    // 캔버스 크기: 1472x1472 (정사각형), 1536x1024 (가로), 1024x1536 (세로)
    const canvas = document.createElement('canvas');
    ctx.fillStyle = '#000000';  // 검은 배경 (letterbox)
    ctx.fillRect(0, 0, targetW, targetH);
    ctx.drawImage(img, x, y, w, h);  // 비율 유지 리사이즈
    return canvas.toDataURL('image/jpeg', 0.95).split(',')[1];
}
```

### API 파라미터
```javascript
director_reference_images: [processedImageBase64],  // JPEG 95%
director_reference_information_extracted: [1.0],
director_reference_strength_values: [1.0],
director_reference_secondary_strength_values: [1.0 - fidelity],  // 반전!
director_reference_descriptions: [{
    caption: { base_caption: "character&style", char_captions: [] },
    legacy_uc: false
}]
```

**주의사항**:
- `fidelity` UI 값이 반전됨: UI 1.0 → API secondary 0.0
- `director_reference_images` 사용 (캐시 구조 아님)
- 이미지 형식: **JPEG 95%** (PNG 아님!)
- 인페인트에서는 미지원

---

## Vibe Transfer

### encode-vibe API
```python
payload = {
    "image": image_base64,
    "information_extracted": info_val,
    "model": model
}
# POST https://image.novelai.net/ai/encode-vibe
```

### 파라미터
```javascript
reference_image_multiple: [encodedVibeData],
reference_information_extracted_multiple: [1.0],
reference_strength_multiple: [0.6]
```

---

## 메타데이터 (PNG Comment)

NAI 웹 호환 형식. PeroPix 확장 필드는 `peropix` 객체에 저장.

### 필드 매핑
| NAI | 앱 내부 |
|-----|---------|
| `uc` | `negative_prompt` |
| `scale` | `cfg` |
| `noise_schedule` | `scheduler` |
| `request_type` | `nai_model` |
| `sm` / `sm_dyn` | `smea` |

### 복원 불가 항목
- 바이브/캐릭터 레퍼런스 이미지 (파일 크기 문제)
- 베이스 이미지/마스크

---

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

---

## 해결한 주요 문제들

### 1. NAI 웹과 Character Reference 결과 불일치
**원인**: PIL 리사이즈 vs 브라우저 Canvas drawImage 알고리즘 차이
**해결**: 프론트엔드에서 Canvas로 처리 후 백엔드로 전송

### 2. NAI 웹과 Vibe 결과 불일치
**원인**: encode-vibe payload 구조 차이
**해결**: `{ image, information_extracted, model }` 최상위 레벨로 수정

### 3. Quality Tags / UC Preset 미적용
**원인**: NAI 서버가 V4.5에서 파라미터 처리 안함
**해결**: 클라이언트에서 직접 프롬프트에 태그 추가

### 4. 인페인트 마스크 경계 아티팩트
**원인**: 축소 디스플레이 크기 캔버스 → 원본 크기 업스케일 시 보간
**해결**: 캔버스 실제 크기 = 원본 크기, CSS로만 디스플레이 축소

### 5. V4 모델에서 제거된 옵션
- `dynamic_thresholding` - 항상 False
- `uncond_scale` - 항상 1.0

### 6. 프롬프트 입력란 하이라이트 overlay 불일치
**원인 1**: 스크롤바 유무에 따라 textarea 내부 너비 변경
**해결**: JS에서 `scrollHeight > clientHeight` 감지하여 overlay right 동적 조정 (1px/9px)

**원인 2**: `prompt-highlight-container`가 textarea보다 4px 더 큼
**해결**: container에 `display: flex` 적용하여 자식 크기에 맞춤

---

## 디버깅

### 콘솔 로그
```
[Generate] Job xxx started - N image(s), WxH, steps
[NAI] CharRef: fidelity=X, data_len=XXX
[NAI] Vibe cache hit/miss
```

### 일반적인 문제
1. **슬롯 안보임** → JavaScript 콘솔 확인
2. **API 실패** → 네트워크 탭 확인
3. **스타일 깨짐** → CSS overflow, z-index 확인

---

## 로컬 엔진 (ComfyUI 포팅)

### 7. 로컬 엔진 성능 문제 (2026-01-11)

**증상**: 20 steps 생성에 85초 이상 소요 (정상: 4-5초)

**발견된 문제들**:

#### 7-1. CLIP 인코딩에 `torch.no_grad()` 누락 ⭐ **핵심 원인**
- **위치**: `local_engine/clip/sdxl_clip.py`
- **문제**: 그래디언트 계산으로 ~7.6GB 추가 VRAM 사용
- **해결**: `encode_chunks()`, `encode()`에 `@torch.no_grad()` 데코레이터 추가
```python
@torch.no_grad()
def encode_chunks(self, chunks_l, chunks_g):
    ...
```

#### 7-2. 체크포인트 3번 로드
- **위치**: `local_engine/model_loader.py`
- **문제**: `load_unet()`, `load_vae()`, `load_clip()` 각각 체크포인트 로드 (~6.5GB × 3)
- **해결**: `load_all()`에서 1회만 로드 후 state_dict 분리
```python
def load_all(self, path):
    sd = self.load_checkpoint(path)  # 1회만 로드
    unet_sd = extract_unet_state_dict(sd)
    vae_sd = extract_vae_state_dict(sd)
    clip_l_sd, clip_g_sd = extract_clip_state_dicts(sd)
    del sd  # 즉시 해제
```

#### 7-3. 과도한 워밍업
- **위치**: `local_engine/model_loader.py`
- **문제**: 3 해상도 × 6 context 길이 = 18회 UNet forward
- **해결**: 1회로 축소 (128×128 latent, 77 tokens)

#### 7-4. 모델 언로드 시 VRAM 미해제
- **위치**: `local_engine/model_loader.py` `ModelCache.unload()`
- **문제**: `_models.clear()`만 호출, GPU 텐서 미해제
- **해결**: CPU로 이동 후 `gc.collect()` + `empty_cache()`
```python
def unload(self):
    for path, models in self._models.items():
        unet, vae, clip = models
        if unet: unet.to("cpu"); del unet
        ...
    gc.collect()
    torch.cuda.empty_cache()
```

#### 7-5. Debug print 문
- **위치**: `local_engine/generate.py`, `local_engine/samplers.py`
- **문제**: `print()` 호출 시 암묵적 GPU 동기화 발생
- **해결**: 모든 debug print 제거

### 로컬 엔진 성능 결과
| 측정 | 수정 전 | 수정 후 |
|------|---------|---------|
| 20 steps | 85초+ | 4.4초 |
| step당 | 4초+ | ~220ms |
| VRAM (CLIP 후) | 16GB | ~9GB |

### 로컬 엔진 주의사항
- 첫 생성은 CUDA 커널 컴파일로 느림 (정상)
- "torch version too old for priority" 로그는 PyTorch 버전 문제 (성능 영향 미미)
- xformers 미설치 시 PyTorch SDPA 사용 (정상 동작)
