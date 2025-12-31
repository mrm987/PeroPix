# PeroPix 컬러 시스템 리뉴얼 가이드

## 목적
기존 퍼플 테마를 마스코트 캐릭터 '페로'(경찰 수인)에 맞춰 네이비+골드 테마로 변경

---

## CSS 변수 변경 사항

### 배경 (퍼플 틴트 → 네이비 틴트)
| 변수 | 기존 | 변경 |
|------|------|------|
| --bg | #0f0f1a | #0C1219 |
| --bg-light | #1a1a2e | #141C26 |
| --bg-lighter | #252542 | #1C2530 |

### 강조색 (퍼플 → 네이비 블루)
| 변수 | 기존 | 변경 |
|------|------|------|
| --accent | #6F29DE | #4A7AB8 |
| --accent-hover | #8240E8 | #5B98D4 |
| --accent-dim | #5b21b6 | #2D4A6F |

### 신규 추가 (골드 - CTA/강조용)
| 변수 | 값 |
|------|------|
| --accent-gold | #F5B942 |
| --accent-gold-hover | #FFD06A |

### 텍스트
| 변수 | 기존 | 변경 |
|------|------|------|
| --text | #e2e8f0 | #FFFFFF |
| --text-dim | #94a3b8 | #7A8BA0 |

### 테두리
| 변수 | 기존 | 변경 |
|------|------|------|
| --border | #374151 | #243044 |

### 시맨틱 (톤 조정)
| 변수 | 기존 | 변경 |
|------|------|------|
| --success | #22c55e | #4CAF82 |
| --error | #ef4444 | #E57373 |
| --warning | #f59e0b | #F5B942 |

---

## 하드코딩 색상 → 변수 치환

| 기존 하드코딩 | 용도 | 치환 |
|--------------|------|------|
| #c8d0dc | 입력 필드 텍스트 | var(--text) |
| #9D6AE8 | 캐릭터 번호, Anlas 비용 | var(--accent-gold) |
| #ff6b6b | 삭제 버튼 | var(--error) |
| #ff6666 | 프리셋 삭제 버튼 | var(--error) |
| #3b82f6 | 드롭 영역 테두리 | var(--accent) |
| #4ade80 | Vibe 캐시 성공 | var(--success) |
| #fbbf24 | Vibe 캐시 경고 | var(--warning) |

---

## 최종 CSS 변수 (복사용)

```css
:root {
    /* Background */
    --bg: #0f1318;
    --bg-light: #162030;
    --bg-lighter: #1E2A3A;
    
    /* Accent - Navy Blue */
    --accent: #4A7AB8;
    --accent-hover: #5B98D4;
    --accent-dim: #2D4A6F;
    
    /* Accent - Gold (CTA, 강조) */
    --accent-gold: #F5B942;
    --accent-gold-hover: #FFD06A;
    
    /* Text */
    --text: #FFFFFF;
    --text-dim: #7A8BA0;
    
    /* Border */
    --border: #243044;
    
    /* Semantic */
    --success: #4CAF82;
    --error: #E57373;
    --warning: #F5B942;
}
```

---

## 적용 가이드

1. CSS 변수 정의 부분을 위 값으로 교체
2. 하드코딩된 색상을 검색하여 해당 변수로 치환
3. 주요 CTA 버튼(Queue 등)은 --accent-gold 사용 권장
4. 일반 토글/탭은 --accent (네이비) 유지
