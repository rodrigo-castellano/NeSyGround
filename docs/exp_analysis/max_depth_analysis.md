# Grounder Memory Analysis: Dense vs Flat Intermediate vs Full Flat

Every function in the forward pass, every tensor, exact memory.
GPU: 24 GB (20 GB available after model/data).

**Legend:**
- **★ body_a**: The enumeration intermediate — the dominant tensor
- Dense: current implementation (all padded to K_f_max)
- Flat-int: body_a flat+offsets, everything else dense (the plan: ~300 lines in enum.py)
- Full-flat: everything flat+offsets (full library rewrite)

---

## fb15k237 depth=2

**Parameters**: B=192, Re=30, M=2, K_f=3612, avg_k=13, depth=2, width=1, K_cap=256, G=3, G_body=4

### Step 0/1 (S_in=1, N=B×S=192)

| Function | Tensor | Dense shape | Dense | Flat-int shape | Flat-int | Full-flat |
|----------|--------|------------|-------|---------------|---------|----------|
|  | queries | [192,1,3] | 5KB | same | 5KB | 5KB |
| | remaining | [192,1,3,3] | 14KB | same | 14KB | 14KB |
|  | candidates | [192,30,256] | 12MB | [74880] | 599KB | 599KB |
|  | **★ body_a** | [192,30,256,2,3] | **71MB** | [74880,2,3] | **4MB** | **4MB** |
|  | exists | [192,30,256,2] | 3MB | [74880,2] | 150KB | 150KB |
|  | mask | [192,30,256] | 1MB | [74880] | 75KB | 75KB |
|  | rule_goals | [192,1,256,3,3] | 4MB | [192,1,256,3,3] | 4MB | [49152,3,3] 4MB |
| | rule_gbody | [192,1,256,2,3] | 2MB | [192,1,256,2,3] | 2MB | 2MB |
|  | proof_goals | [192,256,3,3] | 4MB | same | 4MB | 4MB |
| | grounding_body | [192,256,2,3] | 2MB | same | 2MB | 2MB |
| | accum_body | [192,256,4,3] | 5MB | same | 5MB | 5MB |
|  | collected_body | [192,256,4,3] | 5MB | same | 5MB | 184KB |

**Step 0 peak** (all live tensors):

| Phase | Dense | Flat-int | Full-flat |
|-------|-------|---------|----------|
| Resolve (body_a live) | 91MB | 19MB | 15MB |
| Pack (rule_goals live) | 19MB | 19MB | 14MB |
| **Step peak** | **91MB** | **19MB** | **15MB** |
| Bottleneck | body_a | body_a | body_a |

### Step 1/1 (S_in=256, N=B×S=49152)

| Function | Tensor | Dense shape | Dense | Flat-int shape | Flat-int | Full-flat |
|----------|--------|------------|-------|---------------|---------|----------|
|  | queries | [192,256,3] | 1MB | same | 1MB | 1MB |
| | remaining | [192,256,3,3] | 4MB | same | 4MB | 4MB |
|  | candidates | [49152,30,256] | 3.0GB | [19169280] | 153MB | 153MB |
|  | **★ body_a** | [49152,30,256,2,3] | **18.1GB** | [19169280,2,3] | **920MB** | **920MB** |
|  | exists | [49152,30,256,2] | 755MB | [19169280,2] | 38MB | 38MB |
|  | mask | [49152,30,256] | 377MB | [19169280] | 19MB | 19MB |
|  | rule_goals | [192,256,256,3,3] | 906MB | [192,256,256,3,3] | 906MB | [12582912,3,3] 906MB |
| | rule_gbody | [192,256,256,2,3] | 604MB | [192,256,256,2,3] | 604MB | 604MB |
|  | proof_goals | [192,256,3,3] | 4MB | same | 4MB | 4MB |
| | grounding_body | [192,256,2,3] | 2MB | same | 2MB | 2MB |
| | accum_body | [192,256,4,3] | 5MB | same | 5MB | 5MB |
|  | collected_body | [192,256,4,3] | 5MB | same | 5MB | 369KB |

**Step 1 peak** (all live tensors):

| Phase | Dense | Flat-int | Full-flat |
|-------|-------|---------|----------|
| Resolve (body_a live) | 19.3GB | 993MB | 989MB |
| Pack (rule_goals live) | 1.5GB | 1.5GB | 1.5GB |
| **Step peak** | **19.3GB** | **1.5GB** | **1.5GB** |
| Bottleneck | body_a | body_a | body_a |

### Overall Peak Memory

| Approach | Peak Memory | Fits 24GB? | vs Dense |
|----------|------------|-----------|---------|
| Dense (current) | **19.3GB** | ✅ | 1x |
| **(i) Flat intermediate** | **1.5GB** | ✅ | 13x less |
| (ii) Full flat | **1.5GB** | ✅ | 13x less |
| (ii) vs (i) additional savings | | | 1.0x |

---

## fb15k237 depth=3

**Parameters**: B=192, Re=30, M=2, K_f=3612, avg_k=13, depth=3, width=1, K_cap=256, G=4, G_body=6

### Step 0/2 (S_in=1, N=B×S=192)

| Function | Tensor | Dense shape | Dense | Flat-int shape | Flat-int | Full-flat |
|----------|--------|------------|-------|---------------|---------|----------|
|  | queries | [192,1,3] | 5KB | same | 5KB | 5KB |
| | remaining | [192,1,4,3] | 18KB | same | 18KB | 18KB |
|  | candidates | [192,30,256] | 12MB | [74880] | 599KB | 599KB |
|  | **★ body_a** | [192,30,256,2,3] | **71MB** | [74880,2,3] | **4MB** | **4MB** |
|  | exists | [192,30,256,2] | 3MB | [74880,2] | 150KB | 150KB |
|  | mask | [192,30,256] | 1MB | [74880] | 75KB | 75KB |
|  | rule_goals | [192,1,256,4,3] | 5MB | [192,1,256,4,3] | 5MB | [49152,4,3] 5MB |
| | rule_gbody | [192,1,256,2,3] | 2MB | [192,1,256,2,3] | 2MB | 2MB |
|  | proof_goals | [192,256,4,3] | 5MB | same | 5MB | 5MB |
| | grounding_body | [192,256,2,3] | 2MB | same | 2MB | 2MB |
| | accum_body | [192,256,6,3] | 7MB | same | 7MB | 7MB |
|  | collected_body | [192,256,6,3] | 7MB | same | 7MB | 276KB |

**Step 0 peak** (all live tensors):

| Phase | Dense | Flat-int | Full-flat |
|-------|-------|---------|----------|
| Resolve (body_a live) | 96MB | 25MB | 18MB |
| Pack (rule_goals live) | 26MB | 26MB | 19MB |
| **Step peak** | **96MB** | **26MB** | **19MB** |
| Bottleneck | body_a | rule_goals | rule_goals |

### Step 1/2 (S_in=256, N=B×S=49152)

| Function | Tensor | Dense shape | Dense | Flat-int shape | Flat-int | Full-flat |
|----------|--------|------------|-------|---------------|---------|----------|
|  | queries | [192,256,3] | 1MB | same | 1MB | 1MB |
| | remaining | [192,256,4,3] | 5MB | same | 5MB | 5MB |
|  | candidates | [49152,30,256] | 3.0GB | [19169280] | 153MB | 153MB |
|  | **★ body_a** | [49152,30,256,2,3] | **18.1GB** | [19169280,2,3] | **920MB** | **920MB** |
|  | exists | [49152,30,256,2] | 755MB | [19169280,2] | 38MB | 38MB |
|  | mask | [49152,30,256] | 377MB | [19169280] | 19MB | 19MB |
|  | rule_goals | [192,256,256,4,3] | 1.2GB | [192,256,256,4,3] | 1.2GB | [12582912,4,3] 1.2GB |
| | rule_gbody | [192,256,256,2,3] | 604MB | [192,256,256,2,3] | 604MB | 604MB |
|  | proof_goals | [192,256,4,3] | 5MB | same | 5MB | 5MB |
| | grounding_body | [192,256,2,3] | 2MB | same | 2MB | 2MB |
| | accum_body | [192,256,6,3] | 7MB | same | 7MB | 7MB |
|  | collected_body | [192,256,6,3] | 7MB | same | 7MB | 553KB |

**Step 1 peak** (all live tensors):

| Phase | Dense | Flat-int | Full-flat |
|-------|-------|---------|----------|
| Resolve (body_a live) | 19.3GB | 999MB | 992MB |
| Pack (rule_goals live) | 1.8GB | 1.8GB | 1.8GB |
| **Step peak** | **19.3GB** | **1.8GB** | **1.8GB** |
| Bottleneck | body_a | rule_goals | rule_goals |

### Step 2/2 (S_in=256, N=B×S=49152)

| Function | Tensor | Dense shape | Dense | Flat-int shape | Flat-int | Full-flat |
|----------|--------|------------|-------|---------------|---------|----------|
|  | queries | [192,256,3] | 1MB | same | 1MB | 1MB |
| | remaining | [192,256,4,3] | 5MB | same | 5MB | 5MB |
|  | candidates | [49152,30,256] | 3.0GB | [19169280] | 153MB | 153MB |
|  | **★ body_a** | [49152,30,256,2,3] | **18.1GB** | [19169280,2,3] | **920MB** | **920MB** |
|  | exists | [49152,30,256,2] | 755MB | [19169280,2] | 38MB | 38MB |
|  | mask | [49152,30,256] | 377MB | [19169280] | 19MB | 19MB |
|  | rule_goals | [192,256,256,4,3] | 1.2GB | [192,256,256,4,3] | 1.2GB | [12582912,4,3] 1.2GB |
| | rule_gbody | [192,256,256,2,3] | 604MB | [192,256,256,2,3] | 604MB | 604MB |
|  | proof_goals | [192,256,4,3] | 5MB | same | 5MB | 5MB |
| | grounding_body | [192,256,2,3] | 2MB | same | 2MB | 2MB |
| | accum_body | [192,256,6,3] | 7MB | same | 7MB | 7MB |
|  | collected_body | [192,256,6,3] | 7MB | same | 7MB | 829KB |

**Step 2 peak** (all live tensors):

| Phase | Dense | Flat-int | Full-flat |
|-------|-------|---------|----------|
| Resolve (body_a live) | 19.3GB | 999MB | 993MB |
| Pack (rule_goals live) | 1.8GB | 1.8GB | 1.8GB |
| **Step peak** | **19.3GB** | **1.8GB** | **1.8GB** |
| Bottleneck | body_a | rule_goals | rule_goals |

### Overall Peak Memory

| Approach | Peak Memory | Fits 24GB? | vs Dense |
|----------|------------|-----------|---------|
| Dense (current) | **19.3GB** | ✅ | 1x |
| **(i) Flat intermediate** | **1.8GB** | ✅ | 11x less |
| (ii) Full flat | **1.8GB** | ✅ | 11x less |
| (ii) vs (i) additional savings | | | 1.0x |

---

## fb15k237 depth=4

**Parameters**: B=192, Re=30, M=2, K_f=3612, avg_k=13, depth=4, width=1, K_cap=256, G=5, G_body=8

### Step 0/3 (S_in=1, N=B×S=192)

| Function | Tensor | Dense shape | Dense | Flat-int shape | Flat-int | Full-flat |
|----------|--------|------------|-------|---------------|---------|----------|
|  | queries | [192,1,3] | 5KB | same | 5KB | 5KB |
| | remaining | [192,1,5,3] | 23KB | same | 23KB | 23KB |
|  | candidates | [192,30,256] | 12MB | [74880] | 599KB | 599KB |
|  | **★ body_a** | [192,30,256,2,3] | **71MB** | [74880,2,3] | **4MB** | **4MB** |
|  | exists | [192,30,256,2] | 3MB | [74880,2] | 150KB | 150KB |
|  | mask | [192,30,256] | 1MB | [74880] | 75KB | 75KB |
|  | rule_goals | [192,1,256,5,3] | 6MB | [192,1,256,5,3] | 6MB | [49152,5,3] 6MB |
| | rule_gbody | [192,1,256,2,3] | 2MB | [192,1,256,2,3] | 2MB | 2MB |
|  | proof_goals | [192,256,5,3] | 6MB | same | 6MB | 6MB |
| | grounding_body | [192,256,2,3] | 2MB | same | 2MB | 2MB |
| | accum_body | [192,256,8,3] | 9MB | same | 9MB | 9MB |
|  | collected_body | [192,256,8,3] | 9MB | same | 9MB | 369KB |

**Step 0 peak** (all live tensors):

| Phase | Dense | Flat-int | Full-flat |
|-------|-------|---------|----------|
| Resolve (body_a live) | 102MB | 31MB | 22MB |
| Pack (rule_goals live) | 33MB | 33MB | 24MB |
| **Step peak** | **102MB** | **33MB** | **24MB** |
| Bottleneck | body_a | rule_goals | rule_goals |

### Step 1/3 (S_in=256, N=B×S=49152)

| Function | Tensor | Dense shape | Dense | Flat-int shape | Flat-int | Full-flat |
|----------|--------|------------|-------|---------------|---------|----------|
|  | queries | [192,256,3] | 1MB | same | 1MB | 1MB |
| | remaining | [192,256,5,3] | 6MB | same | 6MB | 6MB |
|  | candidates | [49152,30,256] | 3.0GB | [19169280] | 153MB | 153MB |
|  | **★ body_a** | [49152,30,256,2,3] | **18.1GB** | [19169280,2,3] | **920MB** | **920MB** |
|  | exists | [49152,30,256,2] | 755MB | [19169280,2] | 38MB | 38MB |
|  | mask | [49152,30,256] | 377MB | [19169280] | 19MB | 19MB |
|  | rule_goals | [192,256,256,5,3] | 1.5GB | [192,256,256,5,3] | 1.5GB | [12582912,5,3] 1.5GB |
| | rule_gbody | [192,256,256,2,3] | 604MB | [192,256,256,2,3] | 604MB | 604MB |
|  | proof_goals | [192,256,5,3] | 6MB | same | 6MB | 6MB |
| | grounding_body | [192,256,2,3] | 2MB | same | 2MB | 2MB |
| | accum_body | [192,256,8,3] | 9MB | same | 9MB | 9MB |
|  | collected_body | [192,256,8,3] | 9MB | same | 9MB | 737KB |

**Step 1 peak** (all live tensors):

| Phase | Dense | Flat-int | Full-flat |
|-------|-------|---------|----------|
| Resolve (body_a live) | 19.3GB | 1.0GB | 996MB |
| Pack (rule_goals live) | 2.1GB | 2.1GB | 2.1GB |
| **Step peak** | **19.3GB** | **2.1GB** | **2.1GB** |
| Bottleneck | body_a | rule_goals | rule_goals |

### Step 2/3 (S_in=256, N=B×S=49152)

| Function | Tensor | Dense shape | Dense | Flat-int shape | Flat-int | Full-flat |
|----------|--------|------------|-------|---------------|---------|----------|
|  | queries | [192,256,3] | 1MB | same | 1MB | 1MB |
| | remaining | [192,256,5,3] | 6MB | same | 6MB | 6MB |
|  | candidates | [49152,30,256] | 3.0GB | [19169280] | 153MB | 153MB |
|  | **★ body_a** | [49152,30,256,2,3] | **18.1GB** | [19169280,2,3] | **920MB** | **920MB** |
|  | exists | [49152,30,256,2] | 755MB | [19169280,2] | 38MB | 38MB |
|  | mask | [49152,30,256] | 377MB | [19169280] | 19MB | 19MB |
|  | rule_goals | [192,256,256,5,3] | 1.5GB | [192,256,256,5,3] | 1.5GB | [12582912,5,3] 1.5GB |
| | rule_gbody | [192,256,256,2,3] | 604MB | [192,256,256,2,3] | 604MB | 604MB |
|  | proof_goals | [192,256,5,3] | 6MB | same | 6MB | 6MB |
| | grounding_body | [192,256,2,3] | 2MB | same | 2MB | 2MB |
| | accum_body | [192,256,8,3] | 9MB | same | 9MB | 9MB |
|  | collected_body | [192,256,8,3] | 9MB | same | 9MB | 1MB |

**Step 2 peak** (all live tensors):

| Phase | Dense | Flat-int | Full-flat |
|-------|-------|---------|----------|
| Resolve (body_a live) | 19.3GB | 1.0GB | 996MB |
| Pack (rule_goals live) | 2.1GB | 2.1GB | 2.1GB |
| **Step peak** | **19.3GB** | **2.1GB** | **2.1GB** |
| Bottleneck | body_a | rule_goals | rule_goals |

### Step 3/3 (S_in=256, N=B×S=49152)

| Function | Tensor | Dense shape | Dense | Flat-int shape | Flat-int | Full-flat |
|----------|--------|------------|-------|---------------|---------|----------|
|  | queries | [192,256,3] | 1MB | same | 1MB | 1MB |
| | remaining | [192,256,5,3] | 6MB | same | 6MB | 6MB |
|  | candidates | [49152,30,256] | 3.0GB | [19169280] | 153MB | 153MB |
|  | **★ body_a** | [49152,30,256,2,3] | **18.1GB** | [19169280,2,3] | **920MB** | **920MB** |
|  | exists | [49152,30,256,2] | 755MB | [19169280,2] | 38MB | 38MB |
|  | mask | [49152,30,256] | 377MB | [19169280] | 19MB | 19MB |
|  | rule_goals | [192,256,256,5,3] | 1.5GB | [192,256,256,5,3] | 1.5GB | [12582912,5,3] 1.5GB |
| | rule_gbody | [192,256,256,2,3] | 604MB | [192,256,256,2,3] | 604MB | 604MB |
|  | proof_goals | [192,256,5,3] | 6MB | same | 6MB | 6MB |
| | grounding_body | [192,256,2,3] | 2MB | same | 2MB | 2MB |
| | accum_body | [192,256,8,3] | 9MB | same | 9MB | 9MB |
|  | collected_body | [192,256,8,3] | 9MB | same | 9MB | 1MB |

**Step 3 peak** (all live tensors):

| Phase | Dense | Flat-int | Full-flat |
|-------|-------|---------|----------|
| Resolve (body_a live) | 19.3GB | 1.0GB | 997MB |
| Pack (rule_goals live) | 2.1GB | 2.1GB | 2.1GB |
| **Step peak** | **19.3GB** | **2.1GB** | **2.1GB** |
| Bottleneck | body_a | rule_goals | rule_goals |

### Overall Peak Memory

| Approach | Peak Memory | Fits 24GB? | vs Dense |
|----------|------------|-----------|---------|
| Dense (current) | **19.3GB** | ✅ | 1x |
| **(i) Flat intermediate** | **2.1GB** | ✅ | 9x less |
| (ii) Full flat | **2.1GB** | ✅ | 9x less |
| (ii) vs (i) additional savings | | | 1.0x |

---

## wn18rr depth=2

**Parameters**: B=192, Re=8, M=2, K_f=474, avg_k=4, depth=2, width=1, K_cap=256, G=3, G_body=4

### Step 0/1 (S_in=1, N=B×S=192)

| Function | Tensor | Dense shape | Dense | Flat-int shape | Flat-int | Full-flat |
|----------|--------|------------|-------|---------------|---------|----------|
|  | queries | [192,1,3] | 5KB | same | 5KB | 5KB |
| | remaining | [192,1,3,3] | 14KB | same | 14KB | 14KB |
|  | candidates | [192,8,256] | 3MB | [6144] | 49KB | 49KB |
|  | **★ body_a** | [192,8,256,2,3] | **19MB** | [6144,2,3] | **295KB** | **295KB** |
|  | exists | [192,8,256,2] | 786KB | [6144,2] | 12KB | 12KB |
|  | mask | [192,8,256] | 393KB | [6144] | 6KB | 6KB |
|  | rule_goals | [192,1,256,3,3] | 4MB | [192,1,32,3,3] | 442KB | [6144,3,3] 442KB |
| | rule_gbody | [192,1,256,2,3] | 2MB | [192,1,32,2,3] | 295KB | 295KB |
|  | proof_goals | [192,32,3,3] | 442KB | same | 442KB | 442KB |
| | grounding_body | [192,32,2,3] | 295KB | same | 295KB | 295KB |
| | accum_body | [192,32,4,3] | 590KB | same | 590KB | 590KB |
|  | collected_body | [192,256,4,3] | 5MB | same | 5MB | 184KB |

**Step 0 peak** (all live tensors):

| Phase | Dense | Flat-int | Full-flat |
|-------|-------|---------|----------|
| Resolve (body_a live) | 26MB | 6MB | 2MB |
| Pack (rule_goals live) | 12MB | 6MB | 2MB |
| **Step peak** | **26MB** | **6MB** | **2MB** |
| Bottleneck | body_a | rule_goals | rule_goals |

### Step 1/1 (S_in=32, N=B×S=6144)

| Function | Tensor | Dense shape | Dense | Flat-int shape | Flat-int | Full-flat |
|----------|--------|------------|-------|---------------|---------|----------|
|  | queries | [192,32,3] | 147KB | same | 147KB | 147KB |
| | remaining | [192,32,3,3] | 442KB | same | 442KB | 442KB |
|  | candidates | [6144,8,256] | 101MB | [196608] | 2MB | 2MB |
|  | **★ body_a** | [6144,8,256,2,3] | **604MB** | [196608,2,3] | **9MB** | **9MB** |
|  | exists | [6144,8,256,2] | 25MB | [196608,2] | 393KB | 393KB |
|  | mask | [6144,8,256] | 13MB | [196608] | 197KB | 197KB |
|  | rule_goals | [192,32,256,3,3] | 113MB | [192,32,32,3,3] | 14MB | [196608,3,3] 14MB |
| | rule_gbody | [192,32,256,2,3] | 75MB | [192,32,32,2,3] | 9MB | 9MB |
|  | proof_goals | [192,32,3,3] | 442KB | same | 442KB | 442KB |
| | grounding_body | [192,32,2,3] | 295KB | same | 295KB | 295KB |
| | accum_body | [192,32,4,3] | 590KB | same | 590KB | 590KB |
|  | collected_body | [192,256,4,3] | 5MB | same | 5MB | 369KB |

**Step 1 peak** (all live tensors):

| Phase | Dense | Flat-int | Full-flat |
|-------|-------|---------|----------|
| Resolve (body_a live) | 648MB | 16MB | 12MB |
| Pack (rule_goals live) | 194MB | 29MB | 25MB |
| **Step peak** | **648MB** | **29MB** | **25MB** |
| Bottleneck | body_a | rule_goals | rule_goals |

### Overall Peak Memory

| Approach | Peak Memory | Fits 24GB? | vs Dense |
|----------|------------|-----------|---------|
| Dense (current) | **648MB** | ✅ | 1x |
| **(i) Flat intermediate** | **29MB** | ✅ | 22x less |
| (ii) Full flat | **25MB** | ✅ | 26x less |
| (ii) vs (i) additional savings | | | 1.2x |

---

## wn18rr depth=3

**Parameters**: B=192, Re=8, M=2, K_f=474, avg_k=4, depth=3, width=1, K_cap=256, G=4, G_body=6

### Step 0/2 (S_in=1, N=B×S=192)

| Function | Tensor | Dense shape | Dense | Flat-int shape | Flat-int | Full-flat |
|----------|--------|------------|-------|---------------|---------|----------|
|  | queries | [192,1,3] | 5KB | same | 5KB | 5KB |
| | remaining | [192,1,4,3] | 18KB | same | 18KB | 18KB |
|  | candidates | [192,8,256] | 3MB | [6144] | 49KB | 49KB |
|  | **★ body_a** | [192,8,256,2,3] | **19MB** | [6144,2,3] | **295KB** | **295KB** |
|  | exists | [192,8,256,2] | 786KB | [6144,2] | 12KB | 12KB |
|  | mask | [192,8,256] | 393KB | [6144] | 6KB | 6KB |
|  | rule_goals | [192,1,256,4,3] | 5MB | [192,1,32,4,3] | 590KB | [6144,4,3] 590KB |
| | rule_gbody | [192,1,256,2,3] | 2MB | [192,1,32,2,3] | 295KB | 295KB |
|  | proof_goals | [192,32,4,3] | 590KB | same | 590KB | 590KB |
| | grounding_body | [192,32,2,3] | 295KB | same | 295KB | 295KB |
| | accum_body | [192,32,6,3] | 885KB | same | 885KB | 885KB |
|  | collected_body | [192,256,6,3] | 7MB | same | 7MB | 276KB |

**Step 0 peak** (all live tensors):

| Phase | Dense | Flat-int | Full-flat |
|-------|-------|---------|----------|
| Resolve (body_a live) | 29MB | 9MB | 2MB |
| Pack (rule_goals live) | 16MB | 9MB | 3MB |
| **Step peak** | **29MB** | **9MB** | **3MB** |
| Bottleneck | body_a | rule_goals | rule_goals |

### Step 1/2 (S_in=32, N=B×S=6144)

| Function | Tensor | Dense shape | Dense | Flat-int shape | Flat-int | Full-flat |
|----------|--------|------------|-------|---------------|---------|----------|
|  | queries | [192,32,3] | 147KB | same | 147KB | 147KB |
| | remaining | [192,32,4,3] | 590KB | same | 590KB | 590KB |
|  | candidates | [6144,8,256] | 101MB | [196608] | 2MB | 2MB |
|  | **★ body_a** | [6144,8,256,2,3] | **604MB** | [196608,2,3] | **9MB** | **9MB** |
|  | exists | [6144,8,256,2] | 25MB | [196608,2] | 393KB | 393KB |
|  | mask | [6144,8,256] | 13MB | [196608] | 197KB | 197KB |
|  | rule_goals | [192,32,256,4,3] | 151MB | [192,32,32,4,3] | 19MB | [196608,4,3] 19MB |
| | rule_gbody | [192,32,256,2,3] | 75MB | [192,32,32,2,3] | 9MB | 9MB |
|  | proof_goals | [192,32,4,3] | 590KB | same | 590KB | 590KB |
| | grounding_body | [192,32,2,3] | 295KB | same | 295KB | 295KB |
| | accum_body | [192,32,6,3] | 885KB | same | 885KB | 885KB |
|  | collected_body | [192,256,6,3] | 7MB | same | 7MB | 553KB |

**Step 1 peak** (all live tensors):

| Phase | Dense | Flat-int | Full-flat |
|-------|-------|---------|----------|
| Resolve (body_a live) | 651MB | 19MB | 12MB |
| Pack (rule_goals live) | 235MB | 37MB | 30MB |
| **Step peak** | **651MB** | **37MB** | **30MB** |
| Bottleneck | body_a | rule_goals | rule_goals |

### Step 2/2 (S_in=32, N=B×S=6144)

| Function | Tensor | Dense shape | Dense | Flat-int shape | Flat-int | Full-flat |
|----------|--------|------------|-------|---------------|---------|----------|
|  | queries | [192,32,3] | 147KB | same | 147KB | 147KB |
| | remaining | [192,32,4,3] | 590KB | same | 590KB | 590KB |
|  | candidates | [6144,8,256] | 101MB | [196608] | 2MB | 2MB |
|  | **★ body_a** | [6144,8,256,2,3] | **604MB** | [196608,2,3] | **9MB** | **9MB** |
|  | exists | [6144,8,256,2] | 25MB | [196608,2] | 393KB | 393KB |
|  | mask | [6144,8,256] | 13MB | [196608] | 197KB | 197KB |
|  | rule_goals | [192,32,256,4,3] | 151MB | [192,32,32,4,3] | 19MB | [196608,4,3] 19MB |
| | rule_gbody | [192,32,256,2,3] | 75MB | [192,32,32,2,3] | 9MB | 9MB |
|  | proof_goals | [192,32,4,3] | 590KB | same | 590KB | 590KB |
| | grounding_body | [192,32,2,3] | 295KB | same | 295KB | 295KB |
| | accum_body | [192,32,6,3] | 885KB | same | 885KB | 885KB |
|  | collected_body | [192,256,6,3] | 7MB | same | 7MB | 829KB |

**Step 2 peak** (all live tensors):

| Phase | Dense | Flat-int | Full-flat |
|-------|-------|---------|----------|
| Resolve (body_a live) | 651MB | 19MB | 13MB |
| Pack (rule_goals live) | 235MB | 37MB | 31MB |
| **Step peak** | **651MB** | **37MB** | **31MB** |
| Bottleneck | body_a | rule_goals | rule_goals |

### Overall Peak Memory

| Approach | Peak Memory | Fits 24GB? | vs Dense |
|----------|------------|-----------|---------|
| Dense (current) | **651MB** | ✅ | 1x |
| **(i) Flat intermediate** | **37MB** | ✅ | 18x less |
| (ii) Full flat | **31MB** | ✅ | 21x less |
| (ii) vs (i) additional savings | | | 1.2x |

---

## wn18rr depth=4

**Parameters**: B=192, Re=8, M=2, K_f=474, avg_k=4, depth=4, width=1, K_cap=256, G=5, G_body=8

### Step 0/3 (S_in=1, N=B×S=192)

| Function | Tensor | Dense shape | Dense | Flat-int shape | Flat-int | Full-flat |
|----------|--------|------------|-------|---------------|---------|----------|
|  | queries | [192,1,3] | 5KB | same | 5KB | 5KB |
| | remaining | [192,1,5,3] | 23KB | same | 23KB | 23KB |
|  | candidates | [192,8,256] | 3MB | [6144] | 49KB | 49KB |
|  | **★ body_a** | [192,8,256,2,3] | **19MB** | [6144,2,3] | **295KB** | **295KB** |
|  | exists | [192,8,256,2] | 786KB | [6144,2] | 12KB | 12KB |
|  | mask | [192,8,256] | 393KB | [6144] | 6KB | 6KB |
|  | rule_goals | [192,1,256,5,3] | 6MB | [192,1,32,5,3] | 737KB | [6144,5,3] 737KB |
| | rule_gbody | [192,1,256,2,3] | 2MB | [192,1,32,2,3] | 295KB | 295KB |
|  | proof_goals | [192,32,5,3] | 737KB | same | 737KB | 737KB |
| | grounding_body | [192,32,2,3] | 295KB | same | 295KB | 295KB |
| | accum_body | [192,32,8,3] | 1MB | same | 1MB | 1MB |
|  | collected_body | [192,256,8,3] | 9MB | same | 9MB | 369KB |

**Step 0 peak** (all live tensors):

| Phase | Dense | Flat-int | Full-flat |
|-------|-------|---------|----------|
| Resolve (body_a live) | 32MB | 12MB | 3MB |
| Pack (rule_goals live) | 20MB | 12MB | 3MB |
| **Step peak** | **32MB** | **12MB** | **3MB** |
| Bottleneck | body_a | rule_goals | rule_goals |

### Step 1/3 (S_in=32, N=B×S=6144)

| Function | Tensor | Dense shape | Dense | Flat-int shape | Flat-int | Full-flat |
|----------|--------|------------|-------|---------------|---------|----------|
|  | queries | [192,32,3] | 147KB | same | 147KB | 147KB |
| | remaining | [192,32,5,3] | 737KB | same | 737KB | 737KB |
|  | candidates | [6144,8,256] | 101MB | [196608] | 2MB | 2MB |
|  | **★ body_a** | [6144,8,256,2,3] | **604MB** | [196608,2,3] | **9MB** | **9MB** |
|  | exists | [6144,8,256,2] | 25MB | [196608,2] | 393KB | 393KB |
|  | mask | [6144,8,256] | 13MB | [196608] | 197KB | 197KB |
|  | rule_goals | [192,32,256,5,3] | 189MB | [192,32,32,5,3] | 24MB | [196608,5,3] 24MB |
| | rule_gbody | [192,32,256,2,3] | 75MB | [192,32,32,2,3] | 9MB | 9MB |
|  | proof_goals | [192,32,5,3] | 737KB | same | 737KB | 737KB |
| | grounding_body | [192,32,2,3] | 295KB | same | 295KB | 295KB |
| | accum_body | [192,32,8,3] | 1MB | same | 1MB | 1MB |
|  | collected_body | [192,256,8,3] | 9MB | same | 9MB | 737KB |

**Step 1 peak** (all live tensors):

| Phase | Dense | Flat-int | Full-flat |
|-------|-------|---------|----------|
| Resolve (body_a live) | 653MB | 22MB | 13MB |
| Pack (rule_goals live) | 276MB | 44MB | 36MB |
| **Step peak** | **653MB** | **44MB** | **36MB** |
| Bottleneck | body_a | rule_goals | rule_goals |

### Step 2/3 (S_in=32, N=B×S=6144)

| Function | Tensor | Dense shape | Dense | Flat-int shape | Flat-int | Full-flat |
|----------|--------|------------|-------|---------------|---------|----------|
|  | queries | [192,32,3] | 147KB | same | 147KB | 147KB |
| | remaining | [192,32,5,3] | 737KB | same | 737KB | 737KB |
|  | candidates | [6144,8,256] | 101MB | [196608] | 2MB | 2MB |
|  | **★ body_a** | [6144,8,256,2,3] | **604MB** | [196608,2,3] | **9MB** | **9MB** |
|  | exists | [6144,8,256,2] | 25MB | [196608,2] | 393KB | 393KB |
|  | mask | [6144,8,256] | 13MB | [196608] | 197KB | 197KB |
|  | rule_goals | [192,32,256,5,3] | 189MB | [192,32,32,5,3] | 24MB | [196608,5,3] 24MB |
| | rule_gbody | [192,32,256,2,3] | 75MB | [192,32,32,2,3] | 9MB | 9MB |
|  | proof_goals | [192,32,5,3] | 737KB | same | 737KB | 737KB |
| | grounding_body | [192,32,2,3] | 295KB | same | 295KB | 295KB |
| | accum_body | [192,32,8,3] | 1MB | same | 1MB | 1MB |
|  | collected_body | [192,256,8,3] | 9MB | same | 9MB | 1MB |

**Step 2 peak** (all live tensors):

| Phase | Dense | Flat-int | Full-flat |
|-------|-------|---------|----------|
| Resolve (body_a live) | 653MB | 22MB | 13MB |
| Pack (rule_goals live) | 276MB | 44MB | 36MB |
| **Step peak** | **653MB** | **44MB** | **36MB** |
| Bottleneck | body_a | rule_goals | rule_goals |

### Step 3/3 (S_in=32, N=B×S=6144)

| Function | Tensor | Dense shape | Dense | Flat-int shape | Flat-int | Full-flat |
|----------|--------|------------|-------|---------------|---------|----------|
|  | queries | [192,32,3] | 147KB | same | 147KB | 147KB |
| | remaining | [192,32,5,3] | 737KB | same | 737KB | 737KB |
|  | candidates | [6144,8,256] | 101MB | [196608] | 2MB | 2MB |
|  | **★ body_a** | [6144,8,256,2,3] | **604MB** | [196608,2,3] | **9MB** | **9MB** |
|  | exists | [6144,8,256,2] | 25MB | [196608,2] | 393KB | 393KB |
|  | mask | [6144,8,256] | 13MB | [196608] | 197KB | 197KB |
|  | rule_goals | [192,32,256,5,3] | 189MB | [192,32,32,5,3] | 24MB | [196608,5,3] 24MB |
| | rule_gbody | [192,32,256,2,3] | 75MB | [192,32,32,2,3] | 9MB | 9MB |
|  | proof_goals | [192,32,5,3] | 737KB | same | 737KB | 737KB |
| | grounding_body | [192,32,2,3] | 295KB | same | 295KB | 295KB |
| | accum_body | [192,32,8,3] | 1MB | same | 1MB | 1MB |
|  | collected_body | [192,256,8,3] | 9MB | same | 9MB | 1MB |

**Step 3 peak** (all live tensors):

| Phase | Dense | Flat-int | Full-flat |
|-------|-------|---------|----------|
| Resolve (body_a live) | 653MB | 22MB | 14MB |
| Pack (rule_goals live) | 276MB | 44MB | 36MB |
| **Step peak** | **653MB** | **44MB** | **36MB** |
| Bottleneck | body_a | rule_goals | rule_goals |

### Overall Peak Memory

| Approach | Peak Memory | Fits 24GB? | vs Dense |
|----------|------------|-----------|---------|
| Dense (current) | **653MB** | ✅ | 1x |
| **(i) Flat intermediate** | **44MB** | ✅ | 15x less |
| (ii) Full flat | **36MB** | ✅ | 18x less |
| (ii) vs (i) additional savings | | | 1.2x |

---

## countries_s3 d=2

**Parameters**: B=768, Re=3, M=3, K_f=16, avg_k=5, depth=2, width=1, K_cap=64, G=5, G_body=6

### Step 0/1 (S_in=1, N=B×S=768)

| Function | Tensor | Dense shape | Dense | Flat-int shape | Flat-int | Full-flat |
|----------|--------|------------|-------|---------------|---------|----------|
|  | queries | [768,1,3] | 18KB | same | 18KB | 18KB |
| | remaining | [768,1,5,3] | 92KB | same | 92KB | 92KB |
|  | candidates | [768,3,16] | 295KB | [11520] | 92KB | 92KB |
|  | **★ body_a** | [768,3,16,3,3] | **3MB** | [11520,3,3] | **829KB** | **829KB** |
|  | exists | [768,3,16,3] | 111KB | [11520,3] | 35KB | 35KB |
|  | mask | [768,3,16] | 37KB | [11520] | 12KB | 12KB |
|  | rule_goals | [768,1,48,5,3] | 4MB | [768,1,15,5,3] | 1MB | [11520,5,3] 1MB |
| | rule_gbody | [768,1,48,3,3] | 3MB | [768,1,15,3,3] | 829KB | 829KB |
|  | proof_goals | [768,15,5,3] | 1MB | same | 1MB | 1MB |
| | grounding_body | [768,15,3,3] | 829KB | same | 829KB | 829KB |
| | accum_body | [768,15,6,3] | 2MB | same | 2MB | 2MB |
|  | collected_body | [768,64,6,3] | 7MB | same | 7MB | 1MB |

**Step 0 peak** (all live tensors):

| Phase | Dense | Flat-int | Full-flat |
|-------|-------|---------|----------|
| Resolve (body_a live) | 14MB | 12MB | 6MB |
| Pack (rule_goals live) | 17MB | 12MB | 6MB |
| **Step peak** | **17MB** | **12MB** | **6MB** |
| Bottleneck | rule_goals | rule_goals | rule_goals |

### Step 1/1 (S_in=15, N=B×S=11520)

| Function | Tensor | Dense shape | Dense | Flat-int shape | Flat-int | Full-flat |
|----------|--------|------------|-------|---------------|---------|----------|
|  | queries | [768,15,3] | 276KB | same | 276KB | 276KB |
| | remaining | [768,15,5,3] | 1MB | same | 1MB | 1MB |
|  | candidates | [11520,3,16] | 4MB | [172800] | 1MB | 1MB |
|  | **★ body_a** | [11520,3,16,3,3] | **40MB** | [172800,3,3] | **12MB** | **12MB** |
|  | exists | [11520,3,16,3] | 2MB | [172800,3] | 518KB | 518KB |
|  | mask | [11520,3,16] | 553KB | [172800] | 173KB | 173KB |
|  | rule_goals | [768,15,48,5,3] | 66MB | [768,15,15,5,3] | 21MB | [172800,5,3] 21MB |
| | rule_gbody | [768,15,48,3,3] | 40MB | [768,15,15,3,3] | 12MB | 12MB |
|  | proof_goals | [768,15,5,3] | 1MB | same | 1MB | 1MB |
| | grounding_body | [768,15,3,3] | 829KB | same | 829KB | 829KB |
| | accum_body | [768,15,6,3] | 2MB | same | 2MB | 2MB |
|  | collected_body | [768,64,6,3] | 7MB | same | 7MB | 2MB |

**Step 1 peak** (all live tensors):

| Phase | Dense | Flat-int | Full-flat |
|-------|-------|---------|----------|
| Resolve (body_a live) | 53MB | 24MB | 19MB |
| Pack (rule_goals live) | 116MB | 43MB | 38MB |
| **Step peak** | **116MB** | **43MB** | **38MB** |
| Bottleneck | rule_goals | rule_goals | rule_goals |

### Overall Peak Memory

| Approach | Peak Memory | Fits 24GB? | vs Dense |
|----------|------------|-----------|---------|
| Dense (current) | **116MB** | ✅ | 1x |
| **(i) Flat intermediate** | **43MB** | ✅ | 3x less |
| (ii) Full flat | **38MB** | ✅ | 3x less |
| (ii) vs (i) additional savings | | | 1.1x |

---

## Summary

| Dataset | Depth | Dense | Flat-int (i) | Full-flat (ii) | (i) savings | (ii) vs (i) |
|---------|-------|-------|-------------|----------------|-------------|-------------|
| fb15k237 depth=2 | 2 | ✅ 19.0GB | ✅ 1.8GB | ✅ 1.8GB | 10x | 1.0x |
| fb15k237 depth=3 | 3 | ✅ 19.3GB | ✅ 2.1GB | ✅ 2.1GB | 9x | 1.0x |
| fb15k237 depth=4 | 4 | ✅ 19.7GB | ✅ 2.5GB | ✅ 2.4GB | 8x | 1.0x |
| wn18rr depth=2 | 2 | ✅ 723MB | ✅ 29MB | ✅ 25MB | 25x | 1.2x |
| wn18rr depth=3 | 3 | ✅ 764MB | ✅ 37MB | ✅ 30MB | 21x | 1.2x |
| wn18rr depth=4 | 4 | ✅ 804MB | ✅ 44MB | ✅ 35MB | 18x | 1.3x |
| countries_s3 d=2 | 2 | ✅ 116MB | ✅ 43MB | ✅ 37MB | 3x | 1.2x |

## Conclusion

1. **Flat intermediate (i) is sufficient.** It gives 9-22x memory savings, enabling depth 4+ on fb15k237 at B=192.
2. **Full flat (ii) adds at most 1.2x** additional savings — not worth a library rewrite.
3. **The bottleneck is always ** (the enumeration intermediate). Once that is flat, everything else is negligible.
4. **Implementation: ~300 lines in **, no changes to pack_states, collect_groundings, filters, or reasoning.
---

## Zero Grounding Loss Analysis

The tables above use K_cap=256, which **caps children per state** and loses groundings.
This section shows memory when **no caps are applied** — every valid grounding is kept.

Without caps:
- G_use = K_f (no per-rule cap → enumerate returns all valid candidates)
- K_enum = Re × G_use (no topk cap → all children kept)
- S = K_enum (states = all children from previous step)
- tG = uncapped (collected buffer grows freely)

For **flat approaches**, G_use = avg_k (only valid entries stored, no padding).
For **dense**, G_use = K_f_max (padded to dataset worst case).

### countries_s3 (Re=3, M=3, K_f_max=16, avg_k=5)

| Depth | B | Step | S_in | N=B×S | Tensor | Dense (G_use=16) | Flat-int (G_use≈5) | Full-flat |
|-------|---|------|------|-------|--------|------------------------|-------------------------|----------|
| 1 | 768 | 0 | 1/1 | 768/768 | **body_a** | **3MB** | **829KB** | 829KB |
| | | | | | rule_goals | 3MB | 829KB | 829KB |
| | | | | | states+coll | 8MB | 2MB | 2MB |
| | | | | | **STEP PEAK** | **13MB** | **4MB** | **4MB** |
| **d=1** | | | | | **OVERALL PEAK** | **✅ 13MB** | **✅ 4MB** | **✅ 4MB** |
| | | | | | savings vs dense | 1x | 3x | — |
| | | | | | (ii) vs (i) | | | 1.0x |
| 2 | 768 | 0 | 1/1 | 768/768 | **body_a** | **3MB** | **829KB** | 829KB |
| | | | | | rule_goals | 4MB | 1MB | 1MB |
| | | | | | states+coll | 15MB | 5MB | 5MB |
| | | | | | **STEP PEAK** | **22MB** | **7MB** | **7MB** |
| 2 | 768 | 1 | 48/15 | 36864/11520 | **body_a** | **127MB** | **12MB** | 12MB |
| | | | | | rule_goals | 212MB | 21MB | 21MB |
| | | | | | states+coll | 15MB | 5MB | 5MB |
| | | | | | **STEP PEAK** | **355MB** | **38MB** | **38MB** |
| **d=2** | | | | | **OVERALL PEAK** | **✅ 355MB** | **✅ 38MB** | **✅ 38MB** |
| | | | | | savings vs dense | 1x | 9x | — |
| | | | | | (ii) vs (i) | | | 1.0x |
| 3 | 128 | 0 | 1/1 | 128/128 | **body_a** | **442KB** | **138KB** | 138KB |
| | | | | | rule_goals | 1MB | 323KB | 323KB |
| | | | | | states+coll | 4MB | 1MB | 1MB |
| | | | | | **STEP PEAK** | **5MB** | **2MB** | **2MB** |
| 3 | 128 | 1 | 48/15 | 6144/1920 | **body_a** | **21MB** | **2MB** | 2MB |
| | | | | | rule_goals | 50MB | 5MB | 5MB |
| | | | | | states+coll | 4MB | 1MB | 1MB |
| | | | | | **STEP PEAK** | **74MB** | **8MB** | **8MB** |
| 3 | 128 | 2 | 48/15 | 6144/1920 | **body_a** | **21MB** | **2MB** | 2MB |
| | | | | | rule_goals | 50MB | 5MB | 5MB |
| | | | | | states+coll | 4MB | 1MB | 1MB |
| | | | | | **STEP PEAK** | **74MB** | **8MB** | **8MB** |
| **d=3** | | | | | **OVERALL PEAK** | **✅ 74MB** | **✅ 8MB** | **✅ 8MB** |
| | | | | | savings vs dense | 1x | 9x | — |
| | | | | | (ii) vs (i) | | | 1.0x |

### wn18rr (Re=8, M=2, K_f_max=474, avg_k=4)

| Depth | B | Step | S_in | N=B×S | Tensor | Dense (G_use=474) | Flat-int (G_use≈4) | Full-flat |
|-------|---|------|------|-------|--------|------------------------|-------------------------|----------|
| 1 | 192 | 0 | 1/1 | 192/192 | **body_a** | **35MB** | **295KB** | 295KB |
| | | | | | rule_goals | 35MB | 295KB | 295KB |
| | | | | | states+coll | 105MB | 885KB | 885KB |
| | | | | | **STEP PEAK** | **175MB** | **1MB** | **1MB** |
| **d=1** | | | | | **OVERALL PEAK** | **✅ 175MB** | **✅ 1MB** | **✅ 1MB** |
| | | | | | savings vs dense | 1x | 118x | — |
| | | | | | (ii) vs (i) | | | 1.0x |
| 2 | 192 | 0 | 1/1 | 192/192 | **body_a** | **35MB** | **295KB** | 295KB |
| | | | | | rule_goals | 52MB | 442KB | 442KB |
| | | | | | states+coll | 192MB | 2MB | 2MB |
| | | | | | **STEP PEAK** | **280MB** | **2MB** | **2MB** |
| 2 | 192 | 1 | 3792/32 | 728064/6144 | **body_a** | **132.5GB** | **9MB** | 9MB |
| | | | | | rule_goals | 198.8GB | 14MB | 14MB |
| | | | | | states+coll | 192MB | 2MB | 2MB |
| | | | | | **STEP PEAK** | **331.5GB** | **25MB** | **25MB** |
| **d=2** | | | | | **OVERALL PEAK** | **❌ 331.5GB** | **✅ 25MB** | **✅ 25MB** |
| | | | | | savings vs dense | 1x | 13147x | — |
| | | | | | (ii) vs (i) | | | 1.0x |
| 3 | 192 | 0 | 1/1 | 192/192 | **body_a** | **35MB** | **295KB** | 295KB |
| | | | | | rule_goals | 70MB | 590KB | 590KB |
| | | | | | states+coll | 280MB | 2MB | 2MB |
| | | | | | **STEP PEAK** | **384MB** | **3MB** | **3MB** |
| 3 | 192 | 1 | 3792/32 | 728064/6144 | **body_a** | **132.5GB** | **9MB** | 9MB |
| | | | | | rule_goals | 265.0GB | 19MB | 19MB |
| | | | | | states+coll | 280MB | 2MB | 2MB |
| | | | | | **STEP PEAK** | **397.8GB** | **31MB** | **31MB** |
| 3 | 192 | 2 | 3792/32 | 728064/6144 | **body_a** | **132.5GB** | **9MB** | 9MB |
| | | | | | rule_goals | 265.0GB | 19MB | 19MB |
| | | | | | states+coll | 280MB | 2MB | 2MB |
| | | | | | **STEP PEAK** | **397.8GB** | **31MB** | **31MB** |
| **d=3** | | | | | **OVERALL PEAK** | **❌ 397.8GB** | **✅ 31MB** | **✅ 31MB** |
| | | | | | savings vs dense | 1x | 12971x | — |
| | | | | | (ii) vs (i) | | | 1.0x |

### fb15k237 (Re=30, M=2, K_f_max=3612, avg_k=13)

| Depth | B | Step | S_in | N=B×S | Tensor | Dense (G_use=3612) | Flat-int (G_use≈13) | Full-flat |
|-------|---|------|------|-------|--------|------------------------|-------------------------|----------|
| 1 | 192 | 0 | 1/1 | 192/192 | **body_a** | **999MB** | **4MB** | 4MB |
| | | | | | rule_goals | 999MB | 4MB | 4MB |
| | | | | | states+coll | 3.0GB | 11MB | 11MB |
| | | | | | **STEP PEAK** | **5.0GB** | **18MB** | **18MB** |
| **d=1** | | | | | **OVERALL PEAK** | **✅ 5.0GB** | **✅ 18MB** | **✅ 18MB** |
| | | | | | savings vs dense | 1x | 278x | — |
| | | | | | (ii) vs (i) | | | 1.0x |
| 2 | 192 | 0 | 1/1 | 192/192 | **body_a** | **999MB** | **4MB** | 4MB |
| | | | | | rule_goals | 1.5GB | 5MB | 5MB |
| | | | | | states+coll | 5.5GB | 20MB | 20MB |
| | | | | | **STEP PEAK** | **8.0GB** | **29MB** | **29MB** |
| 2 | 192 | 1 | 108360/390 | 20805120/74880 | **body_a** | **108213.3GB** | **1.4GB** | 1.4GB |
| | | | | | rule_goals | 162319.9GB | 2.1GB | 2.1GB |
| | | | | | states+coll | 5.5GB | 20MB | 20MB |
| | | | | | **STEP PEAK** | **270538.6GB** | **3.5GB** | **3.5GB** |
| **d=2** | | | | | **OVERALL PEAK** | **❌ 270538.6GB** | **✅ 3.5GB** | **✅ 3.5GB** |
| | | | | | savings vs dense | 1x | 76767x | — |
| | | | | | (ii) vs (i) | | | 1.0x |
| 3 | 192 | 0 | 1/1 | 192/192 | **body_a** | **999MB** | **4MB** | 4MB |
| | | | | | rule_goals | 2.0GB | 7MB | 7MB |
| | | | | | states+coll | 8.0GB | 29MB | 29MB |
| | | | | | **STEP PEAK** | **11.0GB** | **40MB** | **40MB** |
| 3 | 192 | 1 | 108360/390 | 20805120/74880 | **body_a** | **108213.3GB** | **1.4GB** | 1.4GB |
| | | | | | rule_goals | 216426.5GB | 2.8GB | 2.8GB |
| | | | | | states+coll | 8.0GB | 29MB | 29MB |
| | | | | | **STEP PEAK** | **324647.8GB** | **4.2GB** | **4.2GB** |
| 3 | 192 | 2 | 108360/390 | 20805120/74880 | **body_a** | **108213.3GB** | **1.4GB** | 1.4GB |
| | | | | | rule_goals | 216426.5GB | 2.8GB | 2.8GB |
| | | | | | states+coll | 8.0GB | 29MB | 29MB |
| | | | | | **STEP PEAK** | **324647.8GB** | **4.2GB** | **4.2GB** |
| **d=3** | | | | | **OVERALL PEAK** | **❌ 324647.8GB** | **✅ 4.2GB** | **✅ 4.2GB** |
| | | | | | savings vs dense | 1x | 76676x | — |
| | | | | | (ii) vs (i) | | | 1.0x |

### Zero-Loss Summary

| Dataset | Depth | Dense (uncapped) | Flat-int (i) | Full-flat (ii) | (i) savings |
|---------|-------|-----------------|-------------|----------------|-------------|
| countries_s3 | 1 | ✅ 138MB | ✅ 16MB | ✅ 16MB | 9x |
| countries_s3 | 2 | ✅ 232MB | ✅ 27MB | ✅ 27MB | 9x |
| countries_s3 | 3 | ✅ 54MB | ✅ 6MB | ✅ 6MB | 9x |
| wn18rr | 1 | ❌ 132.7GB | ✅ 11MB | ✅ 11MB | 12495x |
| wn18rr | 2 | ❌ 199.0GB | ✅ 16MB | ✅ 16MB | 12270x |
| wn18rr | 3 | ❌ 265.4GB | ✅ 22MB | ✅ 22MB | 12161x |
| fb15k237 | 1 | ❌ 108217.2GB | ✅ 1.4GB | ✅ 1.4GB | 76418x |
| fb15k237 | 2 | ❌ 162326.9GB | ✅ 2.1GB | ✅ 2.1GB | 76289x |
| fb15k237 | 3 | ❌ 216436.5GB | ✅ 2.8GB | ✅ 2.8GB | 76225x |

### Key Takeaway

**Without any caps (zero grounding loss):**
- Dense is impossible for fb15k237 at any depth (body_a alone > 100 GB)
- Dense is impossible for wn18rr at depth ≥ 2 (S explodes to Re×K_f=3792)
- **Flat intermediate makes fb15k237 depth 2-3 feasible** at B=192 (~1-4 GB)
- **Flat intermediate makes wn18rr depth 2-3 feasible** at B=192 (~3-40 MB)
- Full-flat gives identical savings (body_a dominates everything)