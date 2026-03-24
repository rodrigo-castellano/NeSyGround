"""
Max depth analysis: every tensor, every step, exact memory.

For each depth step, lists EVERY tensor allocated with its shape and bytes.
Computes peak memory = max across all steps of (sum of live tensors at that step).

Approaches:
  (i)   Flat intermediate: body_a flat, everything else dense
  (ii)  Full flat: everything flat+offsets (zero padding waste)
  (iii) Dense: current implementation (all padded)
"""

BYTES = 8  # int64
FBYTES = 4  # float32

def fmt_mem(b):
    if b < 1e3: return f"{b:.0f}B"
    if b < 1e6: return f"{b/1e3:.1f}KB"
    if b < 1e9: return f"{b/1e6:.1f}MB"
    return f"{b/1e9:.2f}GB"

def analyze(name, B, Re, M, K_f_max, avg_k, max_k, depth, width, K_cap):
    G = 1 + depth * max(M - 1, 1)
    G_body = depth * M
    M_work = M  # working buffer for current depth body
    
    print(f"\n{'='*80}")
    print(f"  {name}  B={B} Re={Re} M={M} depth={depth} width={width} K_cap={K_cap}")
    print(f"  K_f_max={K_f_max} avg_k={avg_k} max_k={max_k} G={G} G_body={G_body}")
    print(f"{'='*80}")
    
    # Track S at each depth step
    # Step 0: S_in=1
    # After step d: S_out = min(children_found, K_cap)
    
    for approach in ['(iii) Dense', '(i) Flat-intermediate', '(ii) Full-flat']:
        print(f"\n  --- {approach} ---")
        
        S = 1  # initial
        peak_mem = 0
        peak_step = -1
        
        for d in range(depth):
            print(f"\n  Step {d}/{depth-1}: S_in={S}")
            N = B * S  # flattened queries for resolve_enum
            
            tensors = []
            
            # ──────────── ENUMERATION (inside resolve_enum) ────────────
            
            if approach == '(iii) Dense':
                # body_a: [N, Re, K_f_max, M, 3] — padded to dataset K_f
                G_use = min(K_f_max, K_cap)  # capped
                shape = f"[{N}, {Re}, {G_use}, {M}, 3]"
                mem = N * Re * G_use * M * 3 * BYTES
                tensors.append(("body_a (enum intermediate)", shape, mem))
                
                # exists: [N, Re, G_use, M] bool
                mem_ex = N * Re * G_use * M * 1
                tensors.append(("exists mask", f"[{N}, {Re}, {G_use}, {M}]", mem_ex))
                
                # candidate mask: [N, Re, G_use]
                tensors.append(("cand_mask", f"[{N}, {Re}, {G_use}]", N*Re*G_use*1))
                
                # Effective children per state
                eff_children = min(Re * G_use, K_cap)
                
            elif approach == '(i) Flat-intermediate':
                # body_a: flat [total_valid_bodies, M, 3]
                total_valid = N * Re * avg_k  # only valid entries
                shape = f"[{int(total_valid)}, {M}, 3]"
                mem = int(total_valid) * M * 3 * BYTES
                tensors.append(("body_a FLAT", shape, mem))
                
                # exists: flat [total_valid, M] bool
                mem_ex = int(total_valid) * M * 1
                tensors.append(("exists mask flat", f"[{int(total_valid)}, {M}]", mem_ex))
                
                # offsets: [N*Re + 1]
                tensors.append(("offsets", f"[{N*Re+1}]", (N*Re+1)*BYTES))
                
                eff_children = min(Re * avg_k, K_cap)
                
            else:  # Full flat
                # Same as flat-intermediate for body_a
                total_valid = N * Re * avg_k
                shape = f"[{int(total_valid)}, {M}, 3]"
                mem = int(total_valid) * M * 3 * BYTES
                tensors.append(("body_a FLAT", shape, mem))
                mem_ex = int(total_valid) * M * 1
                tensors.append(("exists mask flat", f"[{int(total_valid)}, {M}]", mem_ex))
                tensors.append(("offsets", f"[{N*Re+1}]", (N*Re+1)*BYTES))
                
                eff_children = min(Re * avg_k, K_cap)
            
            # ──────────── RESOLVE_ENUM_STEP OUTPUT ────────────
            
            if approach == '(ii) Full-flat':
                # rule_goals: flat [total_children, G, 3]
                total_ch = B * S * eff_children
                tensors.append(("rule_goals flat", f"[{total_ch}, {G}, 3]", total_ch*G*3*BYTES))
                tensors.append(("rule_gbody flat", f"[{total_ch}, {M_work}, 3]", total_ch*M_work*3*BYTES))
                tensors.append(("rule_success flat", f"[{total_ch}]", total_ch*1))
                tensors.append(("sub_rule_idx flat", f"[{total_ch}]", total_ch*BYTES))
            else:
                # Dense: [B, S, K_cap, G, 3]
                K_out = min(eff_children, K_cap)
                tensors.append(("rule_goals", f"[{B}, {S}, {K_out}, {G}, 3]", B*S*K_out*G*3*BYTES))
                tensors.append(("rule_gbody", f"[{B}, {S}, {K_out}, {M_work}, 3]", B*S*K_out*M_work*3*BYTES))
                tensors.append(("rule_success", f"[{B}, {S}, {K_out}]", B*S*K_out*1))
                tensors.append(("sub_rule_idx", f"[{B}, {S}, {K_out}]", B*S*K_out*BYTES))
            
            # ──────────── PACK_STATES OUTPUT ────────────
            
            S_out = min(eff_children, K_cap)  # new S after pack
            
            if approach == '(ii) Full-flat':
                total_states = B * S_out
                tensors.append(("proof_goals flat", f"[{total_states}, {G}, 3]", total_states*G*3*BYTES))
                tensors.append(("grounding_body flat", f"[{total_states}, {M_work}, 3]", total_states*M_work*3*BYTES))
                tensors.append(("accum_body flat", f"[{total_states}, {G_body}, 3]", total_states*G_body*3*BYTES))
                tensors.append(("state_valid flat", f"[{total_states}]", total_states*1))
                tensors.append(("top_ridx flat", f"[{total_states}]", total_states*BYTES))
            else:
                tensors.append(("proof_goals", f"[{B}, {S_out}, {G}, 3]", B*S_out*G*3*BYTES))
                tensors.append(("grounding_body", f"[{B}, {S_out}, {M_work}, 3]", B*S_out*M_work*3*BYTES))
                tensors.append(("accum_body", f"[{B}, {S_out}, {G_body}, 3]", B*S_out*G_body*3*BYTES))
                tensors.append(("state_valid", f"[{B}, {S_out}]", B*S_out*1))
                tensors.append(("top_ridx", f"[{B}, {S_out}]", B*S_out*BYTES))
            
            # ──────────── COLLECTED GROUNDINGS (accumulates across steps) ────────────
            
            tG = K_cap  # collected buffer
            if approach == '(ii) Full-flat':
                avg_collected = min(d * 10 + 10, K_cap)  # grows with depth
                tensors.append(("collected_body flat", f"[{B*avg_collected}, {G_body}, 3]", B*avg_collected*G_body*3*BYTES))
                tensors.append(("collected_mask flat", f"[{B*avg_collected}]", B*avg_collected*1))
                tensors.append(("collected_ridx flat", f"[{B*avg_collected}]", B*avg_collected*BYTES))
            else:
                tensors.append(("collected_body", f"[{B}, {tG}, {G_body}, 3]", B*tG*G_body*3*BYTES))
                tensors.append(("collected_mask", f"[{B}, {tG}]", B*tG*1))
                tensors.append(("collected_ridx", f"[{B}, {tG}]", B*tG*BYTES))
            
            # ──────────── SUM ────────────
            
            step_total = sum(m for _, _, m in tensors)
            if step_total > peak_mem:
                peak_mem = step_total
                peak_step = d
            
            for tname, tshape, tmem in tensors:
                pct = tmem / step_total * 100
                bar = '█' * int(pct / 3)
                if pct > 5:
                    print(f"    {tname:<28s} {tshape:<35s} {fmt_mem(tmem):>10s} ({pct:4.1f}%) {bar}")
            
            print(f"    {'─'*80}")
            print(f"    STEP {d} TOTAL: {fmt_mem(step_total):>10s}")
            
            S = S_out  # next step's S_in
        
        print(f"\n  ★ PEAK MEMORY: {fmt_mem(peak_mem)} (at step {peak_step})")
        fits = peak_mem < 20e9
        print(f"  ★ FITS 24GB GPU (20GB available): {'YES ✓' if fits else 'NO ✗'}")


# ── Run analysis ──

print("█" * 80)
print("  EXACT TENSOR-BY-TENSOR MEMORY ANALYSIS")
print("  Every tensor, every step, every byte")
print("█" * 80)

# fb15k237 depth 2
analyze("fb15k237", B=192, Re=30, M=2, K_f_max=3612, avg_k=13, max_k=3612,
        depth=2, width=1, K_cap=256)

# fb15k237 depth 3
analyze("fb15k237", B=192, Re=30, M=2, K_f_max=3612, avg_k=13, max_k=3612,
        depth=3, width=1, K_cap=256)

# wn18rr depth 2
analyze("wn18rr", B=192, Re=8, M=2, K_f_max=474, avg_k=4, max_k=474,
        depth=2, width=1, K_cap=256)

# wn18rr depth 3
analyze("wn18rr", B=192, Re=8, M=2, K_f_max=474, avg_k=4, max_k=474,
        depth=3, width=1, K_cap=256)
