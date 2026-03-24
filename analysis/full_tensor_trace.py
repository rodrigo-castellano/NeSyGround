"""
Full tensor trace: every function in the grounder forward pass,
every tensor it creates, and its memory under 3 approaches.

Traces the actual call chain:
  BCGrounder.forward()
    → init_states()
    for d in range(depth):
      → _select()
      → _resolve() → resolve_enum_step() → resolve_enum()
          → _enumerate_cartesian() / _enumerate_dir()
          → _fill_body_extended()
          → fact_index.exists()
          → _apply_enum_filters()
      → _pack() → pack_states()
      → _postprocess()
          → _postprocess_goals() → compact_atoms()
          → _sync_accumulated()
          → _collect_groundings() → collect_groundings()
    → filter_terminal()
"""

BYTES = 8  # int64

def fmt(b):
    if b == 0: return "—"
    if b < 1e3: return f"{b:.0f}B"
    if b < 1e6: return f"{b/1e3:.0f}KB"
    if b < 1e9: return f"{b/1e6:.0f}MB"
    return f"{b/1e9:.1f}GB"

def trace_grounder(name, B, Re, M, K_f_max, avg_k, depth, width, K_cap):
    G = 1 + depth * max(M - 1, 1)
    G_body = depth * M
    
    print(f"\n{'━'*120}")
    print(f"  {name}   B={B}  Re={Re}  M={M}  K_f={K_f_max}  avg_k={avg_k}  depth={depth}  w={width}  K_cap={K_cap}  G={G}  G_body={G_body}")
    print(f"{'━'*120}")
    
    # Header
    print(f"\n  {'Function':<45} {'Tensor':<30} {'Shape (Dense)':<30} {'Dense':>10} {'Shape (Flat-int)':<30} {'Flat-int':>10} {'Full-flat':>10}")
    print(f"  {'─'*45} {'─'*30} {'─'*30} {'─':─>10} {'─'*30} {'─':─>10} {'─':─>10}")
    
    S = 1
    total_dense = 0
    total_flat_int = 0
    total_full_flat = 0
    
    # ════════════════════════════════════════════
    # init_states()
    # ════════════════════════════════════════════
    rows = [
        ("init_states()", "proof_goals",     f"[{B},1,{G},3]",       B*1*G*3,       f"[{B},1,{G},3]",       B*1*G*3,       B*1*G*3),
        ("",              "grounding_body",  f"[{B},1,{M},3]",       B*1*M*3,       f"[{B},1,{M},3]",       B*1*M*3,       B*1*M*3),
        ("",              "accum_body",      f"[{B},1,{G_body},3]",  B*1*G_body*3,  f"[{B},1,{G_body},3]",  B*1*G_body*3,  B*1*G_body*3),
        ("",              "state_valid",     f"[{B},1]",             B*1,           f"[{B},1]",             B*1,           B*1),
        ("",              "top_ridx",        f"[{B},1]",             B*1,           f"[{B},1]",             B*1,           B*1),
        ("",              "collected_body",  f"[{B},{K_cap},{G_body},3]", B*K_cap*G_body*3, f"[{B},{K_cap},{G_body},3]", B*K_cap*G_body*3, B*0),  # full-flat: empty init
        ("",              "collected_mask",  f"[{B},{K_cap}]",       B*K_cap,       f"[{B},{K_cap}]",       B*K_cap,       B*0),
        ("",              "collected_ridx",  f"[{B},{K_cap}]",       B*K_cap,       f"[{B},{K_cap}]",       B*K_cap,       B*0),
    ]
    for r in rows:
        d_mem = r[3] * BYTES; fi_mem = r[5] * BYTES; ff_mem = r[6] * BYTES
        print(f"  {r[0]:<45} {r[1]:<30} {r[2]:<30} {fmt(d_mem):>10} {r[4]:<30} {fmt(fi_mem):>10} {fmt(ff_mem):>10}")
    
    init_dense = sum(r[3] for r in rows) * BYTES
    init_flat_int = sum(r[5] for r in rows) * BYTES
    init_full_flat = sum(r[6] for r in rows) * BYTES
    print(f"  {'':45} {'INIT TOTAL':<30} {'':30} {fmt(init_dense):>10} {'':30} {fmt(init_flat_int):>10} {fmt(init_full_flat):>10}")
    
    for d in range(depth):
        S_in = S
        N = B * S_in
        G_use_dense = min(K_f_max, K_cap)
        K_enum_dense = min(Re * G_use_dense, K_cap)
        eff_children = min(Re * avg_k, K_cap)
        S_out = min(eff_children, K_cap)
        
        print(f"\n  {'─'*120}")
        print(f"  DEPTH STEP d={d}: S_in={S_in}, N=B×S={N}")
        print(f"  {'─'*120}")
        
        # ════════════════════════════════════════════
        # _select()
        # ════════════════════════════════════════════
        sel_elems = B * S_in * 3
        print(f"  {'_select()':<45} {'queries':<30} {'['+str(B)+','+str(S_in)+',3]':<30} {fmt(sel_elems*BYTES):>10} {'same':<30} {fmt(sel_elems*BYTES):>10} {fmt(sel_elems*BYTES):>10}")
        sel_rem = B * S_in * G * 3
        print(f"  {'':<45} {'remaining':<30} {'['+str(B)+','+str(S_in)+','+str(G)+',3]':<30} {fmt(sel_rem*BYTES):>10} {'same':<30} {fmt(sel_rem*BYTES):>10} {fmt(sel_rem*BYTES):>10}")
        
        # ════════════════════════════════════════════
        # resolve_enum() internals
        # ════════════════════════════════════════════
        
        # _enumerate (fact_index.enumerate)
        enum_dense = N * Re * G_use_dense
        enum_flat = int(N * Re * avg_k)
        print(f"\n  {'resolve_enum()':<45} {'candidates (enumerate)':<30} {'['+str(N)+','+str(Re)+','+str(G_use_dense)+']':<30} {fmt(enum_dense*BYTES):>10} {'['+str(enum_flat)+']':<30} {fmt(enum_flat*BYTES):>10} {fmt(enum_flat*BYTES):>10}")
        print(f"  {'':<45} {'cand_mask':<30} {'['+str(N)+','+str(Re)+','+str(G_use_dense)+']':<30} {fmt(enum_dense*1):>10} {'['+str(enum_flat)+']':<30} {fmt(enum_flat*1):>10} {fmt(enum_flat*1):>10}")
        
        # ★ body_a — THE BIG ONE
        body_dense = N * Re * G_use_dense * M * 3
        body_flat = int(N * Re * avg_k) * M * 3
        shape_d = f"[{N},{Re},{G_use_dense},{M},3]"
        shape_f = f"[{int(N*Re*avg_k)},{M},3]"
        d_mem = body_dense * BYTES
        f_mem = body_flat * BYTES
        star = " ★" if d_mem > 1e9 else ""
        print(f"  {'':<45} {'★ body_a':<30} {shape_d:<30} {fmt(d_mem):>10}{star} {shape_f:<30} {fmt(f_mem):>10} {fmt(f_mem):>10}")
        
        # exists
        exists_dense = N * Re * G_use_dense * M
        exists_flat = int(N * Re * avg_k) * M
        print(f"  {'fact_index.exists()':<45} {'exists':<30} {'['+str(N)+','+str(Re)+','+str(G_use_dense)+','+str(M)+']':<30} {fmt(exists_dense*1):>10} {'['+str(int(N*Re*avg_k))+','+str(M)+']':<30} {fmt(exists_flat*1):>10} {fmt(exists_flat*1):>10}")
        
        # filter mask
        filt_dense = N * Re * G_use_dense
        filt_flat = int(N * Re * avg_k)
        print(f"  {'_apply_enum_filters()':<45} {'grounding_mask':<30} {'['+str(N)+','+str(Re)+','+str(G_use_dense)+']':<30} {fmt(filt_dense*1):>10} {'['+str(filt_flat)+']':<30} {fmt(filt_flat*1):>10} {fmt(filt_flat*1):>10}")
        
        # ════════════════════════════════════════════
        # resolve_enum_step() output
        # ════════════════════════════════════════════
        K_out = min(K_enum_dense, K_cap)
        K_out_fi = min(eff_children, K_cap)
        
        # Dense output: [B, S, K_out, G, 3]
        rg_dense = B * S_in * K_out * G * 3
        rg_flat_int = B * S_in * K_out_fi * G * 3  # same dense shape, smaller K
        rg_full_flat = int(B * S_in * eff_children) * G * 3  # flat
        
        shape_rg_d = f"[{B},{S_in},{K_out},{G},3]"
        shape_rg_fi = f"[{B},{S_in},{K_out_fi},{G},3]"
        shape_rg_ff = f"[{int(B*S_in*eff_children)},{G},3]"
        
        print(f"\n  {'resolve_enum_step() output':<45} {'rule_goals':<30} {shape_rg_d:<30} {fmt(rg_dense*BYTES):>10} {shape_rg_fi:<30} {fmt(rg_flat_int*BYTES):>10} {fmt(rg_full_flat*BYTES):>10}")
        
        rgb_dense = B * S_in * K_out * M * 3
        rgb_flat_int = B * S_in * K_out_fi * M * 3
        rgb_full_flat = int(B * S_in * eff_children) * M * 3
        shape_rgb_d = f"[{B},{S_in},{K_out},{M},3]"
        shape_rgb_fi = f"[{B},{S_in},{K_out_fi},{M},3]"
        shape_rgb_ff = f"[{int(B*S_in*eff_children)},{M},3]"
        print(f"  {'':<45} {'rule_gbody':<30} {shape_rgb_d:<30} {fmt(rgb_dense*BYTES):>10} {shape_rgb_fi:<30} {fmt(rgb_flat_int*BYTES):>10} {fmt(rgb_full_flat*BYTES):>10}")
        
        rs_dense = B * S_in * K_out
        rs_fi = B * S_in * K_out_fi
        rs_ff = int(B * S_in * eff_children)
        print(f"  {'':<45} {'rule_success':<30} {'['+str(B)+','+str(S_in)+','+str(K_out)+']':<30} {fmt(rs_dense*1):>10} {'['+str(B)+','+str(S_in)+','+str(K_out_fi)+']':<30} {fmt(rs_fi*1):>10} {fmt(rs_ff*1):>10}")
        
        # ════════════════════════════════════════════
        # pack_states() output
        # ════════════════════════════════════════════
        pg_elems = B * S_out * G * 3
        gb_elems = B * S_out * M * 3
        ab_elems = B * S_out * G_body * 3
        sv_elems = B * S_out
        
        pg_ff = int(B * S_out) * G * 3  # same count for full-flat (S_out is same)
        
        print(f"\n  {'pack_states() output':<45} {'proof_goals':<30} {'['+str(B)+','+str(S_out)+','+str(G)+',3]':<30} {fmt(pg_elems*BYTES):>10} {'same':<30} {fmt(pg_elems*BYTES):>10} {fmt(pg_ff*BYTES):>10}")
        print(f"  {'':<45} {'grounding_body':<30} {'['+str(B)+','+str(S_out)+','+str(M)+',3]':<30} {fmt(gb_elems*BYTES):>10} {'same':<30} {fmt(gb_elems*BYTES):>10} {fmt(gb_elems*BYTES):>10}")
        print(f"  {'':<45} {'accum_body':<30} {'['+str(B)+','+str(S_out)+','+str(G_body)+',3]':<30} {fmt(ab_elems*BYTES):>10} {'same':<30} {fmt(ab_elems*BYTES):>10} {fmt(ab_elems*BYTES):>10}")
        print(f"  {'':<45} {'state_valid':<30} {'['+str(B)+','+str(S_out)+']':<30} {fmt(sv_elems*1):>10} {'same':<30} {fmt(sv_elems*1):>10} {fmt(sv_elems*1):>10}")
        
        # ════════════════════════════════════════════
        # collect_groundings()
        # ════════════════════════════════════════════
        tG = K_cap
        coll_dense = B * tG * G_body * 3
        coll_fi = coll_dense  # same
        coll_ff = B * min(d * 10 + 10, K_cap) * G_body * 3  # grows
        print(f"\n  {'collect_groundings()':<45} {'collected_body':<30} {'['+str(B)+','+str(tG)+','+str(G_body)+',3]':<30} {fmt(coll_dense*BYTES):>10} {'same':<30} {fmt(coll_fi*BYTES):>10} {fmt(coll_ff*BYTES):>10}")
        
        # ════════════════════════════════════════════
        # STEP TOTAL (peak = all live tensors during resolve)
        # ════════════════════════════════════════════
        # During resolve_enum, these are all live simultaneously:
        # body_a + exists + cand_mask + prev states + collected
        
        resolve_live_dense = (body_dense*BYTES + exists_dense*1 + filt_dense*1 + 
                              pg_elems*BYTES + gb_elems*BYTES + ab_elems*BYTES + 
                              coll_dense*BYTES)
        resolve_live_fi = (body_flat*BYTES + exists_flat*1 + filt_flat*1 +
                           pg_elems*BYTES + gb_elems*BYTES + ab_elems*BYTES +
                           coll_fi*BYTES)
        resolve_live_ff = (body_flat*BYTES + exists_flat*1 + filt_flat*1 +
                           pg_ff*BYTES + gb_elems*BYTES + ab_elems*BYTES +
                           coll_ff*BYTES)
        
        # During pack: resolve output + prev states + collected
        pack_live_dense = (rg_dense*BYTES + rgb_dense*BYTES + rs_dense*1 +
                           pg_elems*BYTES + ab_elems*BYTES + coll_dense*BYTES)
        pack_live_fi = (rg_flat_int*BYTES + rgb_flat_int*BYTES + rs_fi*1 +
                        pg_elems*BYTES + ab_elems*BYTES + coll_fi*BYTES)
        pack_live_ff = (rg_full_flat*BYTES + rgb_full_flat*BYTES + rs_ff*1 +
                        pg_ff*BYTES + ab_elems*BYTES + coll_ff*BYTES)
        
        step_peak_dense = max(resolve_live_dense, pack_live_dense)
        step_peak_fi = max(resolve_live_fi, pack_live_fi)
        step_peak_ff = max(resolve_live_ff, pack_live_ff)
        
        total_dense = max(total_dense, step_peak_dense)
        total_flat_int = max(total_flat_int, step_peak_fi)
        total_full_flat = max(total_full_flat, step_peak_ff)
        
        dominant_d = "body_a" if body_dense*BYTES > pack_live_dense/2 else "rule_goals"
        dominant_fi = "body_a" if body_flat*BYTES > pack_live_fi/2 else "rule_goals"
        
        print(f"\n  {'STEP PEAK (resolve phase)':<45} {'':30} {'':30} {fmt(resolve_live_dense):>10} {'':30} {fmt(resolve_live_fi):>10} {fmt(resolve_live_ff):>10}")
        print(f"  {'STEP PEAK (pack phase)':<45} {'':30} {'':30} {fmt(pack_live_dense):>10} {'':30} {fmt(pack_live_fi):>10} {fmt(pack_live_ff):>10}")
        print(f"  {'STEP PEAK (max)':<45} {'':30} {'':30} {fmt(step_peak_dense):>10} {'':30} {fmt(step_peak_fi):>10} {fmt(step_peak_ff):>10}")
        print(f"  {'Bottleneck tensor':<45} {'':30} {'':30} {dominant_d:>10} {'':30} {dominant_fi:>10} {'':>10}")
        
        S = S_out
    
    # ════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════
    fits_d = total_dense < 20e9
    fits_fi = total_flat_int < 20e9
    fits_ff = total_full_flat < 20e9
    
    print(f"\n  {'═'*120}")
    print(f"  OVERALL PEAK MEMORY:")
    print(f"    Dense:           {fmt(total_dense):>10}  {'✓ FITS' if fits_d else '✗ OOM'}")
    print(f"    Flat-intermediate: {fmt(total_flat_int):>10}  {'✓ FITS' if fits_fi else '✗ OOM'}")
    print(f"    Full-flat:       {fmt(total_full_flat):>10}  {'✓ FITS' if fits_ff else '✗ OOM'}")
    if total_dense > 0:
        print(f"    Flat-int savings: {total_dense/max(total_flat_int,1):.0f}x less memory")
        print(f"    Full-flat savings: {total_dense/max(total_full_flat,1):.0f}x less memory")
        print(f"    Full-flat vs flat-int: {total_flat_int/max(total_full_flat,1):.1f}x additional savings")


# ════════════════════════════════════════════
# RUN ALL CASES
# ════════════════════════════════════════════

print("█" * 120)
print("  COMPLETE TENSOR TRACE: Every function, every tensor, every byte")
print("█" * 120)

configs = [
    ("fb15k237 depth=2", 192, 30, 2, 3612, 13, 2, 1, 256),
    ("fb15k237 depth=3", 192, 30, 2, 3612, 13, 3, 1, 256),
    ("fb15k237 depth=4", 192, 30, 2, 3612, 13, 4, 1, 256),
    ("wn18rr depth=2",   192,  8, 2,  474,  4, 2, 1, 256),
    ("wn18rr depth=3",   192,  8, 2,  474,  4, 3, 1, 256),
    ("wn18rr depth=4",   192,  8, 2,  474,  4, 4, 1, 256),
    ("countries_s3 d=2", 768,  3, 3,   16,  5, 2, 1,  64),
]

for args in configs:
    trace_grounder(*args)
