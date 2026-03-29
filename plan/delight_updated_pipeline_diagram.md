# Updated DELight Reconstruction / Trigger Pipeline

```text
long waveform stream
   │
   ├── Sliding OF scan ───────────────> A(t), χ²(t), OF candidate peaks
   │
   ├── ResNet / TCN trigger backbone -> feature map over time
   │                                    + trigger head -> score(t), NN candidate peaks
   │
   └──────────────────────────────────> candidate merge
                                           (threshold / local maxima /
                                            priority merge / NMS / deadtime)
                                           │
                                           ▼
                                   candidate windows {x_cand}
                                           │
                ┌──────────────────────────┼───────────────────────────────┬───────────────────────────────┐
                │                          │                               │                               │
                ▼                          ▼                               ▼                               ▼
         local OF fit              EMPCA verification      Transformer candidate encoder         optional SBI /
      (A, t0, χ², width,         (subspace residual,      (geometry-aware multi-channel         uncertainty refinement
       baseline features)          coeffs, consistency)     reconstruction from candidate)       (optional late stage)
                                                                       │
                                                                       ├── position (x, y[, z])
                                                                       ├── energy
                                                                       ├── class probability
                                                                       ├── latent representation z_tr
                                                                       ├── uncertainty / confidence
                                                                       └── masking diagnostics
                                                                           (top/bottom channel tests)
                └──────────────────────────┬───────────────────────────────┴───────────────────────────────┘
                                           ▼
                                  fusion / decision layer
                         (small MLP main model; XGBoost benchmark;
                          rules / logistic as baselines)
                                           │
                        decision policy: reject / accept / escalate / label
                                           │
                                           ▼
                                 final trigger / inference
                                 (event record with t0, energy,
                                  position, scores, uncertainty,
                                  provenance)
```

## Role of each branch

### 1. Sliding OF scan
Use as the first analytic proposal source.

Outputs:
- time-local amplitude statistic `A(t)`
- goodness-of-fit / mismatch statistic `χ²(t)`
- candidate peaks from thresholding and local maxima

Why it stays:
- physically grounded
- cheap enough for long traces
- gives interpretable amplitude / timing / template-consistency information

### 2. Lightweight trigger net
Use a small learned model for proposal generation only.

Recommended implementation now:
- **ResNet trigger backbone + time-local trigger head**
- practical alternative to benchmark later: **TCN / dilated CNN**

Role:
- high-recall learned proposal stream
- complements OF when waveform shape deviates from ideal templates
- cheaper than the Transformer branch
- outputs a time-local score trace `score(t)` rather than a single global class

Practical note:
- the current ResNet-style approach should be treated as **feature extraction + trigger head**
- global pooling should be avoided in the trigger branch because it destroys timing information

### 3. Candidate merge
Combine proposal sources before expensive reconstruction.

Inputs:
- OF proposal statistics such as `A(t)`, `χ²(t)`, and OF peak times
- trigger-net score trace `score(t)` and NN peak times

Typical operations:
- thresholding
- local maxima selection
- priority merge of OF and NN proposals
- non-maximum suppression (NMS)
- deadtime / overlap handling

Role:
- convert raw proposal traces into a clean set of candidate windows
- suppress duplicates from nearby peaks or multiple branches firing on the same event
- attach preliminary metadata to each candidate before downstream verification

Output:
- candidate windows `{x_cand}` with metadata such as center time, source branch, and preliminary scores

### 4. Local OF fit
Run a more precise fit only on candidate windows.

Useful candidate features:
- best-fit amplitude
- refined `t0`
- local `χ²`
- peak width / rise / decay shape summaries
- baseline context

### 5. EMPCA verification
Keep this as the interpretable linear-subspace check.

Role:
- test whether the candidate lies in the learned signal subspace
- compute residual / consistency features
- provide a physically transparent waveform-quality score

### 6. Transformer candidate encoder
This is the main structured reconstruction module.

Best role in DELight:
- candidate-level multi-channel inference
- geometry-aware interpretation across sensor channels
- position and energy reconstruction
- secondary classification signal, especially when useful information exists

Why it belongs here instead of the streaming front end:
- it is more expensive than OF / small CNN proposals
- it is designed around candidate-level multi-channel reasoning
- current evidence suggests it is strongest for position / energy / geometry, not first-pass triggering

### 7. Optional SBI / uncertainty refinement
Use only for ambiguous or escalated candidates.

Possible role:
- posterior refinement
- uncertainty-aware parameter inference
- late-stage expensive disambiguation

### 8. Fusion / decision layer
Fuse analytic, linear-subspace, and learned structured features.

Inputs can include:
- OF features
- EMPCA residuals / coefficients
- Transformer outputs
- uncertainty / confidence measures

Recommended choice:
- **small MLP as the main fusion model**
- **XGBoost as the strongest tabular benchmark**
- rules / logistic regression as transparent baselines

Why MLP is a good main choice:
- smooth nonlinear fusion of heterogeneous candidate features
- easy to extend to multi-output decisions
- natural fit if the rest of the pipeline already contains learned components

Why keep XGBoost:
- often very strong on engineered tabular candidate features
- useful performance benchmark against the MLP

### 9. Final decision policy
This stage should be implemented as a realistic decision policy, not as another large model.

Recommended functionality:
- **reject** obvious noise-like candidates
- **accept** strong candidates
- **escalate** ambiguous candidates to optional late-stage refinement
- **label** the final accepted / rejected candidate state

Practical implementation:
- use score thresholds or calibrated probabilities from the fusion model
- define a reject region, accept region, and escalation band in between
- store a structured event record for accepted candidates

Example final event content:
- `t0`
- amplitude / energy
- position
- OF and EMPCA summary features
- Transformer outputs
- final score
- uncertainty
- provenance of which branch proposed / accepted the event

## Design logic for DELight

This pipeline separates the problem into two different regimes:

### Cheap full-stream detection
Handled by:
- Sliding OF
- lightweight trigger net

Question answered:
> Is there something pulse-like here at all?

### Expensive candidate-level interpretation
Handled by:
- local OF fit
- EMPCA verification
- Transformer candidate encoder

Question answered:
> What kind of event is this across all channels, and what can we reconstruct from it?

That split is important for DELight because the experiment combines:
- long noisy traces
- rare near-threshold signals
- multi-channel structured geometry
- different tasks with different intrinsic difficulty

In particular:
- **OF** is strong for proposal statistics and local fit
- **EMPCA** is strong for interpretable subspace consistency
- **Transformer** is strongest as a geometry-aware candidate reconstruction model
- **classification near threshold** may remain weak if the underlying physical discrimination is weak

## Compact version for slides

```text
Waveform
  ├─ Sliding OF ────────────────┐
  ├─ Light trigger net ─────────┤→ candidate merge → candidate windows
  └─────────────────────────────┘
                                   ├─ local OF fit
                                   ├─ EMPCA verification
                                   ├─ Transformer reconstruction
                                   └─ optional SBI refinement
                                                ↓
                                        fusion / decision
                                                ↓
                                     final trigger / inference
```

## One-sentence summary

A hybrid DELight pipeline should use **OF and a lightweight trigger model for fast candidate proposal**, then apply **EMPCA and a geometry-aware Transformer on candidate windows for interpretable verification and structured multi-channel reconstruction**.

