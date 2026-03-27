# Paper Structure + Figures + Implementation Plan

## 0. Positioning (Core Claim)
EMPCA is a **data-driven, ML-consistent generalization of Optimal Filtering (OF)** that captures realistic signal manifolds while preserving statistical optimality.

---

# 1. Introduction & Problem Setup
**Status:** ✅ Mostly done (Section 1)

### Keep:
- Detector + LAMCAL description
- Signal formation chain
- Simulation pipeline

### Add:
- Clear problem statement:
  - OF assumptions (rank-1, perfect template)
  - Real violations: jitter, shape variation, multi-channel coupling

---

# 2. Optimal Filtering (OF)
**Status:** ✅ Done (Section 2)

### Figures to add:
- OF filter shape in frequency domain
- Example filtered trace + A(t)

### No major new experiments needed

---

# 3. EMPCA Formulation
**Status:** ✅ Done (Section 3)

### Figures to add:
- Basis vectors u_k (time + frequency)
- Explained variance vs k

### Implementation:
```python
# visualize EMPCA basis
U = empca.components_  # (k, d)
plot_time(U[k])
plot_fft(U[k])
```

---

# 4. OF–EMPCA Equivalence (Core Theory)
**Status:** ✅ Strong (Section 4) fileciteturn0file0

### Already done:
- ML unification
- Rank-1 equivalence proof
- Real data verification

### Figures to add:
- Subspace cosine vs dataset
- Amplitude difference histogram

### Keep as theory anchor

---

# 5. Controlled Simulation Benchmarks
**Status:** ⚠️ PARTIAL → NEED EXPANSION

## 5.1 Resolution vs SNR

### Data:
- clean + noisy simulated traces
- noise types: white, pink, MMC

### Experiment:
- vary SNR (scale noise)
- compute:
  - OF amplitude
  - EMPCA (k=1,2,3)

### Figure:
- σ(E) vs SNR

### Pipeline:
```python
for snr in snr_list:
    noisy = inject_noise(clean, snr)
    A_of = of_estimate(noisy)
    A_empca = empca_estimate(noisy)
    resolution.append(std(A - A_true))
```

---

## 5.2 Model Mismatch Study (CRITICAL)

### Vary:
- time jitter
- template distortion

### Figure:
- resolution degradation vs mismatch

### Key claim:
EMPCA degrades slower than OF

---

# 6. Subspace Physics Interpretation
**Status:** ❌ Missing (HIGH VALUE)

### Goal:
Interpret learned basis

### Show:
- u1 ≈ template
- u2 ≈ time derivative
- u3 ≈ shape variation

### Figure:
- overlay u_k with ∂t s

### Pipeline:
```python
u1, u2 = U[0], U[1]
ds_dt = np.gradient(template)
compare(u2, ds_dt)
```

---

# 7. Multi-Channel Structure Learning
**Status:** ⚠️ PARTIAL (theory done, experiment missing)

### Data:
- full 54-channel traces

### Experiments:
1. Correlation heatmap
2. Joint OF vs EMPCA

### Figures:
- channel correlation matrix
- resolution vs #channels

### Pipeline:
```python
cov = np.cov(traces)
plt.imshow(cov)
```

---

# 8. Residual Analysis
**Status:** ❌ Missing

### Compare:
- OF residual
- EMPCA residual

### Figures:
- residual PSD
- autocorrelation

### Pipeline:
```python
res_of = x - of_recon
res_empca = x - empca_recon
plot_psd(res_of)
plot_psd(res_empca)
```

---

# 9. Detection Efficiency (MOST IMPORTANT)
**Status:** ❌ Missing (CRITICAL)

### Data:
- simulated injection (0–20 eV)

### Compare:
- OF trigger
- EMPCA residual threshold
- hybrid

### Figures:
- efficiency vs energy
- ROC curve

### Pipeline:
```python
for E in energies:
    traces = inject_events(noise, E)
    score = compute_trigger_score(traces)
    efficiency[E] = detection_rate(score)
```

---

# 10. Real Data Validation
**Status:** ⚠️ PARTIAL (Section 4.10 done)

### Extend:
- residual distribution tails
- stability across runs

### Figures:
- KS test
- tail comparison

---

# 11. Noise Model Robustness
**Status:** ❌ Missing

### Data:
- real noise samples

### Inject:
- non-Gaussian noise

### Compare:
- OF vs EMPCA

### Key result:
EMPCA more robust to PSD mismatch

---

# 12. Trigger-Level Demonstration
**Status:** ❌ Missing (VERY HIGH IMPACT)

### Pipeline:
```
trace → sliding OF → candidates
      → EMPCA verification
```

### Outputs:
- A(t), χ²(t), projection error

### Figures:
- time trace with detections
- false trigger suppression

---

# 13. Computational Performance
**Status:** ⚠️ PARTIAL (Section 6)

### Measure:
- runtime vs trace length
- scaling vs channels

### Figures:
- latency comparison

---

# 14. Final Key Figures Checklist

### Must include:
- PSD plots (you already noted)
- basis vectors (time + freq)
- resolution vs SNR
- mismatch robustness curves
- residual PSD
- efficiency / ROC curves
- multi-channel correlation heatmap
- trigger trace visualization

---

# 15. Minimal Execution Roadmap

## Phase 1 (1–2 weeks)
- resolution + mismatch

## Phase 2 (1 week)
- subspace interpretation

## Phase 3 (1–2 weeks)
- multi-channel + residual

## Phase 4 (2 weeks)
- efficiency + trigger

## Phase 5 (1 week)
- real data + robustness

---

# Bottom Line

Your current paper:
- Theory: ✅ strong
- Verification: ⚠️ partial
- System impact: ❌ missing

To reach high-quality publication:
👉 Add **efficiency + robustness + trigger validation**

