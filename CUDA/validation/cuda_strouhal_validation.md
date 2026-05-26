# CUDA Strouhal Validation Results

Validation target: `St ≈ 0.16` at `Re = 100`.

This validation keeps the physical geometry fixed and only changes CUDA launch/memory-layout parameters. Therefore, it checks whether the CUDA implementation preserves the numerical result for the same physical problem.

| Grid | Steps | Block | Pitch align | cyl_r | Diameter D | Probe | probe_every | St | Status |
|:-----|------:|:------|------------:|------:|-----------:|:------|------------:|---:|:-------|
| 800×400 | 60000 | 16×16 | 1 | 10 | 20.0 | (280,200) | 1 | 0.1600 | OK |
| 800×400 | 60000 | 16×16 | 32 | 10 | 20.0 | (280,200) | 1 | 0.1600 | OK |
| 800×400 | 60000 | 16×16 | 64 | 10 | 20.0 | (280,200) | 1 | 0.1600 | OK |
| 800×400 | 60000 | 32×8 | 1 | 10 | 20.0 | (280,200) | 1 | 0.1600 | OK |
| 800×400 | 60000 | 32×8 | 32 | 10 | 20.0 | (280,200) | 1 | 0.1600 | OK |
| 800×400 | 60000 | 32×8 | 64 | 10 | 20.0 | (280,200) | 1 | 0.1600 | OK |
| 800×400 | 60000 | 64×4 | 1 | 10 | 20.0 | (280,200) | 1 | 0.1600 | OK |
| 800×400 | 60000 | 64×4 | 32 | 10 | 20.0 | (280,200) | 1 | 0.1600 | OK |
| 800×400 | 60000 | 64×4 | 64 | 10 | 20.0 | (280,200) | 1 | 0.1600 | OK |
