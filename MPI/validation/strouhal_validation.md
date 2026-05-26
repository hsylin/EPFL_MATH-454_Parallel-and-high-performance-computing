# Strouhal Validation Results

Validation target: `St ≈ 0.16` at `Re = 100`.

This validation keeps the physical geometry fixed and only changes the number of MPI ranks. Therefore, it checks whether MPI domain decomposition and halo exchange preserve the numerical result.

| p | Grid | Steps | cyl_r | Diameter D | Probe | St | Status |
|--:|:-----|------:|------:|-----------:|:------|---:|:-------|
| 1 | 800×400 | 60000 | 10 | 20.0 | (280,200) | 0.1600 | OK |
| 2 | 800×400 | 60000 | 10 | 20.0 | (280,200) | 0.1600 | OK |
| 4 | 800×400 | 60000 | 10 | 20.0 | (280,200) | 0.1600 | OK |
| 8 | 800×400 | 60000 | 10 | 20.0 | (280,200) | 0.1600 | OK |
| 16 | 800×400 | 60000 | 10 | 20.0 | (280,200) | 0.1600 | OK |
| 32 | 800×400 | 60000 | 10 | 20.0 | (280,200) | 0.1600 | OK |
| 64 | 800×400 | 60000 | 10 | 20.0 | (280,200) | 0.1600 | OK |
| 128 | 800×400 | 60000 | 10 | 20.0 | (280,200) | 0.1600 | OK |
| 256 | 800×400 | 60000 | 10 | 20.0 | (280,200) | 0.1600 | OK |
