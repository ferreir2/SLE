# SLE

Solves the Schrodinger-Langevin equation 

$$ i\hbar \partial_t \psi(x,t) = \left[\hat{H} + \hbar A\left(S(x,t) - \int\mathrm{d}x'\ |\psi(x',t)|^2 S(x',t)\right) - \hat{F}_{\text{R}}(t) x\right] \psi(x,t) $$,

with \hat{H} = -(\hbar^2/2m)\partial_x^2 + m\omega^2x^2/2. The term $$ \hat{F}_{\text{R}} $$ is treated as described in
https://arxiv.org/abs/1504.08087 for the colored noise case.

To test, run

```bash
python sle.py test
```

This will generate 10 trajectories with inverse temperature $\beta = 1$ (in units of $k_B = 1$), damping coefficient $A = 0.1$, total time $T = 0.5$ ($m = \omega = \hbar = 1$). The particle and wave function trajectories are saved in the file test.hdf5, together with the relevant parameters.
