"""
A simple script to generate trajectories via the Schrodinger-Langevin equation 
for a harmonic oscillator
"""

import argparse
import sys
import numpy as np
from scipy import special, integrate, interpolate, optimize
from scipy.linalg.blas import zgemv
import math
import h5py
import matplotlib.pyplot as plt

def solve_SLE(N_RUNS, A, BETA, T):

    # spatial grid
    L = 20
    NX = 200
    X_GRID = np.linspace(-L/2, L/2, NX)
    DX = L / (NX - 1)

    # time grid
    DT = 0.05
    T_GRID = np.arange(0, T, DT)
    NT = len(T_GRID)

    # defines the truncation length in the inner product 
    # to calculate the colored noise
    NT_NOISE = int(7/DT)

    # Hamiltonian
    X = np.diag(X_GRID[1:-1])
    P2 = -0.5 / DX**2 * (np.diag(-2 * np.ones(NX-2))
                        + np.diag(np.ones(NX-3), k=1)
                        + np.diag(np.ones(NX-3), k=-1))
    V = 0.5 * X**2
    H0 = P2 + V


    B = np.sqrt(A * (np.cosh(0.5 * BETA)/np.sinh(0.5 * BETA) - 1))

    params = {'A': A, 'B': B, 'm':1, 'hbar':1, 'omega':1, 'beta':BETA, 'NT_NOISE': NT_NOISE}

    def eigen(n, x):
        """ 
        Returns the nth eigenfunction of the harmonic oscillator
        evaluated at points x 
        """
        hn = special.eval_hermite(n, x)
        prefactor = np.sqrt(1 / (2**n * math.factorial(n))) * \
            (1 / np.pi)**0.25
        return prefactor * np.exp(- 0.5 * x**2) * hn


    def S(psi, grid=X_GRID[1:-1], grid_size=NX-2):
        """
        Evaluates the phase of the wavefunction psi on a given grid
        """
        s = np.empty(grid_size)
        ds = np.angle(psi[1:]/psi[:-1])
        s[1:] = s[0] + np.cumsum(ds)
        # for i in range(1, grid_size):
        #     s[i] = s[i-1] + np.angle(psi[i]/psi[i-1])

        return s - integrate.trapz(x=grid, y=s*(np.abs(psi)**2))


    def power_spec(gamma, beta, w):
        return 2 * gamma * w / (np.exp(beta*w) - 1)


    def correlationW(gamma, beta, t, eps=1e-12, ulim=100, nrec=5000):
        """ """
        return integrate.quad(lambda w: np.sqrt(power_spec(gamma, beta, w)) *
                            np.cos(w*t)/np.pi, eps, ulim, limit=nrec)


    def gen_ws(gamma, beta):
        """ Calculates the ws needed to generate colored noise """
        # Time points for the noise
        ts = DT * np.arange(-NT_NOISE+1, NT_NOISE)
        # weights
        ws = [correlationW(gamma, beta, t)[0] for t in ts]
        
        return ts, ws


    def generate_noise(ws, gamma, beta, nsamples):
        """ Generates the noise values """
        
        nrand = 2*NT_NOISE + NT - 2
        fr = np.empty((nsamples, NT))
        for s in range(nsamples):
            rs = np.random.normal(loc=0, scale=np.sqrt(DT), size=nrand)
            fr[s, :] = [np.dot(rs[i:i+2*NT_NOISE-1], ws) for i in range(NT)]

        return fr


    def autocorr(noise):
        """ Autocorrelation function for a given sample of the noise """
        nt = noise.shape[1]
        return [np.mean(noise[:, 0]*noise[:, i]) for i in range(nt)]


    def evolve_x(x, psi_in, psi_out):
        """
        Evolves trajectory using the 'conservation of volume trick'.
        """
        cdf_in = integrate.cumtrapz(x=X_GRID, y=np.hstack([0, np.abs(psi_in)**2, 0]), initial=0)
        cdf_in_interp = interpolate.interp1d(X_GRID, cdf_in)
        cdf_out = integrate.cumtrapz(x=X_GRID, y=np.hstack([0, np.abs(psi_out)**2, 0]), initial=0)
        cdf_out_interp = interpolate.interp1d(X_GRID, cdf_out)
        vol_in = cdf_in_interp(x)
        new_x = optimize.root(lambda q: vol_in - cdf_out_interp(q), x)['x'][0]

        return new_x


    def evolve1step(h, psi, dt, x=None):
        M = len(psi)
        a = np.eye(NX-2) + 0.5j * h * dt
        b = zgemv(1, np.eye(NX-2) - 0.5j * h * dt, psi)
        sol = np.linalg.solve(a, b)
        x_new = x
        if x is not None:
            x_new = evolve_x(x, psi, sol)
        return sol, x_new


    def draw_initial_positions(dist, grid, n):
        """
        Approximates the continuous distribution by a discrete one.
        Works as long as the grid is sufficiently fine such that
        the distribution function is approx. constant inside the 
        grid points.
        """
        cdist = integrate.cumtrapz(x=grid, y=dist, initial=0)
        ps = cdist[1:] - cdist[:-1]
        bins_centers = 0.5 * (grid[1:] + grid[:-1])
        bins_widths = grid[1:] - grid[:-1]
        nbins = len(bins_widths)
        
        bins_sample = np.random.choice(range(nbins), n, p=ps)
        noise_sample = np.random.rand(n) - 0.5

        xs = [bins_centers[bin_] + r * bins_widths[bin_] for bin_, r in zip(bins_sample, noise_sample)]

        return xs



    ts, ws = gen_ws(A, BETA)
    sols = []
    trajs = []

    init_psi = eigen(0, X_GRID)
    init_dist = np.conj(init_psi) * init_psi
    init_pos = draw_initial_positions(init_dist, X_GRID, N_RUNS)

    noise_samples = generate_noise(ws, A, BETA, N_RUNS)

    for run in range(N_RUNS):

        sol = np.empty((NX-2, NT), dtype='complex')

        traj = [init_pos[run]]
        sol[:, 0] = init_psi[1:-1]
        noise = noise_samples[run]

        for i, r in zip(range(1, NT), noise):
            s = A * np.diag(S(sol[:, i-1]))
            fr = -r * X
            h = H0 + s + fr
            sol[:, i], x = evolve1step(h, sol[:, i-1], DT, x=traj[-1])
            traj.append(x)

        sols.append(sol)
        trajs.append(traj)

    return X_GRID, T_GRID, sols, trajs, params

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Schrodinger-Langevin equation solver')
    parser.add_argument('NAME', type=str, help='Name for the hdf5 file')
    parser.add_argument('--N_RUNS', type=int, help='Number of simulations', default=10)
    parser.add_argument('--BETA', type=float, help='Inverse temperature beta = 1/T', default=1)
    parser.add_argument('--A', type=float, help='Dissipation constant in SL equation', default=0.1)
    parser.add_argument('--T', type=float, help='Final time (Initial time = 0)', default=0.5)

    args = parser.parse_args()

    x_grid, t_grid, sols, trajs, params = solve_SLE(args.N_RUNS, args.A, args.BETA, args.T)


    # Save to file
    with h5py.File(args.NAME + ".hdf5", "w") as f:
        grp = f.create_group("data")

        grp["sols"] = sols
        grp["trajs"] = trajs
        grp["T_GRID"] = t_grid
        grp['X_GRID'] = x_grid

        grp.update(params)

