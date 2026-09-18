"""Reproducible audit probes; prints JSON and does not modify study outputs.

Run from the repository with .venv/bin/python scripts/audit_short_tracks.py.
These characterize current behavior, rather than bless it as correct.
"""
from pathlib import Path
import json
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

from diffusionkit.classic import compute_all_tamsd, simulate_brownian_tracks
from diffusionkit.classic.api import fit_population as classic_population
from diffusionkit.classic.covariance import step_covariance, tamsd_covariance
from diffusionkit.classic.fitting import (
    fit_all_tracks, localization_offset_by_track, _weighted_nonlinear_gls_solve,
)
from diffusionkit.bayes import fit_track, fit_population
from diffusionkit.bayes.inference import fit_map
from diffusionkit.bayes.likelihood import (
    displacement_covariance, anisotropic_displacement_covariance,
)
from diffusionkit.bayes.nested import dst_displacements, log_likelihood
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def outcome(fn):
    try:
        value = fn()
        return {"returned": str(value)}
    except Exception as exc:
        return {"error": type(exc).__name__, "message": str(exc)}


def main():
    report = {}
    # Independent window-sum construction, not the optimized Toeplitz code.
    max_cov_error = 0.0
    max_port_error = 0.0
    for n in (4, 9, 19):
        for alpha in (0.3, 1.0, 1.7):
            gamma = step_covariance(n, 0.05, 0.033, alpha, 0.025**2)
            max_port_error = max(max_port_error, float(np.max(np.abs(
                gamma - np.asarray(displacement_covariance(n, 0.05, 0.033, alpha, 0.025**2))
            ))))
            windows = [np.array([np.r_[np.zeros(i), np.ones(lag), np.zeros(n-i-lag)]
                                for i in range(n-lag+1)]) for lag in range(1, n+1)]
            direct = np.array([[4*np.mean((a @ gamma @ b.T)**2)
                                for b in windows] for a in windows])
            fast = tamsd_covariance(gamma, np.arange(1, n+1))
            max_cov_error = max(max_cov_error, float(np.max(np.abs(fast-direct))/np.max(direct)))
    report["covariance_checks"] = {"max_relative_window_cov_error": max_cov_error,
                                    "max_absolute_numpy_jax_error": max_port_error}
    rng = np.random.default_rng(10)
    dx, dy = rng.normal(size=(2, 9))*0.05
    Dg, r, psi, sigma, dt = 0.05, 0.6, 0.7, 0.025, 0.033
    dense = np.asarray(anisotropic_displacement_covariance(
        9, Dg*np.cosh(r), np.tanh(r), psi, dt, sigma**2))
    ll_dense = multivariate_normal.logpdf(np.stack([dx, dy], axis=-1).ravel(), cov=dense)
    ll_dst = float(log_likelihood(np.log(Dg), r*np.cos(2*psi), r*np.sin(2*psi),
                                 np.log(sigma), dst_displacements(dx, dy), dt))
    report["anisotropy_dense_vs_dst_loglik_error"] = abs(ll_dense-ll_dst)

    def lognormal_prior_only():
        numpyro.sample("scale", dist.LogNormal(0., 1.))

    def uniform_alpha_gaussian_likelihood():
        unit = numpyro.sample("alpha_unit", dist.Beta(1., 1.))
        alpha = numpyro.deterministic("alpha", 2*unit)
        numpyro.sample("measurement", dist.Normal(alpha, 0.4), obs=jnp.array(0.2))

    report["transformed_mode"] = {
        "lognormal_returned": fit_map(lognormal_prior_only, ()).params["scale"],
        "lognormal_physical_mode": float(np.exp(-1)),
        "uniform_alpha_returned": fit_map(uniform_alpha_gaussian_likelihood, ()).params["alpha"],
        "uniform_alpha_physical_mode_and_mle": 0.2,
    }

    tracks = simulate_brownian_tracks([0.05], 2, 5, dt, sigma, seed=12)
    single = tracks.filter(pl.col("track_id") == 0)
    fit68 = fit_track(single, dt, model="normal", hpdi_prob=0.68)
    fit95 = fit_track(single, dt, model="normal", hpdi_prob=0.95)
    report["map_ignores_interval_probability"] = {
        "identical": fit68.lo == fit95.lo and fit68.hi == fit95.hi,
        "interval_68": [fit68.lo["D"], fit68.hi["D"]],
        "interval_95": [fit95.lo["D"], fit95.hi["D"]],
    }
    gapped = single.with_columns((pl.col("frame")*2).alias("frame"),
                                  (pl.col("t_s")*2).alias("t_s"))
    gapped_fit = fit_track(gapped, dt, model="normal")
    report["gapped_track_silently_accepted"] = {
        "bayes_identical_params": fit68.params == gapped_fit.params,
        "classic_identical_msd": compute_all_tamsd(single, dt).equals(compute_all_tamsd(gapped, dt)),
    }
    report["multiple_ids_accepted_by_fit_track"] = outcome(
        lambda: (lambda f: {"track_id": f.track_id, "track_length": f.track_length,
                            "n_disp": f.n_disp})(fit_track(tracks, dt, model="normal")))
    report["default_five_frame_population_bayes"] = outcome(lambda: fit_population(tracks, dt, show_progress=False))
    report["small_population_classic"] = outcome(lambda: classic_population(tracks, dt, min_track_length=5))

    # Independent two-dimensional numerical integration of the Brownian
    # posterior. The sine basis diagonalizes constant localization noise.
    # Integrate in log coordinates, with Gaussian log-scale prior densities.
    report["brownian_laplace_vs_quadrature"] = []
    sample = simulate_brownian_tracks([0.01], 4, 5, dt, sigma, seed=444)
    for track_id in range(4):
        track = sample.filter(pl.col("track_id") == track_id)
        fit = fit_track(track, dt, model="normal")
        dxy = np.diff(track.select("x_um", "y_um").to_numpy(), axis=0)
        n = len(dxy)
        k = np.arange(1, n+1)
        U = np.sqrt(2/(n+1))*np.sin(np.outer(k, k)*np.pi/(n+1))
        power = np.sum((U @ dxy)**2, axis=1)
        lam = 2-2*np.cos(k*np.pi/(n+1))
        comparisons = []
        for resolution in (700, 1400):
            z = np.linspace(-45, 3, resolution)
            v = np.linspace(-7, -1, resolution//2)
            variance = 2*dt*np.exp(z[:, None, None]) + np.exp(2*v[None, :, None])*lam
            logp = -np.sum(np.log(variance)+0.5*power/variance, axis=-1)
            logp -= 0.5*((z[:, None]-np.log(0.05))/(2*np.log(10)))**2
            logp -= 0.5*((v[None, :]-np.log(sigma))/(1/np.sqrt(10)))**2
            mass = np.exp(logsumexp(logp, axis=1)-logsumexp(logp))
            cdf = np.cumsum(mass)-mass/2
            interval_mass = np.interp(np.log(fit.hi["D"]), z, cdf)-np.interp(np.log(fit.lo["D"]), z, cdf)
            comparisons.append(float(interval_mass))
        report["brownian_laplace_vs_quadrature"].append({
            "track_id": track_id, "N": 5,
            "posterior_mass_in_nominal_68pct_laplace_interval": comparisons[-1],
            "grid_refinement_mass_change": abs(comparisons[1]-comparisons[0]),
        })

    # A negative corrected MSD has no finite optimum in positive K for alpha=0.
    # Check whether the custom optimizer reports its unidentifiable boundary.
    tau = np.arange(1, 4)*dt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _weighted_nonlinear_gls_solve(tau, -np.ones(3), np.eye(3), np.array([0., 1.]))
    report["nonlinear_negative_signal"] = None if result is None else {
        "theta": result[0].tolist(), "stderr": result[1].tolist(), "r_squared": result[2]}

    # Small fixed-seed stress test of the current recommended classical alpha.
    report["short_track_classic"] = []
    for length in (5, 10, 20):
        sim = simulate_brownian_tracks([0.01, 0.05], 100, length, dt, sigma, seed=120+length)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_all_tracks(compute_all_tamsd(sim, dt), min_track_length=5,
                                    localization_offset=localization_offset_by_track(sim))
        result = result.join(sim.select("track_id", "true_D_um2_s").unique(), on="track_id")
        for D in (0.01, 0.05):
            group = result.filter(pl.col("true_D_um2_s") == D)
            a = group["alpha_nlgls_corrected"].to_numpy()
            flag = group["nlgls_corrected_singular"].to_numpy()
            se = group["alpha_nlgls_corrected_stderr"].to_numpy()
            report["short_track_classic"].append({
                "N": length, "D": D, "n": len(a), "median_alpha": float(np.nanmedian(a)),
                "alpha_outside_0_2": int(np.sum((a < 0) | (a > 2))),
                "outside_support_unflagged": int(np.sum(((a < 0) | (a > 2)) & ~flag)),
                "nonfinite_alpha": int(np.sum(~np.isfinite(a))),
                "singular_flag": int(np.sum(flag)),
                "nonfinite_se_unflagged": int(np.sum(~np.isfinite(se) & ~flag)),
                "nominal_1sigma_coverage": float(np.mean(np.abs(a-1) <= se)),
            })
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
