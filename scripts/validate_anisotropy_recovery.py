"""Ground-truth checks for `diffusionkit.bayes.nested` (anisotropy evidence
by nested sampling). Everything is per-track: no pooling anywhere.

  check 1  the DST-I likelihood is exactly the dense one
  check 2  isotropy reduction: h=(0,0) is the isotropic model, bit for bit
  check 3  null calibration -- false-positive rate vs. track length
  check 4  detectability -- can a single track resolve its own anisotropy?
  check 5  eps/psi recovery where the question is answerable

Checks 1-2 are exact and instant. Checks 3-5 each run a nested sampler twice
per simulated track (~3-5 s per track), so the whole script takes roughly
20-30 minutes; `--quick` cuts the replicate counts.

Check 4 is the one that matters for interpretation: it establishes that a
single short track *cannot* answer the question and correctly reports so,
which is why `bayes.anisotropy` gates nothing on track length and reports
evidence rather than a flag.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import polars as pl

from diffusionkit.bayes.likelihood import (
    anisotropic_displacement_covariance,
    displacement_covariance,
)
from diffusionkit.bayes.nested import (
    LogEuclideanAnisotropicPrior,
    dst_displacements,
    fit_track_nested,
    log_likelihood,
)
from diffusionkit.bayes.simulate import simulate_anisotropic_tracks

WORKFLOW = "validate_anisotropy_recovery"
TABLE_DIR = REPO_ROOT / "results" / "tables" / WORKFLOW
DT_S = 0.033
D_MEAN = 0.05
SIGMA_LOC = 0.025


def _tracks_to_arrays(sim: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(n_tracks, n_disp) dx/dy. One group_by pass, not one scan per track."""
    g = (
        sim.sort("track_id", "frame").group_by("track_id", maintain_order=True)
        .agg(pl.col("x_um"), pl.col("y_um"))
    )
    x = np.stack(g["x_um"].to_list())
    y = np.stack(g["y_um"].to_list())
    return np.diff(x, axis=1), np.diff(y, axis=1)


def check_1_dst_equals_dense() -> None:
    """The DST-I block-diagonal likelihood == the dense (2n,2n) MVN.

    `nested.log_likelihood` evaluates n independent 2x2 blocks instead of
    factorizing a (2n,2n) matrix. That is an exact identity, not an
    approximation, and this is where it is checked against
    `likelihood.anisotropic_displacement_covariance` -- the dense reference
    implementation the rest of the package no longer calls.
    """
    print("\ncheck 1: DST-I likelihood vs dense MVN")
    rng = np.random.default_rng(0)
    worst = 0.0
    for n_disp in (4, 9, 19, 49):
        for (u, h1, h2, v) in [(np.log(0.05), 0.0, 0.0, np.log(0.025)),
                               (np.log(0.05), 0.4, -0.2, np.log(0.025)),
                               (np.log(0.2), -0.6, 0.35, np.log(0.01))]:
            dx, dy = rng.standard_normal((2, n_disp)) * 0.05
            r = np.hypot(h1, h2)
            cov = anisotropic_displacement_covariance(
                n_disp, np.exp(u) * np.cosh(r), np.tanh(r),
                0.5 * np.arctan2(h2, h1) % np.pi, DT_S, np.exp(2 * v)
            )
            d = jnp.stack([jnp.asarray(dx), jnp.asarray(dy)], -1).reshape(-1)
            dense = float(dist.MultivariateNormal(
                jnp.zeros(2 * n_disp), covariance_matrix=cov).log_prob(d))
            fast = float(log_likelihood(u, h1, h2, v, dst_displacements(dx, dy), DT_S))
            worst = max(worst, abs(dense - fast))
    print(f"   max |dense - DST| over 12 cases = {worst:.3e}")
    assert worst < 1e-8, "DST-I block-diagonalization is not exact"
    print("   PASS -- exact to numerical precision")


def check_2_isotropy_reduction() -> None:
    """h=(0,0) must be exactly the isotropic normal-diffusion model."""
    print("\ncheck 2: h=(0,0) reduces to isotropic diffusion")
    worst = 0.0
    for n_disp in (4, 19):
        iso = np.asarray(displacement_covariance(n_disp, D_MEAN, DT_S, 1.0, SIGMA_LOC**2))
        aniso = np.asarray(anisotropic_displacement_covariance(
            n_disp, D_MEAN, 0.0, 1.234, DT_S, SIGMA_LOC**2))
        # the anisotropic form interleaves (dx,dy); its x-x sub-block is the isotropic matrix
        worst = max(worst, np.abs(aniso[0::2, 0::2] - iso).max(),
                    np.abs(aniso[1::2, 1::2] - iso).max(),
                    np.abs(aniso[0::2, 1::2]).max())
    print(f"   max deviation (any psi) = {worst:.3e}")
    assert worst < 1e-12, "eps=0 does not reduce to the isotropic model"
    print("   PASS -- H0 is nested exactly inside H1")


def _log_bfs(eps: float, track_length: int, n_rep: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psis = rng.uniform(0, np.pi, n_rep)
    out = []
    for i, psi in enumerate(psis):
        sim = simulate_anisotropic_tracks(
            params=[(D_MEAN, eps, float(psi))], n_replicates=1,
            track_length=track_length, dt_s=DT_S, sigma_loc_um=SIGMA_LOC,
            seed=int(rng.integers(0, 1_000_000)),
        )
        dx, dy = _tracks_to_arrays(sim)
        out.append(fit_track_nested(dx[0], dy[0], DT_S, seed=i).log_bf10)
    return np.array(out)


def check_3_null_false_positive_rate(n_rep: int) -> None:
    """Truly isotropic tracks must not accumulate evidence for anisotropy."""
    print(f"\ncheck 3: null calibration, true eps=0, {n_rep} tracks per length")
    rows = []
    for L in (5, 10, 20, 50, 100):
        v = _log_bfs(0.0, L, n_rep, seed=100 + L)
        rows.append({"track_length": L, "median_log_bf10": float(np.median(v)),
                     "p90": float(np.percentile(v, 90)), "max": float(v.max()),
                     "frac_moderate_or_more": float((v > 1.1).mean())})
        print(f"   L={L:>3}: median {rows[-1]['median_log_bf10']:+.2f}  "
              f"p90 {rows[-1]['p90']:+.2f}  max {rows[-1]['max']:+.2f}  "
              f"frac>1.1 {rows[-1]['frac_moderate_or_more']:.0%}")
    pl.DataFrame(rows).write_csv(TABLE_DIR / "null_false_positive_rate.csv")


def check_4_detectability(n_rep: int) -> None:
    """How long must a track be before it can answer for itself?"""
    print(f"\ncheck 4: per-track detectability, {n_rep} tracks per cell")
    rows = []
    for L in (5, 10, 20, 50, 100, 200):
        line = f"   L={L:>3}:"
        for eps in (0.0, 0.5, 0.8):
            v = _log_bfs(eps, L, n_rep, seed=200 + L + int(100 * eps))
            rows.append({"track_length": L, "true_eps": eps,
                         "median_log_bf10": float(np.median(v)),
                         "frac_strong": float((v > 3.0).mean())})
            line += f"  eps={eps}: med {np.median(v):+7.2f} P(>3)={np.mean(v > 3):.0%}"
        print(line)
    pl.DataFrame(rows).write_csv(TABLE_DIR / "detectability_by_track_length.csv")


def check_5_eps_psi_recovery(n_rep: int) -> None:
    """Where the question is answerable, is the answer right?"""
    print(f"\ncheck 5: eps/psi recovery on long tracks, {n_rep} tracks per cell")
    rows = []
    for L in (50, 200):
        for eps_true in (0.0, 0.5, 0.8):
            rng = np.random.default_rng(300 + L)
            errs, cover, psi_err = [], [], []
            for i in range(n_rep):
                psi_true = float(rng.uniform(0, np.pi))
                sim = simulate_anisotropic_tracks(
                    params=[(D_MEAN, eps_true, psi_true)], n_replicates=1,
                    track_length=L, dt_s=DT_S, sigma_loc_um=SIGMA_LOC,
                    seed=int(rng.integers(0, 1_000_000)))
                dx, dy = _tracks_to_arrays(sim)
                f = fit_track_nested(dx[0], dy[0], DT_S, seed=i)
                errs.append(f.eps_median - eps_true)
                cover.append(f.eps_lo <= eps_true <= f.eps_hi)
                d = (f.psi_rad - psi_true) % np.pi
                psi_err.append(min(d, np.pi - d))
            rows.append({"track_length": L, "true_eps": eps_true,
                         "eps_bias": float(np.mean(errs)),
                         "hpdi90_coverage": float(np.mean(cover)),
                         "psi_abs_err_rad": float(np.median(psi_err))})
            print(f"   L={L:>3} eps={eps_true}: bias {np.mean(errs):+.3f}  "
                  f"90% HPDI coverage {np.mean(cover):.0%}  "
                  f"|psi err| {np.median(psi_err):.3f} rad")
    pl.DataFrame(rows).write_csv(TABLE_DIR / "eps_psi_recovery.csv")


def main(quick: bool) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    prior = LogEuclideanAnisotropicPrior()
    print(f"prior: tau_log_ratio={prior.tau_log_ratio}, D={D_MEAN} um^2/s, "
          f"sigma_loc={SIGMA_LOC} um, dt={DT_S} s")
    check_1_dst_equals_dense()
    check_2_isotropy_reduction()
    n_rep = 6 if quick else 20
    check_3_null_false_positive_rate(n_rep)
    check_4_detectability(n_rep)
    check_5_eps_psi_recovery(n_rep)
    print(f"\nTables written to {TABLE_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fewer replicates")
    main(**vars(ap.parse_args()))
