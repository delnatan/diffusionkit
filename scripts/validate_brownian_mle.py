"""Short-track study of the Brownian MLE and its calibrated non-Brownian score.

Independent position-space simulators (fBm by Cholesky, exact OU steps,
blur by sub-frame averaging); no production likelihood code is used to
generate data. Questions:

1. Null: under Brownian motion + the stated localization errors, is
   z_nonbrownian ~ N(0,1) for every length and D, and uncorrelated with D_hat?
2. Power: how often does it flag fBm, confinement and drift, compared with
   the MSD alpha fit thresholded at the same false-positive rate?
3. Robustness: how far does z shift if localization SDs or exposure are wrong?
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
from scipy import stats

from diffusionkit import Acquisition
from diffusionkit.classic import MLEOptions, analyze_tracks

DT = .033


def localization_sd(n):
    return .025*np.column_stack((np.linspace(.4, 1.6, n), np.linspace(1.5, .5, n)))


def fbm(n, K, alpha, count, rng):
    # Cov(X(t), X(s)) = K(t^alpha + s^alpha - |t-s|^alpha) per axis.
    times = np.arange(1, n)*DT
    cov = K*(times[:, None]**alpha + times[None, :]**alpha - np.abs(times[:, None]-times[None, :])**alpha)
    out = np.zeros((count, n, 2))
    if K == 0:
        return out
    out[:, 1:] = np.einsum("ij,bjd->bid", np.linalg.cholesky(cov), rng.normal(size=(count, n-1, 2)))
    return out


def blurred_brownian(n, D, count, rng, exposure, substeps=60):
    # Positions are means of a fine Brownian path over [t_i - exposure, t_i].
    fine_dt = DT/substeps
    steps = rng.normal(size=(count, n*substeps, 2))*np.sqrt(2*D*fine_dt)
    path = np.cumsum(steps, axis=1)
    window = int(round(exposure/fine_dt))
    ends = (np.arange(n)+1)*substeps
    return np.stack([path[:, e-window:e].mean(axis=1) for e in ends], axis=1)


def ou(n, D, kappa, count, rng):
    decay = np.exp(-kappa*DT)
    step_sd = np.sqrt(D/kappa*(1-decay**2))
    x = np.zeros((count, n, 2))
    x[:, 0] = rng.normal(size=(count, 2))*np.sqrt(D/kappa)
    for i in range(1, n):
        x[:, i] = decay*x[:, i-1] + step_sd*rng.normal(size=(count, 2))
    return x


def drift(n, D, speed, count, rng):
    angle = rng.uniform(0, 2*np.pi, count)
    direction = np.column_stack((np.cos(angle), np.sin(angle)))
    return fbm(n, D, 1., count, rng) + speed*DT*np.arange(n)[None, :, None]*direction[:, None, :]


def table(true_positions, rng, sd_scale=1.):
    count, n, _ = true_positions.shape
    sd = localization_sd(n)
    observed = true_positions + rng.normal(size=true_positions.shape)*sd
    return pl.DataFrame({
        "track_id": np.repeat(np.arange(count), n), "frame": np.tile(np.arange(n), count),
        "x_um": observed[:, :, 0].ravel(), "y_um": observed[:, :, 1].ravel(),
        "sigma_x_um": np.tile(sd[:, 0]*sd_scale, count), "sigma_y_um": np.tile(sd[:, 1]*sd_scale, count),
    })


def analyze(true_positions, rng, n_boot, *, exposure=0., sd_scale=1.):
    fits = analyze_tracks(table(true_positions, rng, sd_scale), Acquisition(DT, exposure),
                          mle_options=MLEOptions(n_boot=n_boot, seed=int(rng.integers(2**31)))).fits
    mle = fits.filter(pl.col("model") == "brownian_mle").sort("track_id")
    alpha = fits.filter(pl.col("model") == "power_law").sort("track_id")["alpha"]
    msd_D = fits.filter(pl.col("model") == "brownian").sort("track_id")["D_um2_s"]
    return mle, alpha, msd_D


def null_summary(mle, D_true, msd_D):
    statuses = mle["status"].value_counts().sort("status")
    D_hat = mle["D_um2_s"].to_numpy()
    z = mle["z_nonbrownian"].drop_nulls().to_numpy()
    resolved = mle.filter(pl.col("z_nonbrownian").is_not_null())
    out = {
        "status_counts": dict(zip(statuses["status"].to_list(), statuses["count"].to_list())),
        "fraction_D_hat_zero": float(np.mean(D_hat == 0)),
        "fraction_p_motion_below_0.05": float(np.mean(mle["p_motion"].to_numpy() < .05)),
        "n_z": int(z.size),
    }
    if D_true > 0:
        out["mle_D_relative_bias"] = float(np.mean(D_hat)/D_true - 1)
        out["mle_D_relative_rmse"] = float(np.sqrt(np.mean((D_hat-D_true)**2))/D_true)
        m = msd_D.drop_nulls().to_numpy()
        if m.size:
            out["msd3_D_relative_rmse"] = float(np.sqrt(np.mean((m-D_true)**2))/D_true)
    if z.size >= 20:
        out.update({
            "z_mean": float(z.mean()), "z_sd": float(z.std(ddof=1)),
            "fraction_abs_z_above_1.96": float(np.mean(np.abs(z) > 1.96)),
            "ks_p_vs_standard_normal": float(stats.kstest(z, "norm").pvalue),
            "spearman_z_vs_log_D_hat": float(stats.spearmanr(z, np.log(resolved["D_um2_s"].to_numpy()))[0]),
            "z_asymptotic_sd": float(resolved["z_nonbrownian_asymptotic"].to_numpy().std(ddof=1)),
        })
    return out


def power_summary(mle, alpha, null_alpha):
    z = mle["z_nonbrownian"].to_numpy()
    p = mle["p_nonbrownian"].to_numpy()
    a = alpha.to_numpy()
    lo, hi = np.nanquantile(null_alpha, [.025, .975])
    has_z = np.isfinite(z)
    return {
        "fraction_z_defined": float(has_z.mean()),
        "z_mean": float(np.nanmean(z)) if has_z.any() else None,
        "power_score_p_below_0.05": float(np.mean(np.nan_to_num(p, nan=1.) < .05)),
        "power_msd_alpha_outside_null_95": float(np.mean((a < lo) | (a > hi))),
        "alpha_1step_median": float(np.nanmedian(mle["alpha_1step"].to_numpy())) if has_z.any() else None,
        "msd_alpha_median": float(np.nanmedian(a)) if np.isfinite(a).any() else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"audit/brownian_mle_validation.json")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    lengths, D_values = (5, 8, 10, 15, 20), (0., .01, .05, .2)
    result = {"settings": {"dt_s": DT, "count_per_cell": args.count, "n_boot": args.n_boot,
                           "seed": args.seed, "localization_sd_um": "0.025 x linear ramps 0.4-1.6 (x), 1.5-0.5 (y)"},
              "null": [], "power": [], "robustness": []}
    null_alpha = {}
    for n in lengths:
        for D in D_values:
            mle, alpha, msd_D = analyze(fbm(n, D, 1., args.count, rng), rng, args.n_boot)
            if D == .05:
                null_alpha[n] = alpha.to_numpy()
            result["null"].append({"n_frames": n, "D_um2_s": D, **null_summary(mle, D, msd_D)})
            print("null", n, D, result["null"][-1].get("z_sd"), flush=True)
        mle, _, _ = analyze(blurred_brownian(n, .05, args.count, rng, DT), rng, args.n_boot, exposure=DT)
        result["null"].append({"n_frames": n, "D_um2_s": .05, "exposure_s": DT, **null_summary(mle, .05, pl.Series([]))})
    for n in (5, 10, 20):
        cases = [(f"fbm_alpha_{a}", fbm(n, .05, a, args.count, rng)) for a in (.5, .75, 1.25, 1.5)]
        cases += [("ou_kappa_6_per_s", ou(n, .05, 6., args.count, rng)),
                  ("ou_kappa_30_per_s", ou(n, .05, 30., args.count, rng)),
                  ("drift_1_um_per_s", drift(n, .05, 1., args.count, rng))]
        for name, positions in cases:
            mle, alpha, _ = analyze(positions, rng, args.n_boot)
            result["power"].append({"n_frames": n, "case": name, "K_or_D_um2_s": .05,
                                    **power_summary(mle, alpha, null_alpha[n])})
            print("power", n, name, result["power"][-1]["power_score_p_below_0.05"], flush=True)
        for name, positions, kwargs in (
                ("sd_reported_0.8x", fbm(n, .05, 1., args.count, rng), {"sd_scale": .8}),
                ("sd_reported_1.2x", fbm(n, .05, 1., args.count, rng), {"sd_scale": 1.2}),
                ("exposure_dt_ignored", blurred_brownian(n, .05, args.count, rng, DT), {})):
            mle, _, msd_D = analyze(positions, rng, args.n_boot, **kwargs)
            result["robustness"].append({"n_frames": n, "case": name, **null_summary(mle, .05, msd_D)})
    result["runtime_s"] = time.perf_counter() - started
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output} in {result['runtime_s']:.0f} s")


if __name__ == "__main__":
    main()
