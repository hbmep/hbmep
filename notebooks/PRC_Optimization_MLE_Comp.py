import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from scipy.special import gammaln
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# Disable warnings for cleaner output
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist

from hbmep import functional as F, smooth_functional as SF
from hbmep.model import BaseModel
from hbmep.util import site, make_pdf

EPS = 1e-3
n_inits = 1


# HBMEP Bayesian Model Definition


class HB(BaseModel):
    def __init__(self, *args, **kw):
        super(HB, self).__init__(*args, **kw)
        self.use_mixture = False

    def hb_rl(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1
        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        a_loc = pyro.sample(site.a.log, dist.TruncatedNormal(5., 10., low=0))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(10.))
        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(10.))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(50.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(1.))
        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate_stack(site.num_features, num_features, rightmost_dim=-2):
                a = pyro.sample(site.a, dist.TruncatedNormal(a_loc, a_scale, low=0))

                b = pyro.deterministic(site.b, b_scale * pyro.sample(site.b.raw, dist.HalfNormal(1)))
                g = pyro.deterministic(site.g, g_scale * pyro.sample(site.g.raw, dist.HalfNormal(1)))
                h = pyro.deterministic(site.h, h_scale * pyro.sample(site.h.raw, dist.HalfNormal(1)))
                v = pyro.deterministic(site.v, v_scale * pyro.sample(site.v.raw, dist.HalfNormal(1)))
                c1 = pyro.deterministic(site.c1, c1_scale * pyro.sample(site.c1.raw, dist.HalfNormal(1)))
                c2 = pyro.deterministic(site.c2, c2_scale * pyro.sample(site.c2.raw, dist.HalfNormal(1)))

        if self.use_mixture:
            q = pyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu = SF.rectified_logistic(
                        intensity, a[*features.T], b[*features.T],
                        g[*features.T], h[*features.T], v[*features.T], EPS
                    )
                    # Gamma likelihood from HBMEP model
                    alpha, beta = self.gamma_likelihood(mu, c1[*features.T], c2[*features.T])
                    pyro.deterministic(site.mu, mu)
                    pyro.sample(
                        site.obs,
                        dist.Gamma(concentration=alpha, rate=beta),
                        obs=response
                    )

# Maximum Likelihood helpers and models

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def rectified_logistic(x01, a, b, g, h, v):
    return g + np.maximum(0, (-v + (h + v) * sigmoid(b * (x01 - a) - np.log(h / v))))

def scale_minmax(x):
    xmin, xmax = float(np.min(x)), float(np.max(x))
    rng = xmax - xmin if xmax > xmin else 1.0
    return (x - xmin) / rng, xmin, rng

# ------------------------- Gamma model 1: constant k -------------------------

def gamma_negloglik(theta, x01, y):

    a, b, g, h, v, log_k = theta

    # Predicted mean curve
    mu = rectified_logistic(x01, a, b, g, h, v)

    # Clip to avoid log(0) and division by zero
    mu = np.clip(mu, 1e-6, None)
    y_clip = np.clip(y, 1e-6, None)

    # Shape parameter, enforce positivity via exp
    k = np.exp(log_k)

    # Scale so that mean = mu
    theta_scale = mu / k  # mean = k * theta_scale = mu

    # Shape scale gamma:
    # log p(y) = (k-1) * log y - y/theta - k * log theta - log Gamma(k)
    log_pdf = (k - 1.0) * np.log(y_clip) - (y_clip / theta_scale) - k * np.log(theta_scale) - gammaln(k)

    # Negative log likelihood
    nll = -np.sum(log_pdf)

    if not np.isfinite(nll):
        return 1e12
    return nll

# -------------------- Gamma model 2: alpha = mu * beta ----------------------

def gamma_negloglik_c1c2(theta, x01, y):
    a, b, g, h, v, log_c1, log_c2 = theta

    # Predicted mean curve
    mu = rectified_logistic(x01, a, b, g, h, v)

    mu = np.clip(mu, 1e-6, None)
    y_clip = np.clip(y, 1e-6, None)

    c1 = np.exp(log_c1)
    c2 = np.exp(log_c2)

    # beta = 1/c1 + (1/c2) * mu
    beta = 1.0 / c1 + (1.0 / c2) * mu
    beta = np.clip(beta, 1e-6, None)

    # alpha = mu * beta
    alpha = mu * beta
    alpha = np.clip(alpha, 1e-6, None)

    # Gamma with shape alpha, rate beta:
    # log p(y) = (alpha - 1) * log y - beta * y + alpha * log beta - log Gamma(alpha)
    log_pdf = (alpha - 1.0) * np.log(y_clip) - beta * y_clip + alpha * np.log(beta) - gammaln(alpha)

    nll = -np.sum(log_pdf)

    if not np.isfinite(nll):
        return 1e12
    return nll

# ------------------ Single init runner for constant k model ------------------

def run_single_init(args):
    p0, idx, x01, y, lb, ub = args
    try:
        print(f"[Init {idx+1}] Starting MLE fit (k model)...")

        log_k0 = np.log(5.0)  # initial shape
        theta0 = np.concatenate([p0, [log_k0]])

        # Bounds for a,b,g,h,v plus log_k
        bounds = [(lb[i], ub[i]) for i in range(len(lb))]
        bounds.append((-5.0, 5.0))  # log_k

        res = minimize(
            gamma_negloglik,
            theta0,
            args=(x01, y),
            method="L-BFGS-B",
            bounds=bounds
        )

        if (not res.success) or (not np.isfinite(res.x).all()):
            raise RuntimeError("MLE fit failed")

        a, b, g, h, v, log_k = res.x
        k = np.exp(log_k)

        mu = rectified_logistic(x01, a, b, g, h, v)
        rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
        ss_res = np.sum((y - mu) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        print(f"[Init {idx+1}] Done (k model) RMSE={rmse:.4f}, R²={r2:.4f}")
        print(f"→ Parameters: a={a:.3f}, b={b:.3f}, g={g:.3f}, h={h:.3f}, v={v:.3f}, k={k:.3f}\n")

        return dict(
            success=True,
            rmse=rmse,
            r2=r2,
            params=dict(a=a, b=b, g=g, h=h, v=v, log_k=log_k)
        )
    except Exception as e:
        print(f"[Init {idx+1}] Failed (k model): {e}")
        return dict(success=False, rmse=np.inf, r2=np.nan, params=None)

# ---------------- Single init runner for alpha = mu * beta model -------------

def run_single_init_c1c2(args):
    p0, idx, x01, y, lb, ub = args
    try:
        print(f"[Init {idx+1}] Starting MLE c1c2 fit...")

        # Initial guess for log c1, log c2
        log_c1_0 = np.log(1.0)
        log_c2_0 = np.log(1.0)
        theta0 = np.concatenate([p0, [log_c1_0, log_c2_0]])

        # Bounds for a, b, g, h, v, plus log_c1 and log_c2
        bounds = [(lb[i], ub[i]) for i in range(len(lb))]
        bounds.append((-5.0, 5.0))  # log_c1
        bounds.append((-5.0, 5.0))  # log_c2

        res = minimize(
            gamma_negloglik_c1c2,
            theta0,
            args=(x01, y),
            method="L-BFGS-B",
            bounds=bounds
        )

        if (not res.success) or (not np.isfinite(res.x).all()):
            raise RuntimeError("MLE c1c2 fit failed")

        a, b, g, h, v, log_c1, log_c2 = res.x
        c1 = np.exp(log_c1)
        c2 = np.exp(log_c2)

        mu = rectified_logistic(x01, a, b, g, h, v)
        rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
        ss_res = np.sum((y - mu) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        print(f"[Init {idx+1}] Done (c1c2) RMSE={rmse:.4f}, R²={r2:.4f}")
        print(
            f"→ c1c2 Params: a={a:.3f}, b={b:.3f}, g={g:.3f}, h={h:.3f}, "
            f"v={v:.3f}, c1={c1:.3f}, c2={c2:.3f}\n"
        )

        return dict(
            success=True,
            rmse=rmse,
            r2=r2,
            params=dict(a=a, b=b, g=g, h=h, v=v, log_c1=log_c1, log_c2=log_c2)
        )
    except Exception as e:
        print(f"[Init {idx+1}] c1c2 fit failed: {e}")
        return dict(success=False, rmse=np.inf, r2=np.nan, params=None)

# --------------------- Fit rectified logistic, constant k -------------------

def fit_rectified_logistic(x, y, n_inits=n_inits):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 5:
        raise ValueError("Need at least 5 points")

    # Normalize x
    x01, xmin, xrng = scale_minmax(x)

    # Parameter bounds for a,b,g,h,v
    lb = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    ub = np.array([20.0, 500.0, 10.0, 2000.0, 50.0])

    # Create starting vectors
    offsets = np.linspace(0, 0.1, n_inits)
    init_list = [lb + (ub - lb) * off for off in offsets]

    args_list = [(p0, i, x01, y, lb, ub) for i, p0 in enumerate(init_list)]

    results = []
    with ProcessPoolExecutor(max_workers=min(8, n_inits)) as executor:
        futures = [executor.submit(run_single_init, args) for args in args_list]
        for f in as_completed(futures):
            results.append(f.result())

    successful = [r for r in results if r["success"]]
    if not successful:
        return dict(params=None, rmse=np.nan, r2=np.nan, success=False)

    best = min(successful, key=lambda r: (r["rmse"], -np.nan_to_num(r["r2"])))
    best["params"].update(dict(xmin=xmin, xrng=xrng))
    best["success"] = True

    print("\n=== Parallel fitting summary (MLE k model) ===")
    for i, r in enumerate(results):
        if r["success"]:
            print(f"Init {i+1}: RMSE={r['rmse']:.4f}, R²={r['r2']:.4f}")
        else:
            print(f"Init {i+1}: FAILED")
    print(f"→ Best initialization RMSE={best['rmse']:.4f}, R²={best['r2']:.4f}\n")

    return best

# ---------------- Fit rectified logistic, alpha = mu * beta model -----------

def fit_rectified_logistic_c1c2(x, y, n_inits=n_inits):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 5:
        raise ValueError("Need at least 5 points")

    x01, xmin, xrng = scale_minmax(x)

    lb = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    ub = np.array([20.0, 500.0, 10.0, 2000.0, 50.0])

    offsets = np.linspace(0, 0.1, n_inits)
    init_list = [lb + (ub - lb) * off for off in offsets]

    args_list = [(p0, i, x01, y, lb, ub) for i, p0 in enumerate(init_list)]

    results = []
    with ProcessPoolExecutor(max_workers=min(8, n_inits)) as executor:
        futures = [executor.submit(run_single_init_c1c2, args) for args in args_list]
        for f in as_completed(futures):
            results.append(f.result())

    successful = [r for r in results if r["success"]]
    if not successful:
        return dict(params=None, rmse=np.nan, r2=np.nan, success=False)

    best = min(successful, key=lambda r: (r["rmse"], -np.nan_to_num(r["r2"])))
    best["params"].update(dict(xmin=xmin, xrng=xrng))
    best["success"] = True

    print("\n=== Parallel fitting summary (MLE c1c2 model) ===")
    for i, r in enumerate(results):
        if r["success"]:
            print(f"Init {i+1}: RMSE={r['rmse']:.4f}, R²={r['r2']:.4f}")
        else:
            print(f"Init {i+1}: FAILED")
    print(f"→ Best c1c2 initialization RMSE={best['rmse']:.4f}, R²={best['r2']:.4f}\n")

    return best

# --------------------------- Prediction helper -------------------------------

def predict_ls_curve(x_new, fit):
    p = fit["params"]
    x01 = (np.asarray(x_new) - p["xmin"]) / (p["xrng"] if p["xrng"] != 0 else 1.0)
    return rectified_logistic(x01, p["a"], p["b"], p["g"], p["h"], p["v"])

# Grouped fitting and plotting

def fit_by_group_ls(df, intensity_col, response_col, group_cols):
    """
    Fits the constant k Gamma model per group.
    """
    rows = []
    for combo, gdf in df.groupby(group_cols):
        combo = combo if isinstance(combo, tuple) else (combo,)
        print(f"Starting group {combo} (k model)...")
        try:
            fit = fit_rectified_logistic(gdf[intensity_col].values, gdf[response_col].values)
            row = {c: v for c, v in zip(group_cols, combo)}
            row.update(fit["params"])
            row.update(dict(success=fit["success"], rmse=fit["rmse"], r2=fit["r2"]))
            rows.append(row)
        except Exception as e:
            row = {c: v for c, v in zip(group_cols, combo)}
            row.update(
                dict(
                    a=np.nan, b=np.nan, g=np.nan, h=np.nan,
                    xmin=np.nan, xrng=np.nan,
                    success=False, rmse=np.nan, r2=np.nan
                )
            )
            rows.append(row)
            print(f"Group {combo} failed (k model): {e}")
    return pd.DataFrame(rows)

def fit_by_group_ls_c1c2(df, intensity_col, response_col, group_cols):
    """
    Fits the alpha = mu * beta Gamma model per group.
    """
    rows = []
    for combo, gdf in df.groupby(group_cols):
        combo = combo if isinstance(combo, tuple) else (combo,)
        print(f"Starting group {combo} (c1c2 model)...")
        try:
            fit = fit_rectified_logistic_c1c2(gdf[intensity_col].values, gdf[response_col].values)
            row = {c: v for c, v in zip(group_cols, combo)}
            row.update(fit["params"])
            row.update(dict(success=fit["success"], rmse=fit["rmse"], r2=fit["r2"]))
            rows.append(row)
        except Exception as e:
            row = {c: v for c, v in zip(group_cols, combo)}
            row.update(
                dict(
                    a=np.nan, b=np.nan, g=np.nan, h=np.nan, v=np.nan,
                    log_c1=np.nan, log_c2=np.nan,
                    xmin=np.nan, xrng=np.nan,
                    success=False, rmse=np.nan, r2=np.nan
                )
            )
            rows.append(row)
            print(f"Group {combo} failed (c1c2 model): {e}")
    return pd.DataFrame(rows)

def compare_and_plot(
    df,
    pred_df,
    ls_table_k,
    ls_table_c1c2,
    intensity_col="sc_current",
    response_col="FCR",
    group_cols=("participant", "recr_curve")
):
    """
    Plot data, HBMEP posterior mean, MLE k-model, and MLE c1c2-model
    """

    unique_groups = list(
        ls_table_k[list(group_cols)].drop_duplicates().itertuples(index=False, name=None)
    )
    n = len(unique_groups)
    ncols, nrows = 2, int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows), squeeze=False)

    for idx, grp in enumerate(unique_groups):
        ax = axes[idx // ncols][idx % ncols]

        # Mask for this group
        mask_data = np.ones(len(df), dtype=bool)
        mask_pred = np.ones(len(pred_df), dtype=bool)
        for c, v in zip(group_cols, grp):
            mask_data &= (df[c] == v)
            mask_pred &= (pred_df[c] == v)

        # Extract data
        gdf = df.loc[mask_data, [intensity_col, response_col]].sort_values(intensity_col)

        # Extract posterior mean
        gpred = pred_df.loc[
            mask_pred, [intensity_col, f"mu_post_mean_{response_col}"]
        ].sort_values(intensity_col)

        ax.scatter(
            gdf[intensity_col],
            gdf[response_col],
            s=14,
            alpha=0.7,
            color="k",
            label="Data"
        )

        if gpred.empty:
            ax.set_title(f"{grp} — No predictions")
            continue

        # Common prediction grid
        xgrid = np.linspace(gdf[intensity_col].min(), gdf[intensity_col].max(), 200)

        # HBMEP curve
        xpred = gpred[intensity_col].values
        ypred = gpred[f"mu_post_mean_{response_col}"].values
        yhat_post = np.interp(xgrid, xpred, ypred)
        ax.plot(xgrid, yhat_post, "b--", lw=2, label="HBMEP mean")

        # MLE k-model curve
        row_k = ls_table_k
        for c, v in zip(group_cols, grp):
            row_k = row_k[row_k[c] == v]

        if len(row_k) == 1 and bool(row_k.iloc[0]["success"]):
            params_k = {
                "params": {k: row_k.iloc[0][k] for k in ["a", "b", "g", "h", "v", "xmin", "xrng"]}
            }
            yhat_k = predict_ls_curve(xgrid, params_k)
            ax.plot(xgrid, yhat_k, "r", lw=2, label="MLE k")

        # MLE c1c2-model curve
        row_c = ls_table_c1c2
        for c, v in zip(group_cols, grp):
            row_c = row_c[row_c[c] == v]

        if len(row_c) == 1 and bool(row_c.iloc[0]["success"]):
            params_c = {
                "params": {k: row_c.iloc[0][k] for k in ["a", "b", "g", "h", "v", "xmin", "xrng"]}
            }
            yhat_c = predict_ls_curve(xgrid, params_c)
            ax.plot(xgrid, yhat_c, "g", lw=2, label="MLE c1c2")

        ax.set_title(f"{grp}")
        ax.legend()

    plt.tight_layout()
    return fig, None



#muscles = ["ADM", "APB", "ECR", "FCR", "Triceps"]
muscles = ["FCR"]

def main():
    csv_path = "/Users/suheylatozan/Desktop/Movement Recovery Lab/sc_ramp.csv"
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["sc_current", "participant", "recr_curve"]).copy()
    group_cols = ["participant", "recr_curve"]

    model = HB()
    model.intensity = "sc_current"
    model.features = group_cols
    model.response = muscles
    model._model = model.hb_rl
    model.use_mixture = True
    model.mcmc_params = dict(num_chains=2, num_warmup=500, num_samples=250)

    # Run HBMEP model
    df_enc, enc = model.load(df)
    mcmc, posterior = model.run(df=df_enc)
    pred_df = model.make_prediction_dataset(df=df_enc, num_points=200)
    predictive = model.predict(
        df=pred_df,
        posterior=posterior,
        num_samples=500,
        return_sites=[site.mu]
    )

    # Posterior mean for each muscle
    mu_post = np.asarray(predictive[site.mu])   # (samples, points, muscles)
    mu_mean = mu_post.mean(axis=0)             # (points, muscles)

    for i, m in enumerate(muscles):
        pred_df[f"mu_post_mean_{m}"] = mu_mean[:, i]

    # Decode participants and recruitment curves
    for col in model.features:
        if col in pred_df and col in enc:
            le = enc[col]
            if hasattr(le, "classes_"):
                pred_df[col] = le.inverse_transform(pred_df[col].astype(int))

    print("Decoded participants:", pred_df["participant"].unique())
    print("Decoded recr_curve:", pred_df["recr_curve"].unique())

    output_dir = "/Users/suheylatozan/Desktop/Movement Recovery Lab/hbmep/notebooks/hbmep_outputs"
    os.makedirs(output_dir, exist_ok=True)
    all_figures = []

    for muscle in muscles:
        print("\n\n==============================")
        print(f"Processing muscle: {muscle}")
        print("==============================\n")

        df_m = df.dropna(subset=[muscle]).copy()

        # Compute MLE fits for both models
        ls_table_k = fit_by_group_ls(
            df_m,
            intensity_col="sc_current",
            response_col=muscle,
            group_cols=group_cols
        )

        ls_table_c1c2 = fit_by_group_ls_c1c2(
            df_m,
            intensity_col="sc_current",
            response_col=muscle,
            group_cols=group_cols
        )

        # Compare HBMEP vs MLE k vs MLE c1c2
        fig, report = compare_and_plot(
            df_m,
            pred_df,
            ls_table_k,
            ls_table_c1c2,
            intensity_col="sc_current",
            response_col=muscle,
            group_cols=group_cols
        )

        fig.suptitle(f"Recruitment Curves — {muscle}", fontsize=18)
        all_figures.append(fig)

    output_path = os.path.join(
        output_dir,
        f"paired_curves_all_muscles_n={n_inits}_MLE_k_vs_c1c2.pdf"
    )
    make_pdf(figures=all_figures, output_path=output_path)
    print(f"\nSaved combined PDF for all muscles (MLE k vs c1c2) to:\n{output_path}\n")

if __name__ == "__main__":
    main()
