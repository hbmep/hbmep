import pandas as pd

url = "/home/vishu/sc_ramp.csv"
df = pd.read_csv(url)

import sys, os
# sys.path.append("/Users/suheylatozan/Desktop/hbmep/src")
df.shape
idx = (df.participant == "s109") & (df.recr_curve == "scramp-002")
temp_df = df[idx].reset_index(drop=True).copy()
import seaborn as sns

sns.scatterplot(x=temp_df["sc_current"], y=temp_df["FCR"])
df

import os

current_directory = os.getcwd()
output_path = os.path.join(current_directory, "paired_dataset.pdf")

# Plot dataset and save it as a PDF
# model.plot(df, output_path=output_path)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import jax
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist

# Disable JAX warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

import logging

import numpy as np
import jax
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist

from hbmep import functional as F, smooth_functional as SF
from hbmep.model import BaseModel
from hbmep.util import site

EPS = 1e-3


class HB(BaseModel):
    def __init__(self, *args, **kw):
        super(HB, self).__init__(*args, **kw)
        self.use_mixture = False

    def hb_rl(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        # Mask missing observations
        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        # Hyper-priors
        a_loc = pyro.sample(
            site.a.log, dist.TruncatedNormal(5., 10., low=0)
        )
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(10.))
        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(10.))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(50.))

        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(1.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        # Priors
        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate_stack(
                site.num_features, num_features, rightmost_dim=-2
            ):
                a = pyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                b = pyro.deterministic(site.b, b_scale * b_raw)

                g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                g = pyro.deterministic(site.g, g_scale * g_raw)

                h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                h = pyro.deterministic(site.h, h_scale * h_raw)

                v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                v = pyro.deterministic(site.v, v_scale * v_raw)

                c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        # Outlier probability
        if self.use_mixture:
            q = pyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

        # Observation model
        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu = SF.rectified_logistic(
                        intensity,
                        a[*features.T],
                        b[*features.T],
                        g[*features.T],
                        h[*features.T],
                        v[*features.T],
                        EPS
                    )
                    alpha, beta = self.gamma_likelihood(
                        mu, 
                        c1[*features.T],
                        c2[*features.T],
                    )
                    pyro.deterministic(site.mu, mu)

                    # Mixture distribution
                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Gamma(concentration=alpha, rate=beta),
                            dist.HalfNormal(
                                scale=(g[*features.T] + h[*features.T])
                            )
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    # Observations
                    y_ = pyro.sample(
                        site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.Gamma(concentration=alpha, rate=beta)
                        ),
                        obs=response
                    )


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def rectified_logistic_scaled(x01, a, b, g, h):
    return g + h * sigmoid(b * (x01 - a))

def rectified_logistic(x01, a, b, g, h, v):
    # v controls sharpness
    return g + max(0, (-v + (h+v)*sigmoid(b*(x-a)-np.log(h/v))))
    


def scale_minmax(x):
    xmin, xmax = float(np.min(x)), float(np.max(x))
    rng = xmax - xmin if xmax > xmin else 1.0
    return (x - xmin) / rng, xmin, rng

def fit_rectified_logistic(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 4:
        raise ValueError("Need at least 4 points")
    x01, xmin, xrng = scale_minmax(x)
    g0 = np.percentile(y, 10.0)
    u0 = np.percentile(y, 90.0)
    h0 = max(u0 - g0, 1e-6)
    a0, b0 = 0.5, 5.0
    p0 = np.array([a0, b0, g0, h0], float)
    lb = np.array([0.0, 1e-4, 0.0, 0.0])
    ub = np.array([1.0, 200.0, np.inf, np.inf])

    def residuals(theta):
        a, b, g, h = theta
        mu = rectified_logistic_scaled(x01, a, b, g, h)
        return mu - y

    res = least_squares(residuals, p0, bounds=(lb, ub),
                        loss="soft_l1", f_scale=1.0, max_nfev=20000)
    a, b, g, h = res.x
    mu = rectified_logistic_scaled(x01, a, b, g, h)
    rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
    ss_res = np.sum((y - mu)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"params": dict(a=a,b=b,g=g,h=h,xmin=xmin,xrng=xrng),
            "rmse": rmse, "r2": r2, "success": bool(res.success)}

def predict_ls_curve(x_new, fit):
    p = fit["params"]
    x01 = (np.asarray(x_new) - p["xmin"]) / (p["xrng"] if p["xrng"] != 0 else 1.0)
    return rectified_logistic_scaled(x01, p["a"], p["b"], p["g"], p["h"])



csv_path = "/home/vishu/sc_ramp.csv"
df = pd.read_csv(csv_path)
req_cols = ["sc_current", "FCR", "participant", "recr_curve"]
df = df.dropna(subset=req_cols).copy()

group_cols = ["participant", "recr_curve"]

#Run hbMEP Bayesian Model

model = HB()
model.intensity = "sc_current"
model.features = group_cols
model.response = ["FCR"]
model._model = model.hb_rl
model.use_mixture = True
model.mcmc_params = dict(num_chains=2, num_warmup=500, num_samples=250)

# Encode and run
df_enc, enc = model.load(df)
mcmc, posterior = model.run(df=df_enc)
pred_df = model.make_prediction_dataset(df=df_enc, num_points=200)
predictive = model.predict(df=pred_df, posterior=posterior, num_samples=500, return_sites=[site.mu])
pred_df["mu_post_mean"] = np.asarray(predictive[site.mu]).mean(axis=0)
