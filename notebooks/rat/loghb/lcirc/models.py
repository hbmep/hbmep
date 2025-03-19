import logging

import numpy as np
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import (
    autoguide,
	Predictive,
	SVI,
	Trace_ELBO,
	TraceEnum_ELBO,
	TraceMeanField_ELBO
)

from hbmep import functional as F, smooth_functional as SF
from hbmep.model import BaseModel
from hbmep.util import site, timing

from hbmep.notebooks.rat.util import get_subname

logger = logging.getLogger(__name__)
EPS = 1e-3


class HB(BaseModel):
    def __init__(self, *args, **kw):
        super(HB, self).__init__(*args, **kw)

        # self._model = self.hb_mvn_rl_nov
        self._model = self.hb_mvn_rl_nov_masked
        self.use_mixture = False
        self.run_id = "diam"
        # self.run_id = "vertices"
        # self.run_id = "radii"

        # # self._model = self.hb_mvn_l4_svi
        # self._model = self.hb_mvn_l4_masked
        # self.use_mixture = False
        # self.run_id = "all"

        self.mcmc_params = {
            "num_chains": 4,
            # "thinning": 4,
            # "num_warmup": 4000,
            # "num_samples": 4000,
            "thinning": 1,
            "num_warmup": 1000,
            "num_samples": 1000,
            # "thinning": 1,
            # "num_warmup": 400,
            # "num_samples": 400,
        }
        self.nuts_params = {
            "max_tree_depth": (15, 15),
            "target_accept_prob": .95,
        }

    @property
    def name(self): return get_subname(self)

    @name.setter
    def name(self, value): return value

    @timing
    def run_svi(self, df, lr=1e-2, steps=2000, PROGRESS_BAR=True):
        optimizer = optim.ClippedAdam(step_size=lr)
        _guide = autoguide.AutoLowRankMultivariateNormal(self._model)
        svi = SVI(
            self._model,
            _guide,
            optimizer,
            loss=Trace_ELBO(num_particles=20)
        )
        self.sample_sites
        self._update_sites(df)
        svi_result = svi.run(
            self.key,
            steps,
            *self.get_regressors(df),
            *self.get_response(df),
            progress_bar=PROGRESS_BAR
        )
        predictive = Predictive(
            _guide,
            params=svi_result.params,
            return_sites=self.sample_sites + self.reparam_sites,
            num_samples=4000
        )
        posterior = predictive(self.key, *self.get_regressors(df=df))
        posterior = {u: np.array(v) for u, v in posterior.items()}
        return svi_result, posterior

    def hb_mvn_rl_nov(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        # v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        a_loc = pyro.sample(site.a.loc, dist.Normal(7, 2.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(2.))
        Rho = pyro.sample("Rho" ,dist.LKJ(self.num_response, 1.))

        with pyro.plate(site.num_features[0], num_features[0]):
            a_raw = pyro.sample(
                site.a.raw, dist.MultivariateNormal(
                    0, (
                        jnp.array([a_scale])[:, None] @ jnp.array([a_scale]) 
                    ) * Rho
                )
            )
            a = pyro.deterministic(site.a, a_loc + a_raw)

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[0], num_features[0]):
                b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                b = pyro.deterministic(site.b, b_scale * b_raw)

                g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                g = pyro.deterministic(site.g, g_scale * g_raw)

                h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                h = pyro.deterministic(site.h, h_scale * h_raw)

                # v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                # v = pyro.deterministic(site.v, v_scale * v_raw)

                c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_data, num_data):
                mu, alpha, beta = self.gamma_likelihood(
                    SF.rectified_logistic,
                    intensity,
                    (
                        a[feature0],
                        b[feature0],
                        g[feature0],
                        h[feature0],
                        # v[feature0],
                        h[feature0],
                        EPS,
                    ),
                    c1[feature0],
                    c2[feature0],
                )
                pyro.deterministic(site.mu, mu)

                if self.use_mixture:
                    mixing_distribution = dist.Categorical(
                        probs=jnp.stack([1 - q, q], axis=-1)
                    )
                    component_distributions=[
                        dist.Gamma(concentration=alpha, rate=beta),
                        dist.HalfNormal(scale=(g[feature0] + h[feature0]))
                    ]
                    Mixture = dist.MixtureGeneral(
                        mixing_distribution=mixing_distribution,
                        component_distributions=component_distributions
                    )

                pyro.sample(
                    site.obs,
                    (
                        Mixture if self.use_mixture
                        else dist.Gamma(concentration=alpha, rate=beta)
                    ),
                    obs=response
                )

    def hb_mvn_l4(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        a_loc = pyro.sample(site.a.loc, dist.Normal(7., 2.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(2.))
        Rho = pyro.sample("Rho" ,dist.LKJ(self.num_response, 1.))

        with pyro.plate(site.num_features[0], num_features[0]):
            a_raw = pyro.sample(
                site.a.raw, dist.MultivariateNormal(
                    0, (
                        jnp.array([a_scale])[:, None] @ jnp.array([a_scale]) 
                    ) * Rho
                )
            )
            a = pyro.deterministic(site.a, a_loc + a_raw)

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[0], num_features[0]):
                b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                b = pyro.deterministic(site.b, b_scale * b_raw)

                g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                g = pyro.deterministic(site.g, g_scale * g_raw)

                h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                h = pyro.deterministic(site.h, h_scale * h_raw)

                c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_data, num_data):
                mu, alpha, beta = self.gamma_likelihood(
                    F.logistic4,
                    intensity,
                    (
                        a[feature0],
                        b[feature0],
                        g[feature0],
                        h[feature0],
                    ),
                    c1[feature0],
                    c2[feature0],
                )
                pyro.deterministic(site.mu, mu)

                if self.use_mixture:
                    mixing_distribution = dist.Categorical(
                        probs=jnp.stack([1 - q, q], axis=-1)
                    )
                    component_distributions=[
                        dist.Gamma(concentration=alpha, rate=beta),
                        dist.HalfNormal(scale=(g[feature0] + h[feature0]))
                    ]
                    Mixture = dist.MixtureGeneral(
                        mixing_distribution=mixing_distribution,
                        component_distributions=component_distributions
                    )

                pyro.sample(
                    site.obs,
                    (
                        Mixture if self.use_mixture
                        else dist.Gamma(concentration=alpha, rate=beta)
                    ),
                    obs=response
                )

    def hb_mvn_rl_nov_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        mask_features = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))
            mask_features = np.full((*num_features, self.num_response), False)
            mask_features[*features.T] = True

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        # v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        a_loc = pyro.sample(site.a.loc, dist.Normal(7., 2.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(2.))
        Rho = pyro.sample("Rho" ,dist.LKJ(self.num_response, 1.))

        with pyro.plate_stack(site.num_features, num_features, rightmost_dim=-1):
            a_raw = pyro.sample(
                site.a.raw, dist.MultivariateNormal(
                    0, (a_scale ** 2) * Rho
                    
                )
            )
            a = pyro.deterministic(site.a, a_loc + a_raw)

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate_stack(site.num_features, num_features, rightmost_dim=-2):
                b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                b = pyro.deterministic(site.b, b_scale * b_raw)

                g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                g = pyro.deterministic(site.g, g_scale * g_raw)

                h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                h = pyro.deterministic(site.h, h_scale * h_raw)

                # v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                # v = pyro.deterministic(site.v, v_scale * v_raw)

                c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_data, num_data):
                mu, alpha, beta = self.gamma_likelihood(
                    SF.rectified_logistic,
                    intensity,
                    (
                        a[*features.T],
                        b[*features.T],
                        g[*features.T],
                        h[*features.T],
                        # v[*features.T],
                        h[*features.T],
                        EPS,
                    ),
                    c1[*features.T],
                    c2[*features.T],
                )
                pyro.deterministic(site.mu, mu)

                if self.use_mixture:
                    mixing_distribution = dist.Categorical(
                        probs=jnp.stack([1 - q, q], axis=-1)
                    )
                    component_distributions=[
                        dist.Gamma(concentration=alpha, rate=beta),
                        dist.HalfNormal(scale=(g[*features.T] + h[*features.T]))
                    ]
                    Mixture = dist.MixtureGeneral(
                        mixing_distribution=mixing_distribution,
                        component_distributions=component_distributions
                    )

                pyro.sample(
                    site.obs,
                    (
                        Mixture if self.use_mixture
                        else dist.Gamma(concentration=alpha, rate=beta)
                    ),
                    obs=response
                )

    def hb_mvn_l4_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        mask_features = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))
            mask_features = np.full((*num_features, self.num_response), False)
            mask_features[*features.T] = True

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        a_loc = pyro.sample(site.a.loc, dist.Normal(7., 2.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(2.))
        Rho = pyro.sample("Rho" ,dist.LKJ(self.num_response, 1.))

        with pyro.plate_stack(site.num_features, num_features, rightmost_dim=-1):
            a_raw = pyro.sample(
                site.a.raw, dist.MultivariateNormal(
                    0, (a_scale ** 2) * Rho
                    
                )
            )
            a = pyro.deterministic(site.a, a_loc + a_raw)

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate_stack(site.num_features, num_features, rightmost_dim=-2):
                b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                b = pyro.deterministic(site.b, b_scale * b_raw)

                g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                g = pyro.deterministic(site.g, g_scale * g_raw)

                h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                h = pyro.deterministic(site.h, h_scale * h_raw)

                c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_data, num_data):
                mu, alpha, beta = self.gamma_likelihood(
                    F.logistic4,
                    intensity,
                    (
                        a[*features.T],
                        b[*features.T],
                        g[*features.T],
                        h[*features.T],
                    ),
                    c1[*features.T],
                    c2[*features.T],
                )
                pyro.deterministic(site.mu, mu)

                if self.use_mixture:
                    mixing_distribution = dist.Categorical(
                        probs=jnp.stack([1 - q, q], axis=-1)
                    )
                    component_distributions=[
                        dist.Gamma(concentration=alpha, rate=beta),
                        dist.HalfNormal(scale=(g[*features.T] + h[*features.T]))
                    ]
                    Mixture = dist.MixtureGeneral(
                        mixing_distribution=mixing_distribution,
                        component_distributions=component_distributions
                    )

                pyro.sample(
                    site.obs,
                    (
                        Mixture if self.use_mixture
                        else dist.Gamma(concentration=alpha, rate=beta)
                    ),
                    obs=response
                )

    def hb_mvn_l4_svi(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        mask_features = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))
            mask_features = np.full((*num_features, self.num_response), False)
            mask_features[*features.T, :] = True

        a_loc = pyro.sample(site.a.loc, dist.Normal(7., 2.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(2.))
        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.handlers.mask(mask=mask_features):
            with pyro.plate(site.num_response, self.num_response):
                # # Create a list of plates for each dimension
                # plates = [
                #     pyro.plate(, num_features[i])
                #     for i in range(self.num_features - 1, -1, -1)
                # ]

                with pyro.plate_stack(site.num_features, num_features, rightmost_dim=-2):
                    a_raw = pyro.sample(site.a.raw, dist.Normal(0, 1))
                    a = pyro.deterministic(site.a, a_loc + a_scale * a_raw)

                    b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                    b = pyro.deterministic(site.b, b_scale * b_raw)

                    g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                    g = pyro.deterministic(site.g, g_scale * g_raw)

                    h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                    h = pyro.deterministic(site.h, h_scale * h_raw)

                    c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                    c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                    c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                    c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu, alpha, beta = self.gamma_likelihood(
                        F.logistic4,
                        intensity,
                        (
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                        ),
                        c1[*features.T],
                        c2[*features.T],
                    )
                    pyro.deterministic(site.mu, mu)

                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Gamma(concentration=alpha, rate=beta),
                            dist.HalfNormal(scale=(g[*features.T] + h[*features.T]))
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    pyro.sample(
                        site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.Gamma(concentration=alpha, rate=beta)
                        ),
                        obs=response
                    )


def scratch():
    import pickle
    from jax import random
    import matplotlib.pyplot as plt
    import seaborn as sns
    key = random.key(0)

    corr = dist.LKJ(3, 1).sample(key)
    corr = np.array(corr)
    np.testing.assert_almost_equal(corr.T, corr)

    scale = dist.HalfNormal(5.).sample(key)
    scale = np.array(scale)
    cov = (np.array([scale])[:, None] @ np.array([scale])) * corr

    key, subkey = random.split(key)
    aloc = dist.Normal(6, 2).sample(subkey, (1000,))
    key, subkey = random.split(key)
    ascale = dist.HalfNormal(3).sample(subkey, (1000,))
    key, subkey = random.split(key)
    a = dist.Normal(aloc, ascale).sample(subkey)
    plt.close("all")
    sns.kdeplot(a)
    hdi = pyro.diagnostics.hpdi(a, prob=.95)
    plt.axvline(hdi[0])
    plt.axvline(hdi[1])

    src = "/home/vishu/reports/hbmep/notebooks/rat/lognhb/nhb__4000w_1000s_4c_1t_20d_95a_tm/lcirc/rl_no_v/inf.pkl"
    with open(src, "rb") as f:
        *_, posterior = pickle.load(f)
    
    a = posterior[site.a]
    a = a.mean(axis=0)
    a.shape
    np.nanstd(a)
    np.nanmean(a)
