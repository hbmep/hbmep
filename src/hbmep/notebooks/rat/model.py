import logging

import numpy as np
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist

from hbmep import functional as F, smooth_functional as SF
from hbmep.model import BaseModel, NonHierarchicalBaseModel
from hbmep.util import site

from hbmep.notebooks.rat.util import get_subname

logger = logging.getLogger(__name__)
EPS = 1e-3


class Estimation(BaseModel):
    def __init__(self, *args, **kw):
        super(Estimation, self).__init__(*args, **kw)
        self.use_mixture = False
        self.run_id = None
        self.test_run = False

    @property
    def name(self): return get_subname(self)

    @name.setter
    def name(self, value): return value

    def circ_ln_est_mvn_reference_rl_nov_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        # mask_features = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))
            # mask_features = np.full((*num_features, self.num_response), False)
            # mask_features[*features.T] = True

        a_fixed_loc = pyro.sample("a_fixed_loc", dist.Normal(5., 5.))
        a_fixed_scale = pyro.sample("a_fixed_scale", dist.HalfNormal(5.))
        Rho_fixed = pyro.sample("Rho_fixed" ,dist.LKJ(self.num_response, 1.))

        with pyro.plate(site.num_features[0], num_features[0], dim=-2):
                a_fixed_raw = pyro.sample(
                    "a_fixed_raw",
                    dist.MultivariateNormal(0, (a_fixed_scale ** 2) * Rho_fixed)
                )
                a_fixed = pyro.deterministic("a_fixed", a_fixed_loc + a_fixed_raw)

        with pyro.plate(site.num_features[1], num_features[1]):
            a_delta_loc = pyro.sample("a_delta_loc", dist.Normal(0., 5.))
            a_delta_scale = pyro.sample("a_delta_scale", dist.HalfNormal(5.))
            Rho_delta = pyro.sample("Rho_delta", dist.LKJ(self.num_response, 1.))

            with pyro.plate(site.num_features[0], num_features[0]):
                a_delta_raw = pyro.sample(
                    "a_delta_raw",
                    dist.MultivariateNormal(
                        0, (a_delta_scale[:, None, None] ** 2) * Rho_delta
                    )
                )
                a_delta = pyro.deterministic("a_delta", a_delta_loc[None, :, None] + a_delta_raw)

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        # v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        # c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            # with pyro.plate_stack(site.num_features, num_features, rightmost_dim=-2):
            with pyro.plate(site.num_features[1], num_features[1]):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a = pyro.deterministic(site.a, a_fixed + a_delta)

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

                    # c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                    # c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu = pyro.deterministic(
                        site.mu,
                        SF.rectified_logistic(
                            intensity,
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            # v[*features.T],
                            h[*features.T],
                            EPS
                        )
                    )
                    loc = jnp.log(mu)
                    scale = c1[*features.T]

                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Normal(loc=loc, scale=scale),
                            dist.Normal(loc=0, scale=g[*features.T] + h[*features.T])
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    obs_log = pyro.sample(
                        site.obs.log,
                        (
                            Mixture if self.use_mixture
                            else dist.Normal(loc=loc, scale=scale)
                        ),
                        obs=jnp.log(response) if response is not None else None
                    )
                    pyro.deterministic(site.obs, jnp.exp(obs_log))


class HB(BaseModel):
    def __init__(self, *args, **kw):
        super(HB, self).__init__(*args, **kw)
        self.use_mixture = False
        self.run_id = None
        self.test_run = False

    @property
    def name(self): return get_subname(self)

    @name.setter
    def name(self, value): return value

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

        a_loc = pyro.sample(site.a.loc, dist.Normal(5., 5.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(5.))
        Rho = pyro.sample("Rho" ,dist.LKJ(self.num_response, 1.))

        with pyro.plate_stack(site.num_features, num_features, rightmost_dim=-1):
            a_raw = pyro.sample(
                site.a.raw,
                dist.MultivariateNormal(0, (a_scale ** 2) * Rho)
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

        a_loc = pyro.sample(site.a.loc, dist.Normal(5., 5.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(5.))
        Rho = pyro.sample("Rho" ,dist.LKJ(self.num_response, 1.))

        with pyro.plate_stack(site.num_features, num_features, rightmost_dim=-1):
            a_raw = pyro.sample(
                site.a.raw,
                dist.MultivariateNormal(0, (a_scale ** 2) * Rho)
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

        with pyro.handlers.mask(mask=mask_obs):
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

    def ln_hb_mvn_l4_masked(self, intensity, features, response=None, **kw):
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
        # c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        a_loc = pyro.sample(site.a.loc, dist.Normal(5., 5.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(5.))
        Rho = pyro.sample("Rho" ,dist.LKJ(self.num_response, 1.))

        with pyro.plate_stack(site.num_features, num_features, rightmost_dim=-1):
            a_raw = pyro.sample(
                site.a.raw,
                dist.MultivariateNormal(0, (a_scale ** 2) * Rho)
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

                # c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                # c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu = pyro.deterministic(
                        site.mu,
                        F.logistic4(
                            intensity,
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                        )
                    )
                    loc = jnp.log(mu)
                    scale = c1[*features.T]

                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Normal(loc=loc, scale=scale),
                            dist.Normal(loc=0, scale=g[*features.T] + h[*features.T])
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    obs_log = pyro.sample(
                        site.obs.log,
                        (
                            Mixture if self.use_mixture
                            else dist.Normal(loc=loc, scale=scale)
                        ),
                        obs=jnp.log(response) if response is not None else None
                    )
                    pyro.deterministic(site.obs, jnp.exp(obs_log))

    def ln_hb_mvn_rl_nov_masked(self, intensity, features, response=None, **kw):
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
        # c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        a_loc = pyro.sample(site.a.loc, dist.Normal(5., 5.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(5.))
        Rho = pyro.sample("Rho" ,dist.LKJ(self.num_response, 1.))

        with pyro.plate_stack(site.num_features, num_features, rightmost_dim=-1):
            a_raw = pyro.sample(
                site.a.raw,
                dist.MultivariateNormal(0, (a_scale ** 2) * Rho)
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

                # c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                # c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu = pyro.deterministic(
                        site.mu,
                        SF.rectified_logistic(
                            intensity,
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            # v[*features.T],
                            h[*features.T],
                            EPS
                        )
                    )
                    loc = jnp.log(mu)
                    scale = c1[*features.T]

                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Normal(loc=loc, scale=scale),
                            dist.Normal(loc=0, scale=g[*features.T] + h[*features.T])
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    obs_log = pyro.sample(
                        site.obs.log,
                        (
                            Mixture if self.use_mixture
                            else dist.Normal(loc=loc, scale=scale)
                        ),
                        obs=jnp.log(response) if response is not None else None
                    )
                    pyro.deterministic(site.obs, jnp.exp(obs_log))


class nHB(NonHierarchicalBaseModel):
    def __init__(self, *args, **kw):
        super(nHB, self).__init__(*args, **kw)
        self._name = "nhb"
        self.n_jobs = -1
        self.use_mixture = True
        self.test_run = False

    @property
    def name(self): return f"{self._name}__{get_subname(self)}"

    @name.setter
    def name(self, value): return  value

    def rectified_logistic(self, intensity, features, response=None, **kw):
        intensity = intensity[..., 0]
        a = pyro.sample(site.a, dist.Normal(5., 5.))
        b = pyro.sample(site.b, dist.HalfNormal(scale=5.))

        g = pyro.sample(site.g, dist.HalfNormal(scale=.1))
        h = pyro.sample(site.h, dist.HalfNormal(scale=5.))
        v = pyro.sample(site.v, dist.HalfNormal(scale=1.))

        c1 = pyro.sample(site.c1, dist.HalfNormal(scale=5.))
        c2 = pyro.sample(site.c2, dist.HalfNormal(scale=.5))

        if self.use_mixture:
            q = pyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

        mu, alpha, beta = self.gamma_likelihood(
            F.rectified_logistic,
            intensity,
            (a, b, g, h, v),
            c1, c2
        )
        pyro.deterministic(site.mu, mu)

        if self.use_mixture:
            mixing_distribution = dist.Categorical(
                probs=jnp.stack([1 - q, q], axis=-1)
            )
            component_distributions=[
                dist.Gamma(concentration=alpha, rate=beta),
                dist.HalfNormal(scale=(g + h))
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

    def rl_no_v(self, intensity, features, response=None, **kw):
        intensity = intensity[..., 0]
        a = pyro.sample(site.a, dist.Normal(5., 5.))
        b = pyro.sample(site.b, dist.HalfNormal(scale=5.))

        g = pyro.sample(site.g, dist.HalfNormal(scale=.1))
        h = pyro.sample(site.h, dist.HalfNormal(scale=5.))
        # v = pyro.sample(site.v, dist.HalfNormal(scale=1.))

        c1 = pyro.sample(site.c1, dist.HalfNormal(scale=5.))
        c2 = pyro.sample(site.c2, dist.HalfNormal(scale=.5))

        if self.use_mixture:
            q = pyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

        mu, alpha, beta = self.gamma_likelihood(
            F.rectified_logistic,
            intensity,
            (a, b, g, h, h),
            c1, c2
        )
        pyro.deterministic(site.mu, mu)

        if self.use_mixture:
            mixing_distribution = dist.Categorical(
                probs=jnp.stack([1 - q, q], axis=-1)
            )
            component_distributions=[
                dist.Gamma(concentration=alpha, rate=beta),
                dist.HalfNormal(scale=(g + h))
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

    def logistic4(self, intensity, features, response=None, **kw):
        intensity = intensity[..., 0]
        a = pyro.sample(site.a, dist.Normal(5., 5.))
        b = pyro.sample(site.b, dist.HalfNormal(scale=5.))

        g = pyro.sample(site.g, dist.HalfNormal(scale=.1))
        h = pyro.sample(site.h, dist.HalfNormal(scale=5.))

        c1 = pyro.sample(site.c1, dist.HalfNormal(scale=5.))
        c2 = pyro.sample(site.c2, dist.HalfNormal(scale=.5))

        if self.use_mixture:
            q = pyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

        mu, alpha, beta = self.gamma_likelihood(
            F.logistic4,
            intensity,
            (a, b, g, h),
            c1, c2
        )
        pyro.deterministic(site.mu, mu)

        if self.use_mixture:
            mixing_distribution = dist.Categorical(
                probs=jnp.stack([1 - q, q], axis=-1)
            )
            component_distributions=[
                dist.Gamma(concentration=alpha, rate=beta),
                dist.HalfNormal(scale=(g + h))
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

    def rl_no_v_ln(self, intensity, features, response=None, **kw):
        intensity = intensity[..., 0]
        a = pyro.sample(site.a, dist.Normal(5., 5.))
        b = pyro.sample(site.b, dist.HalfNormal(scale=5.))

        g = pyro.sample(site.g, dist.HalfNormal(scale=.1))
        h = pyro.sample(site.h, dist.HalfNormal(scale=5.))
        # v = pyro.sample(site.v, dist.HalfNormal(scale=1.))

        c1 = pyro.sample(site.c1, dist.HalfNormal(scale=5.))
        # c2 = pyro.sample(site.c2, dist.HalfNormal(scale=.5))

        mu = pyro.deterministic(
            site.mu,
            F.rectified_logistic(
                intensity, a, b, g, h, h
            )
        )
        loc = jnp.log(mu)

        # Observation
        pyro.sample(
            "obs",
            dist.Normal(loc=loc, scale=c1),
            obs=jnp.log(response) if response is not None else None
        )
