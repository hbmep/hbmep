import logging

import numpy as np
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist

from hbmep import functional as F, smooth_functional as SF
from hbmep.model import BaseModel, NonHierarchicalBaseModel
from hbmep.util import site

# from hbmep.notebooks.rat.distributions import LKJ
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

    def circ_est_mvn_reference_rl_nov_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        # mask_features = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))
            # mask_features = np.full((*num_features, self.num_response), False)
            # mask_features[*features.T] = True

        num_fixed = 1
        num_delta = num_features[1] - 1

        a_fixed_loc = pyro.sample("a_fixed_loc", dist.Normal(5., 5.))
        a_fixed_scale = pyro.sample("a_fixed_scale", dist.HalfNormal(5.))
        Rho_fixed = pyro.sample("Rho_fixed", dist.LKJ(self.num_response, 1.))

        with pyro.plate("num_fixed", num_fixed):
            with pyro.plate(site.num_features[0], num_features[0]):
                a_fixed_raw = pyro.sample(
                    "a_fixed_raw",
                    dist.MultivariateNormal(0, (a_fixed_scale ** 2) * Rho_fixed)
                )
                a_fixed = pyro.deterministic("a_fixed", a_fixed_loc + a_fixed_raw)

        with pyro.plate("num_delta", num_delta):
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
                a_fixed_plus_delta = a_fixed + a_delta

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        # v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            # with pyro.plate_stack(site.num_features, num_features, rightmost_dim=-2):
            with pyro.plate(site.num_features[1], num_features[1]):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a = pyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed_plus_delta], axis=1)
                    )

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

    def circ_est_mvn_reference_rl_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        num_fixed = 1
        num_delta = num_features[1] - 1

        a_fixed_loc = pyro.sample("a_fixed_loc", dist.Normal(5., 5.))
        a_fixed_scale = pyro.sample("a_fixed_scale", dist.HalfNormal(5.))
        Rho_fixed = pyro.sample("Rho_fixed", dist.LKJ(self.num_response, 1.))

        with pyro.plate("num_fixed", num_fixed):
            with pyro.plate(site.num_features[0], num_features[0]):
                a_fixed_raw = pyro.sample(
                    "a_fixed_raw",
                    dist.MultivariateNormal(0, (a_fixed_scale ** 2) * Rho_fixed)
                )
                a_fixed = pyro.deterministic("a_fixed", a_fixed_loc + a_fixed_raw)

        with pyro.plate("num_delta", num_delta):
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
                a_fixed_plus_delta = a_fixed + a_delta

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[1], num_features[1]):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a = pyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed_plus_delta], axis=1)
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
                            v[*features.T],
                            # h[*features.T],
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

    def robust_circ_est_mvn_reference_rl_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        num_fixed = 1
        num_delta = num_features[1] - 1

        a_fixed_loc = pyro.sample("a_fixed_loc", dist.Normal(5., 5.))
        a_fixed_scale = pyro.sample("a_fixed_scale", dist.HalfNormal(5.))
        Rho_fixed = pyro.sample("Rho_fixed", dist.LKJ(self.num_response, 1.))

        df_fixed = pyro.sample("df_fixed", dist.Gamma(2, 0.1))
        chi_fixed = pyro.sample("chi_fixed", dist.Chi2(3 + df_fixed))

        with pyro.plate("num_fixed", num_fixed):
            with pyro.plate(site.num_features[0], num_features[0]):
                a_fixed_raw = pyro.sample(
                    "a_fixed_raw",
                    dist.MultivariateNormal(0, (a_fixed_scale ** 2) * Rho_fixed)
                )
                a_fixed = pyro.deterministic(
                    "a_fixed",
                    a_fixed_loc + (a_fixed_raw / jnp.sqrt(chi_fixed / (3 + df_fixed)))
                )

        with pyro.plate("num_delta", num_delta):
            a_delta_loc = pyro.sample("a_delta_loc", dist.Normal(0., 5.))
            a_delta_scale = pyro.sample("a_delta_scale", dist.HalfNormal(5.))
            Rho_delta = pyro.sample("Rho_delta", dist.LKJ(self.num_response, 1.))

            df_delta = pyro.sample("df_delta", dist.Gamma(2, 0.1))
            chi_delta = pyro.sample("chi_delta", dist.Chi2(3 + df_delta))

            with pyro.plate(site.num_features[0], num_features[0]):
                a_delta_raw = pyro.sample(
                    "a_delta_raw",
                    dist.MultivariateNormal(
                        0, (a_delta_scale[:, None, None] ** 2) * Rho_delta
                    )
                )
                a_delta = pyro.deterministic(
                    "a_delta",
                    a_delta_loc[None, :, None] 
                    + (
                        a_delta_raw
                        / jnp.sqrt(chi_delta[None, :, None] / (3 + df_delta[None, :, None]))
                    )
                )
                a_fixed_plus_delta = a_fixed + a_delta

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[1], num_features[1]):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a = pyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed_plus_delta], axis=1)
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
                            v[*features.T],
                            # h[*features.T],
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

    def lat_est_mvn_reference_rl_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        num_fixed = 1
        num_delta = num_features[1] - 1

        a_fixed_loc = pyro.sample("a_fixed_loc", dist.Normal(5., 5.))
        a_fixed_scale = pyro.sample("a_fixed_scale", dist.HalfNormal(5.))

        rho_block_fixed = pyro.sample("rho_block_fixed", dist.LKJ(self.num_response, 1.))
        zeros = jnp.zeros((self.num_response, self.num_response))
        rho_fixed = jnp.block([[rho_block_fixed, zeros], [zeros, rho_block_fixed]])

        with pyro.plate("num_fixed", num_fixed):
            with pyro.plate(site.num_features[0], num_features[0]):
                a_fixed_raw = pyro.sample(
                    "a_fixed_raw",
                    dist.MultivariateNormal(0, (a_fixed_scale ** 2) * rho_fixed)
                )
                a_fixed_flat = a_fixed_loc + a_fixed_raw

        with pyro.plate("num_delta", num_delta):
            a_delta_loc = pyro.sample("a_delta_loc", dist.Normal(0., 5.))
            a_delta_scale = pyro.sample("a_delta_scale", dist.HalfNormal(5.))

            rho_block_delta = pyro.sample("rho_block_delta", dist.LKJ(self.num_response, 1.))
            zeros = jnp.zeros((num_delta, self.num_response, self.num_response))
            rho_delta = jnp.block([[rho_block_delta, zeros], [zeros, rho_block_delta]])

            with pyro.plate(site.num_features[0], num_features[0]):
                a_delta_raw = pyro.sample(
                    "a_delta_raw",
                    dist.MultivariateNormal(
                        0, (a_delta_scale[:, None, None] ** 2) * rho_delta
                    )
                )
                a_delta_flat = a_delta_loc[None, :, None] + a_delta_raw
                a_fixed_plus_delta_flat = a_fixed_flat + a_delta_flat

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[2], num_features[2]):
                with pyro.plate(site.num_features[1], num_features[1]):
                    with pyro.plate(site.num_features[0], num_features[0]):
                        a_flat = jnp.concatenate([a_fixed_flat, a_fixed_plus_delta_flat], axis=1) 
                        a = pyro.deterministic(
                            site.a, a_flat.reshape(*num_features, self.num_response)
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
                            v[*features.T],
                            # h[*features.T],
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

    def lat_est_mvn_block_reference_rl_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        num_fixed = 1
        num_delta = num_features[1] - 1

        a_fixed_loc = pyro.sample("a_fixed_loc", dist.Normal(5., 5.))
        a_fixed_scale = pyro.sample("a_fixed_scale", dist.HalfNormal(5.))

        with pyro.plate(site.num_features[2], num_features[2]):
            rho_block_fixed = pyro.sample("rho_block_fixed", dist.LKJ(self.num_response, 1.))

        zeros = jnp.zeros((self.num_response, self.num_response))
        rho_fixed = jnp.block([[rho_block_fixed[0], zeros], [zeros, rho_block_fixed[0]]])

        with pyro.plate("num_fixed", num_fixed):
            with pyro.plate(site.num_features[0], num_features[0]):
                a_fixed_raw = pyro.sample(
                    "a_fixed_raw",
                    dist.MultivariateNormal(0, (a_fixed_scale ** 2) * rho_fixed)
                )
                a_fixed_flat = a_fixed_loc + a_fixed_raw

        with pyro.plate("num_delta", num_delta):
            with pyro.plate(site.num_features[2], num_features[2]):
                rho_block_delta = pyro.sample("rho_block_delta", dist.LKJ(self.num_response, 1.))

        with pyro.plate("num_delta", num_delta):
            a_delta_loc = pyro.sample("a_delta_loc", dist.Normal(0., 5.))
            a_delta_scale = pyro.sample("a_delta_scale", dist.HalfNormal(5.))

            zeros = jnp.zeros((num_delta, self.num_response, self.num_response))
            rho_delta = jnp.block([[rho_block_delta[0], zeros], [zeros, rho_block_delta[0]]])

            with pyro.plate(site.num_features[0], num_features[0]):
                a_delta_raw = pyro.sample(
                    "a_delta_raw",
                    dist.MultivariateNormal(
                        0, (a_delta_scale[:, None, None] ** 2) * rho_delta
                    )
                )
                a_delta_flat = a_delta_loc[None, :, None] + a_delta_raw
                a_fixed_plus_delta_flat = a_fixed_flat + a_delta_flat

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[2], num_features[2]):
                with pyro.plate(site.num_features[1], num_features[1]):
                    with pyro.plate(site.num_features[0], num_features[0]):
                        a_flat = jnp.concatenate([a_fixed_flat, a_fixed_plus_delta_flat], axis=1) 
                        a = pyro.deterministic(
                            site.a, a_flat.reshape(*num_features, self.num_response)
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
                            v[*features.T],
                            # h[*features.T],
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

    def robust_lat_est_mvn_block_reference_rl_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        num_fixed = 1
        num_delta = num_features[1] - 1

        a_fixed_loc = pyro.sample("a_fixed_loc", dist.Normal(5., 5.))
        a_fixed_scale = pyro.sample("a_fixed_scale", dist.HalfNormal(5.))

        with pyro.plate(site.num_features[2], num_features[2]):
            rho_block_fixed = pyro.sample("rho_block_fixed", dist.LKJ(self.num_response, 1.))

        zeros = jnp.zeros((self.num_response, self.num_response))
        rho_fixed = jnp.block([[rho_block_fixed[0], zeros], [zeros, rho_block_fixed[0]]])

        df_fixed = pyro.sample("df_fixed", dist.Gamma(2, 0.1))
        chi_fixed = pyro.sample("chi_fixed", dist.Chi2(3 + df_fixed))

        with pyro.plate("num_fixed", num_fixed):
            with pyro.plate(site.num_features[0], num_features[0]):
                a_fixed_raw = pyro.sample(
                    "a_fixed_raw",
                    dist.MultivariateNormal(0, (a_fixed_scale ** 2) * rho_fixed)
                )
                a_fixed_flat = a_fixed_loc + (a_fixed_raw / jnp.sqrt(chi_fixed / (3 + df_fixed)))

        with pyro.plate("num_delta", num_delta):
            with pyro.plate(site.num_features[2], num_features[2]):
                rho_block_delta = pyro.sample("rho_block_delta", dist.LKJ(self.num_response, 1.))

        with pyro.plate("num_delta", num_delta):
            a_delta_loc = pyro.sample("a_delta_loc", dist.Normal(0., 5.))
            a_delta_scale = pyro.sample("a_delta_scale", dist.HalfNormal(5.))

            zeros = jnp.zeros((num_delta, self.num_response, self.num_response))
            rho_delta = jnp.block([[rho_block_delta[0], zeros], [zeros, rho_block_delta[0]]])

            df_delta = pyro.sample("df_delta", dist.Gamma(2, 0.1))
            chi_delta = pyro.sample("chi_delta", dist.Chi2(3 + df_delta))

            with pyro.plate(site.num_features[0], num_features[0]):
                a_delta_raw = pyro.sample(
                    "a_delta_raw",
                    dist.MultivariateNormal(
                        0, (a_delta_scale[:, None, None] ** 2) * rho_delta
                    )
                )
                a_delta_flat = pyro.deterministic(
                    "a_delta_flat",
                    a_delta_loc[None, :, None] 
                    + (
                        a_delta_raw
                        / jnp.sqrt(chi_delta[None, :, None] / (3 + df_delta[None, :, None]))
                    )
                )
                a_fixed_plus_delta_flat = a_fixed_flat + a_delta_flat

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[2], num_features[2]):
                with pyro.plate(site.num_features[1], num_features[1]):
                    with pyro.plate(site.num_features[0], num_features[0]):
                        a_flat = jnp.concatenate([a_fixed_flat, a_fixed_plus_delta_flat], axis=1) 
                        a = pyro.deterministic(
                            site.a, a_flat.reshape(*num_features, self.num_response)
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
                            v[*features.T],
                            # h[*features.T],
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

    def size_est_mvn_reference_rl_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        num_fixed = 1
        num_delta = num_features[1] - 1

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_fixed", num_fixed):
                a_fixed_loc = pyro.sample("a_fixed_loc", dist.Normal(5., 5.))
                a_fixed_scale = pyro.sample("a_fixed_scale", dist.HalfNormal(5.))
                rho_block_fixed = pyro.sample("rho_block_fixed", dist.LKJ(self.num_response, 1.))

                zeros = jnp.zeros((num_fixed, num_features[2], self.num_response, self.num_response))
                rho_fixed = jnp.block([[rho_block_fixed, zeros], [zeros, rho_block_fixed]])

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_fixed", num_fixed):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a_fixed_raw = pyro.sample(
                        "a_fixed_raw",
                        dist.MultivariateNormal(0, (a_fixed_scale[..., None, None] ** 2) * rho_fixed)
                    )
                    a_fixed_flat = a_fixed_loc[None, ..., None] + a_fixed_raw

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_delta", num_delta):
                a_delta_loc = pyro.sample("a_delta_loc", dist.Normal(0., 5.))
                a_delta_scale = pyro.sample("a_delta_scale", dist.HalfNormal(5.))
                rho_block_delta = pyro.sample("rho_block_delta", dist.LKJ(self.num_response, 1.))

                zeros = jnp.zeros((num_delta, num_features[2], self.num_response, self.num_response))
                rho_delta = jnp.block([[rho_block_delta, zeros], [zeros, rho_block_delta]])

                with pyro.plate(site.num_features[0], num_features[0]):
                    a_delta_raw = pyro.sample(
                        "a_delta_raw",
                        dist.MultivariateNormal(
                            0, (a_delta_scale[..., None, None] ** 2) * rho_delta
                        )
                    )
                    a_delta_flat = a_delta_loc[None, ..., None] + a_delta_raw
                    a_fixed_plus_delta_flat = a_fixed_flat + a_delta_flat

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[3], num_features[3]):
                with pyro.plate(site.num_features[2], num_features[2]):
                    with pyro.plate(site.num_features[1], num_features[1]):
                        with pyro.plate(site.num_features[0], num_features[0]):
                            a_flat = jnp.concatenate([a_fixed_flat, a_fixed_plus_delta_flat], axis=1) 
                            a = pyro.deterministic(
                                site.a, a_flat.reshape(*num_features, self.num_response)
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
                            v[*features.T],
                            # h[*features.T],
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

    def size_est_mvn_block_reference_rl_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        num_fixed = 1
        num_delta = num_features[1] - 1

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_fixed", num_fixed):
                with pyro.plate(site.num_features[3], num_features[3]):
                    rho_block_fixed = pyro.sample("rho_block_fixed", dist.LKJ(self.num_response, 1.))

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_fixed", num_fixed):
                a_fixed_loc = pyro.sample("a_fixed_loc", dist.Normal(5., 5.))
                a_fixed_scale = pyro.sample("a_fixed_scale", dist.HalfNormal(5.))

                zeros = jnp.zeros((num_fixed, num_features[2], self.num_response, self.num_response))
                rho_fixed = jnp.block([[rho_block_fixed[0], zeros], [zeros, rho_block_fixed[0]]])

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_fixed", num_fixed):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a_fixed_raw = pyro.sample(
                        "a_fixed_raw",
                        dist.MultivariateNormal(0, (a_fixed_scale[..., None, None] ** 2) * rho_fixed)
                    )
                    a_fixed_flat = a_fixed_loc[None, ..., None] + a_fixed_raw

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_delta", num_delta):
                with pyro.plate(site.num_features[3], num_features[3]):
                    rho_block_delta = pyro.sample("rho_block_delta", dist.LKJ(self.num_response, 1.))

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_delta", num_delta):
                a_delta_loc = pyro.sample("a_delta_loc", dist.Normal(0., 5.))
                a_delta_scale = pyro.sample("a_delta_scale", dist.HalfNormal(5.))

                zeros = jnp.zeros((num_delta, num_features[2], self.num_response, self.num_response))
                rho_delta = jnp.block([[rho_block_delta[0], zeros], [zeros, rho_block_delta[0]]])

                with pyro.plate(site.num_features[0], num_features[0]):
                    a_delta_raw = pyro.sample(
                        "a_delta_raw",
                        dist.MultivariateNormal(
                            0, (a_delta_scale[..., None, None] ** 2) * rho_delta
                        )
                    )
                    a_delta_flat = a_delta_loc[None, ..., None] + a_delta_raw
                    a_fixed_plus_delta_flat = a_fixed_flat + a_delta_flat

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[3], num_features[3]):
                with pyro.plate(site.num_features[2], num_features[2]):
                    with pyro.plate(site.num_features[1], num_features[1]):
                        with pyro.plate(site.num_features[0], num_features[0]):
                            a_flat = jnp.concatenate([a_fixed_flat, a_fixed_plus_delta_flat], axis=1) 
                            a = pyro.deterministic(
                                site.a, a_flat.reshape(*num_features, self.num_response)
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
                            v[*features.T],
                            # h[*features.T],
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

    def robust_size_est_mvn_block_reference_rl_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        num_fixed = 1
        num_delta = num_features[1] - 1

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_fixed", num_fixed):
                with pyro.plate(site.num_features[3], num_features[3]):
                    rho_block_fixed = pyro.sample("rho_block_fixed", dist.LKJ(self.num_response, 1.))

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_fixed", num_fixed):
                a_fixed_loc = pyro.sample("a_fixed_loc", dist.Normal(5., 5.))
                a_fixed_scale = pyro.sample("a_fixed_scale", dist.HalfNormal(5.))

                zeros = jnp.zeros((num_fixed, num_features[2], self.num_response, self.num_response))
                rho_fixed = jnp.block([[rho_block_fixed[0], zeros], [zeros, rho_block_fixed[0]]])

                df_fixed = pyro.sample("df_fixed", dist.Gamma(2, 0.1))
                chi_fixed = pyro.sample("chi_fixed", dist.Chi2(3 + df_fixed))

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_fixed", num_fixed):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a_fixed_raw = pyro.sample(
                        "a_fixed_raw",
                        dist.MultivariateNormal(0, (a_fixed_scale[..., None, None] ** 2) * rho_fixed)
                    )
                    a_fixed_flat = (
                        a_fixed_loc[None, ..., None]
                        + (
                            a_fixed_raw
                            / jnp.sqrt(chi_fixed[None, ..., None] / (3 + df_fixed[None, ..., None]))
                        )
                    )

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_delta", num_delta):
                with pyro.plate(site.num_features[3], num_features[3]):
                    rho_block_delta = pyro.sample("rho_block_delta", dist.LKJ(self.num_response, 1.))

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_delta", num_delta):
                a_delta_loc = pyro.sample("a_delta_loc", dist.Normal(0., 5.))
                a_delta_scale = pyro.sample("a_delta_scale", dist.HalfNormal(5.))

                zeros = jnp.zeros((num_delta, num_features[2], self.num_response, self.num_response))
                rho_delta = jnp.block([[rho_block_delta[0], zeros], [zeros, rho_block_delta[0]]])

                df_delta = pyro.sample("df_delta", dist.Gamma(2, 0.1))
                chi_delta = pyro.sample("chi_delta", dist.Chi2(3 + df_delta))

                with pyro.plate(site.num_features[0], num_features[0]):
                    a_delta_raw = pyro.sample(
                        "a_delta_raw",
                        dist.MultivariateNormal(
                            0, (a_delta_scale[..., None, None] ** 2) * rho_delta
                        )
                    )
                    a_delta_flat = pyro.deterministic(
                        "a_delta_flat",
                        a_delta_loc[None, ..., None] 
                        + (
                            a_delta_raw
                            / jnp.sqrt(chi_delta[None, ..., None] / (3 + df_delta[None, ..., None]))
                        )
                    )
                    a_fixed_plus_delta_flat = a_fixed_flat + a_delta_flat

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[3], num_features[3]):
                with pyro.plate(site.num_features[2], num_features[2]):
                    with pyro.plate(site.num_features[1], num_features[1]):
                        with pyro.plate(site.num_features[0], num_features[0]):
                            a_flat = jnp.concatenate([a_fixed_flat, a_fixed_plus_delta_flat], axis=1) 
                            a = pyro.deterministic(
                                site.a, a_flat.reshape(*num_features, self.num_response)
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
                            v[*features.T],
                            # h[*features.T],
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

    def hb_mvn_rl_masked(self, intensity, features, response=None, **kw):
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
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

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

                v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                v = pyro.deterministic(site.v, v_scale * v_raw)

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
                            v[*features.T],
                            # h[*features.T],
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

    def hb_l4_masked(self, intensity, features, response=None, **kw):
        # Used for running the logistic-4 model independently on each response
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        mask_features = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))
            mask_features = np.full((*num_features, self.num_response), False)
            mask_features[*features.T] = True

        a_loc = pyro.sample(site.a.loc, dist.Normal(5., 5.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(5.))

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
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

    def hb_l4_masked_mmax0(self, intensity, features, response=None, **kw):
        # Used for running the logistic-4 model independently on each response,
        # while estimating M_max
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        mask_features = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))
            mask_features = np.full((*num_features, self.num_response), False)
            mask_features[*features.T] = True

        a_loc = pyro.sample(site.a.loc, dist.Normal(5., 5.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(5.))

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        # h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            # h_prior = pyro.sample("h_prior", dist.HalfNormal(self.h_prior))
            h_max_global = pyro.sample("h_max_global", dist.Exponential(self.h_prior))

            with pyro.plate(site.num_features[0], num_features[0], dim=-3):
                h_max_fraction = pyro.sample("h_max_fraction", dist.Beta(concentration1=self.concentration1, concentration0=1))
                h_max = pyro.deterministic("h_max", h_max_fraction * h_max_global)

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[1], num_features[1]):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a_raw = pyro.sample(site.a.raw, dist.Normal(0, 1))
                    a = pyro.deterministic(site.a, a_loc + a_scale * a_raw)

                    b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                    b = pyro.deterministic(site.b, b_scale * b_raw)

                    g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                    g = pyro.deterministic(site.g, g_scale * g_raw)

                    h_fraction = pyro.sample("h_fraction", dist.Beta(concentration1=1, concentration0=1))
                    h = pyro.deterministic(site.h, h_fraction * h_max)

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

    def hb_l4_masked_mmax1(self, intensity, features, response=None, **kw):
        # Used for running the logistic-4 model independently on each response,
        # while estimating M_max
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        mask_features = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))
            mask_features = np.full((*num_features, self.num_response), False)
            mask_features[*features.T] = True

        a_loc = pyro.sample(site.a.loc, dist.Normal(5., 5.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(5.))

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        # h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            # h_prior = pyro.sample("h_prior", dist.HalfNormal(self.h_prior))
            h_max_global = pyro.sample("h_max_global", dist.Exponential(self.h_prior))
            concentration1 = pyro.sample("concentration1", dist.HalfNormal(5))

            with pyro.plate(site.num_features[0], num_features[0], dim=-3):
                h_max_fraction = pyro.sample("h_max_fraction", dist.Beta(concentration1=1 + concentration1, concentration0=1))
                h_max = pyro.deterministic("h_max", h_max_fraction * h_max_global)

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[1], num_features[1]):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a_raw = pyro.sample(site.a.raw, dist.Normal(0, 1))
                    a = pyro.deterministic(site.a, a_loc + a_scale * a_raw)

                    b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                    b = pyro.deterministic(site.b, b_scale * b_raw)

                    g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                    g = pyro.deterministic(site.g, g_scale * g_raw)

                    # h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                    # h = pyro.deterministic(site.h, h_scale * h_raw)
                    h_fraction = pyro.sample("h_fraction", dist.Beta(concentration1=1, concentration0=1))
                    h = pyro.deterministic(site.h, h_fraction * h_max)

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

    def hb_l4_masked_mmax2(self, intensity, features, response=None, **kw):
        # Used for running the logistic-4 model independently on each response,
        # while estimating M_max
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        mask_features = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))
            mask_features = np.full((*num_features, self.num_response), False)
            mask_features[*features.T] = True

        a_loc = pyro.sample(site.a.loc, dist.Normal(5., 5.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(5.))

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        # h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            # h_prior = pyro.sample("h_prior", dist.HalfNormal(self.h_prior))
            h_max_global = pyro.sample("h_max_global", dist.Exponential(self.h_prior))

            with pyro.plate(site.num_features[0], num_features[0], dim=-3):
                h_max_fraction = pyro.sample("h_max_fraction", dist.Beta(concentration1=self.concentration1, concentration0=1))
                h_max = pyro.deterministic("h_max", h_max_fraction * h_max_global)

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[1], num_features[1]):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a_raw = pyro.sample(site.a.raw, dist.Normal(0, 1))
                    a = pyro.deterministic(site.a, a_loc + a_scale * a_raw)

                    b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                    b = pyro.deterministic(site.b, b_scale * b_raw)

                    g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                    g = pyro.deterministic(site.g, g_scale * g_raw)

                    h_fraction = pyro.sample("h_fraction", dist.Beta(concentration1=1, concentration0=self.concentration1))
                    h = pyro.deterministic(site.h, h_fraction * h_max)

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

    def hb_l4_masked_hmax(self, intensity, features, response=None, **kw):
        # Used for running the logistic-4 model independently on each response,
        # while estimating M_max
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None: mask_obs = np.invert(np.isnan(response))

        a_loc = pyro.sample(site.a.loc, dist.Normal(5., 5.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(5.))

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        # h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[0], num_features[0], dim=-3):
                h_max = pyro.sample("h_max", dist.Exponential(self.h_prior))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[1], num_features[1]):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a_raw = pyro.sample(site.a.raw, dist.Normal(0, 1))
                    a = pyro.deterministic(site.a, a_loc + a_scale * a_raw)

                    b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                    b = pyro.deterministic(site.b, b_scale * b_raw)

                    g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                    g = pyro.deterministic(site.g, g_scale * g_raw)

                    h_fraction = pyro.sample("h_fraction", dist.Beta(concentration1=1, concentration0=1))
                    h = pyro.deterministic(site.h, h_fraction * h_max)

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

    def hb_l4_masked_hmaxPooled(self, intensity, features, response=None, **kw):
        # Used for running the logistic-4 model independently on each response,
        # while estimating M_max
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None: mask_obs = np.invert(np.isnan(response))

        a_loc = pyro.sample(site.a.loc, dist.Normal(5., 5.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(5.))

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        # h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            h_max_loc = pyro.sample("h_max_loc", dist.Exponential(.1))
            h_max_scale = pyro.sample("h_max_scale", dist.HalfNormal(5.))

            with pyro.plate(site.num_features[0], num_features[0], dim=-3):
                h_max = pyro.sample("h_max", dist.TruncatedNormal(h_max_loc, h_max_scale, low=0))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[1], num_features[1]):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a_raw = pyro.sample(site.a.raw, dist.Normal(0, 1))
                    a = pyro.deterministic(site.a, a_loc + a_scale * a_raw)

                    b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                    b = pyro.deterministic(site.b, b_scale * b_raw)

                    g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                    g = pyro.deterministic(site.g, g_scale * g_raw)

                    h_fraction = pyro.sample("h_fraction", dist.Beta(concentration1=1, concentration0=1))
                    h = pyro.deterministic(site.h, h_fraction * h_max)

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

    def hb_rl_masked_hmaxPooled(self, intensity, features, response=None, **kw):
        # Used for running the rl model independently on each response,
        # while estimating h_max
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None: mask_obs = np.invert(np.isnan(response))

        a_loc = pyro.sample(site.a.loc, dist.Normal(5., 5.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(5.))

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        # h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            h_max_loc = pyro.sample("h_max_loc", dist.Exponential(.1))
            h_max_scale = pyro.sample("h_max_scale", dist.HalfNormal(5.))

            with pyro.plate(site.num_features[0], num_features[0], dim=-3):
                h_max = pyro.sample("h_max", dist.TruncatedNormal(h_max_loc, h_max_scale, low=0))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[1], num_features[1]):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a_raw = pyro.sample(site.a.raw, dist.Normal(0, 1))
                    a = pyro.deterministic(site.a, a_loc + a_scale * a_raw)

                    b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                    b = pyro.deterministic(site.b, b_scale * b_raw)

                    g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                    g = pyro.deterministic(site.g, g_scale * g_raw)

                    h_fraction = pyro.sample("h_fraction", dist.Beta(concentration1=1, concentration0=1))
                    h = pyro.deterministic(site.h, h_fraction * h_max)

                    v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                    v = pyro.deterministic(site.v, v_scale * v_raw)

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
                            v[*features.T],
                            # h[*features.T],
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

    def hb_l5_masked_hmaxPooled(self, intensity, features, response=None, **kw):
        # Used for running the logistic-5 model independently on each response,
        # while estimating h_max
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None: mask_obs = np.invert(np.isnan(response))

        a_loc = pyro.sample(site.a.loc, dist.Normal(5., 5.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(5.))

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        # h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            h_max_loc = pyro.sample("h_max_loc", dist.Exponential(.1))
            h_max_scale = pyro.sample("h_max_scale", dist.HalfNormal(5.))

            with pyro.plate(site.num_features[0], num_features[0], dim=-3):
                h_max = pyro.sample("h_max", dist.TruncatedNormal(h_max_loc, h_max_scale, low=0))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[1], num_features[1]):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a_raw = pyro.sample(site.a.raw, dist.Normal(0, 1))
                    a = pyro.deterministic(site.a, a_loc + a_scale * a_raw)

                    b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                    b = pyro.deterministic(site.b, b_scale * b_raw)

                    g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                    g = pyro.deterministic(site.g, g_scale * g_raw)

                    h_fraction = pyro.sample("h_fraction", dist.Beta(concentration1=1, concentration0=1))
                    h = pyro.deterministic(site.h, h_fraction * h_max)

                    v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                    v = pyro.deterministic(site.v, v_scale * v_raw)

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
                        F.logistic5,
                        intensity,
                        (
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            v[*features.T],
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
