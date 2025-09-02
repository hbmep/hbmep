import numpy as np
from jax import numpy as jnp
import numpyro as pyro
from numpyro import distributions as dist
from hbmep.model import BaseModel, NonHierarchicalBaseModel

from hbmep.notebooks.rat.util import get_subname
import functional as F
from util import Site as site

EPS = 1e-3


class nHB(NonHierarchicalBaseModel):
    def __init__(self, *args, **kw):
        super(nHB, self).__init__(*args, **kw)
        self.run_id = None
        self.test_run = True
        self.use_mixture = True
        self.n_jobs = -1

    @property
    def name(self): return get_subname(self)

    @name.setter
    def name(self, value): return value

    def non_hierarchical(self, intensity, features, response=None, **kw):
        intensity = intensity[..., 0]

        b1_log = pyro.sample(site.b1.log, dist.Normal(0, 10))
        b1 = pyro.deterministic(site.b1, jnp.exp(b1_log))

        b2_log = pyro.sample(site.b2.log, dist.Normal(0, 10))
        b2 = pyro.deterministic(site.b2, jnp.exp(b2_log))

        b3_log = pyro.sample(site.b3.log, dist.Normal(0, 10))
        b3 = pyro.deterministic(site.b3, jnp.exp(b3_log))

        b4_log = pyro.sample(site.b4.log, dist.Normal(0, 10))
        b4 = pyro.deterministic(site.b4, jnp.exp(b4_log))

        sigma = pyro.sample(site.sigma, dist.HalfNormal(10.))

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        mu = pyro.deterministic(site.mu, F.ro1(intensity, b3, b4, b1, b2))
        loc = jnp.log(mu)
        scale = sigma

        if self.use_mixture:
            mixing_distribution = dist.Categorical(
                probs=jnp.stack([1 - q, q], axis=-1)
            )
            component_distributions=[
                dist.LogNormal(loc=loc, scale=scale),
                dist.HalfNormal(scale=5)
            ]
            Mixture = dist.MixtureGeneral(
                mixing_distribution=mixing_distribution,
                component_distributions=component_distributions
            )

        # Observations
        pyro.sample(
            site.obs,
            (
                Mixture if self.use_mixture
                else dist.LogNormal(loc=loc, scale=scale)
            ),
            obs=response
        )


class HB(BaseModel):
    def __init__(self, *args, **kw):
        super(HB, self).__init__(*args, **kw)
        self.run_id = None
        self.test_run = True
        self.use_mixture = True

    @property
    def name(self): return get_subname(self)

    @name.setter
    def name(self, value): return value

    def hierarchical_sharedb1b4(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        b1_log_param = pyro.sample(site.b1.log + "_param", dist.Normal(0, 10))
        b4_log_param = pyro.sample(site.b4.log + "_param", dist.Normal(0, 10))

        b2_log_scale = pyro.sample(site.b2.log.scale, dist.HalfNormal(10))
        b3_log_scale = pyro.sample(site.b3.log.scale, dist.HalfNormal(10))

        sigma_scale = pyro.sample(site.sigma.scale, dist.HalfNormal(10))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate_stack(
                site.num_features, num_features, rightmost_dim=-2
            ):
                b1_log = pyro.deterministic(
                    site.b1.log,
                    jnp.ones((*num_features, self.num_response)) * b1_log_param
                )
                b1 = pyro.deterministic(site.b1, jnp.exp(b1_log))

                b2_log_raw = pyro.sample(
                    site.b2.log.raw, dist.Normal(0, 1)
                )
                b2_log = pyro.deterministic(
                    site.b2.log, b2_log_scale * b2_log_raw
                )
                b2 = pyro.deterministic(site.b2, jnp.exp(b2_log))

                b3_log_raw = pyro.sample(
                    site.b3.log.raw, dist.Normal(0, 1)
                )
                b3_log = pyro.deterministic(
                    site.b3.log, b3_log_scale * b3_log_raw
                )
                b3 = pyro.deterministic(site.b3, jnp.exp(b3_log))

                b4_log = pyro.deterministic(
                    site.b4.log,
                    jnp.ones((*num_features, self.num_response)) * b4_log_param
                )
                b4 = pyro.deterministic(site.b4, jnp.exp(b4_log))

                sigma_raw = pyro.sample(site.sigma.raw, dist.HalfNormal(1))
                sigma = pyro.deterministic(site.sigma, sigma_scale * sigma_raw)

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_data, num_data):
                mu = pyro.deterministic(
                    site.mu,
                    F.ro1(
                        intensity,
                        b3[*features.T],
                        b4[*features.T],
                        b1[*features.T],
                        b2[*features.T]
                    )
                )
                loc = jnp.log(mu)
                scale = sigma[*features.T]

                if self.use_mixture:
                    mixing_distribution = dist.Categorical(
                        probs=jnp.stack([1 - q, q], axis=-1)
                    )
                    component_distributions=[
                        dist.LogNormal(loc=loc, scale=scale),
                        dist.HalfNormal(scale=5)
                    ]
                    Mixture = dist.MixtureGeneral(
                        mixing_distribution=mixing_distribution,
                        component_distributions=component_distributions
                    )

                # Observations
                pyro.sample(
                    site.obs,
                    (
                        Mixture if self.use_mixture
                        else dist.LogNormal(loc=loc, scale=scale)
                    ),
                    obs=response
                )

    # def hierarchical_strongPartialPoolb1b4(self, intensity, features, response=None, **kw):
    #     num_data = intensity.shape[0]
    #     num_features = np.max(features, axis=0) + 1

    #     # b1_log_param = pyro.sample(site.b1.log + "_param", dist.Normal(0, 10))
    #     # b4_log_param = pyro.sample(site.b4.log + "_param", dist.Normal(0, 10))

    #     b2_log_scale = pyro.sample(site.b2.log.scale, dist.HalfNormal(10))
    #     b3_log_scale = pyro.sample(site.b3.log.scale, dist.HalfNormal(10))

    #     b2_log_scale = pyro.sample(site.b2.log.scale, dist.HalfNormal(10))
    #     b3_log_scale = pyro.sample(site.b3.log.scale, dist.HalfNormal(10))

    #     sigma_scale = pyro.sample(site.sigma.scale, dist.HalfNormal(10))

    #     with pyro.plate(site.num_response, self.num_response):
    #         with pyro.plate_stack(
    #             site.num_features, num_features, rightmost_dim=-2
    #         ):
    #             b1_log = pyro.deterministic(
    #                 site.b1.log,
    #                 jnp.ones((*num_features, self.num_response)) * b1_log_param
    #             )
    #             b1 = pyro.deterministic(site.b1, jnp.exp(b1_log))

    #             b2_log_raw = pyro.sample(
    #                 site.b2.log.raw, dist.Normal(0, 1)
    #             )
    #             b2_log = pyro.deterministic(
    #                 site.b2.log, b2_log_scale * b2_log_raw
    #             )
    #             b2 = pyro.deterministic(site.b2, jnp.exp(b2_log))

    #             b3_log_raw = pyro.sample(
    #                 site.b3.log.raw, dist.Normal(0, 1)
    #             )
    #             b3_log = pyro.deterministic(
    #                 site.b3.log, b3_log_scale * b3_log_raw
    #             )
    #             b3 = pyro.deterministic(site.b3, jnp.exp(b3_log))

    #             b4_log = pyro.deterministic(
    #                 site.b4.log,
    #                 jnp.ones((*num_features, self.num_response)) * b4_log_param
    #             )
    #             b4 = pyro.deterministic(site.b4, jnp.exp(b4_log))

    #             sigma_raw = pyro.sample(site.sigma.raw, dist.HalfNormal(1))
    #             sigma = pyro.deterministic(site.sigma, sigma_scale * sigma_raw)

    #     if self.use_mixture: q = pyro.sample(
    #         site.outlier_prob, dist.Uniform(0., 0.01)
    #     )

    #     with pyro.plate(site.num_response, self.num_response):
    #         with pyro.plate(site.num_data, num_data):
    #             mu = pyro.deterministic(
    #                 site.mu,
    #                 F.ro1(
    #                     intensity,
    #                     b3[*features.T],
    #                     b4[*features.T],
    #                     b1[*features.T],
    #                     b2[*features.T]
    #                 )
    #             )
    #             loc = jnp.log(mu)
    #             scale = sigma[*features.T]

    #             if self.use_mixture:
    #                 mixing_distribution = dist.Categorical(
    #                     probs=jnp.stack([1 - q, q], axis=-1)
    #                 )
    #                 component_distributions=[
    #                     dist.LogNormal(loc=loc, scale=scale),
    #                     dist.HalfNormal(scale=5)
    #                 ]
    #                 Mixture = dist.MixtureGeneral(
    #                     mixing_distribution=mixing_distribution,
    #                     component_distributions=component_distributions
    #                 )

    #             # Observations
    #             pyro.sample(
    #                 site.obs,
    #                 (
    #                     Mixture if self.use_mixture
    #                     else dist.LogNormal(loc=loc, scale=scale)
    #                 ),
    #                 obs=response
    #             )
