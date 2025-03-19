import numpy as np
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist

from hbmep import functional as F, smooth_functional as SF
from hbmep.model import BaseModel
from hbmep.util import site, timing

from hbmep.notebooks.rat.util import get_subname
EPS = 1e-3


class HB(BaseModel):
    def __init__(self, *args, **kw):
        super(HB, self).__init__(*args, **kw)
        self._name = "hb"

        """ Set these """
        # self._model = self.rl_no_v
        # self._model = self.rl
        self._model = self.rl_lg
        self.use_mixture = False

        self.mcmc_params = {
            "num_chains": 4,
            # "thinning": 1,
            # "num_warmup": 1000,
            # "num_samples": 1000,
            "thinning": 4,
            "num_warmup": 4000,
            "num_samples": 4000,
        }
        # self.nuts_params["max_tree_depth"] = (15, 15)
        self.nuts_params["target_accept_prob"] = .95

    @property
    def name(self): return f"{self._name}__{get_subname(self)}"

    @name.setter
    def name(self, value): return  value

    def rectified_logistic(self, intensity, features, response=None, **kwargs):
        pass

    def rl_no_v(self, intensity, features, response=None, **kwargs):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        a_loc = pyro.sample(site.a.loc, dist.Normal(5., 5.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(5.))

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        # v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[1], num_features[1]):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a_raw = pyro.sample(site.a.raw, dist.Normal(0, 1))
                    a = pyro.deterministic(site.a, a_loc + a_raw * a_scale)

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

        if self.use_mixture: q = pyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_data, num_data):
                mu, alpha, beta = self.gamma_likelihood(
                    SF.rectified_logistic,
                    intensity,
                    (
                        a[feature0, feature1],
                        b[feature0, feature1],
                        g[feature0, feature1],
                        h[feature0, feature1],
                        # v[feature0, feature1],
                        h[feature0, feature1],
                        EPS,
                    ),
                    c1[feature0, feature1],
                    c2[feature0, feature1],
                )
                pyro.deterministic(site.mu, mu)

                if self.use_mixture:
                    mixing_distribution = dist.Categorical(
                        probs=jnp.stack([1 - q, q], axis=-1)
                    )
                    component_distributions=[
                        dist.Gamma(concentration=alpha, rate=beta),
                        dist.HalfNormal(g[feature0, feature1] + h[feature0, feature1])
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

    def rl(self, intensity, features, response=None, **kwargs):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        a_loc = pyro.sample(site.a.loc, dist.Normal(5., 5.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(5.))

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[1], num_features[1]):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a_raw = pyro.sample(site.a.raw, dist.Normal(0, 1))
                    a = pyro.deterministic(site.a, a_loc + a_raw * a_scale)

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

        if self.use_mixture: q = pyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_data, num_data):
                mu, alpha, beta = self.gamma_likelihood(
                    SF.rectified_logistic,
                    intensity,
                    (
                        a[feature0, feature1],
                        b[feature0, feature1],
                        g[feature0, feature1],
                        h[feature0, feature1],
                        v[feature0, feature1],
                        EPS,
                    ),
                    c1[feature0, feature1],
                    c2[feature0, feature1],
                )
                pyro.deterministic(site.mu, mu)

                if self.use_mixture:
                    mixing_distribution = dist.Categorical(
                        probs=jnp.stack([1 - q, q], axis=-1)
                    )
                    component_distributions=[
                        dist.Gamma(concentration=alpha, rate=beta),
                        dist.HalfNormal(g[feature0, feature1] + h[feature0, feature1])
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

    def rl_lg(self, intensity, features, response=None, **kwargs):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        a_loc = pyro.sample(site.a.loc, dist.Normal(5., 5.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(5.))

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        # v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        # c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[1], num_features[1]):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a_raw = pyro.sample(site.a.raw, dist.Normal(0, 1))
                    a = pyro.deterministic(site.a, a_loc + a_raw * a_scale)

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

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_data, num_data):
                mu = pyro.deterministic(
                    site.mu,
                    SF.rectified_logistic(
                        intensity,
                        a[feature0, feature1],
                        b[feature0, feature1],
                        g[feature0, feature1],
                        h[feature0, feature1],
                        # v[feature0, feature1],
                        h[feature0, feature1],
                        EPS,
                    )   
                )
                loc = jnp.log(mu)
                scale = c1[feature0, feature1]

                # Observation
                pyro.sample(
                    "obs",
                    dist.Normal(loc=loc, scale=scale),
                    obs=jnp.log(response) if response is not None else None
                )
