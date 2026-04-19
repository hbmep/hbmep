import logging

import numpy as np
import jax
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist

from hbmep import functional as F
from hbmep.model import BaseModel
from hbmep.util import site

logger = logging.getLogger(__name__)
EPS = 1e-3


class StandardHB(BaseModel):
    """
    Standard hierarchical Bayesian model.

    Features and responses are modeled as conditionally independent given
    their model parameters. In particular, each feature and response combination
    has its own set of curve and likelihood parameters, which are partially
    pooled using shared hyperpriors.

    This class implements models with three curve functions: rectified-logistic,
    logistic5, and logistic4. Observations are modeled using a gamma likelihood.

    Notes
    -----
    - The curve function can be selected as:
      `model._model = model.rectified_logistic` (default),
      `model._model = model.logistic5`, or
      `model._model = model.logistic4`.

    - The priors are specified for TMS data with
      intensity in 0-100% MSO and response in millivolts (mV). To change the
      prior specification, inherit from this class and override the
      `_sample_priors` method.

    - Setting `self.use_mixture = True` uses a mixture likelihood with an
      additional outlier component.
    """
    def __init__(self, *args, **kw):
        super(StandardHB, self).__init__(*args, **kw)
        self.use_mixture = False

    def _model(self, *args, **kw):
        return self.rectified_logistic(*args, **kw)

    def _sample_priors(
        self,
        *,
        num_features,
        include_v: bool,
    ):
        # Hyper-priors
        a_loc = pyro.sample(
            site.a.loc, dist.TruncatedNormal(50., 50., low=0)
        )
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(50.))
        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))

        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))

        if include_v:
            v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        params = {}
        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate_stack(
                site.num_features, num_features, rightmost_dim=-2
            ):
                params[site.a] = pyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                params[site.b] = pyro.deterministic(site.b, b_scale * b_raw)

                g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                params[site.g] = pyro.deterministic(site.g, g_scale * g_raw)

                h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                params[site.h] = pyro.deterministic(site.h, h_scale * h_raw)

                if include_v:
                    v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                    params[site.v] = pyro.deterministic(site.v, v_scale * v_raw)

                c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                params[site.c1] = pyro.deterministic(site.c1, c1_scale * c1_raw)

                c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                params[site.c2] = pyro.deterministic(site.c2, c2_scale * c2_raw)

        return params

    def _observe(
        self,
        *,
        mu,
        response,
        mask_obs,
        num_data,
        params,
        features,
    ):
        c1 = params[site.c1]
        c2 = params[site.c2]
        g = params.get(site.g)
        h = params.get(site.h)

        if self.use_mixture:
            if response is None:
                q = 0.
            else:
                q = pyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu = pyro.deterministic(site.mu, mu)

                    alpha, beta = self.gamma_likelihood(
                        mu, c1[*features.T], c2[*features.T]
                    )

                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions = [
                            dist.Gamma(concentration=alpha, rate=beta),
                            dist.HalfNormal(
                                scale=(g[*features.T] + h[*features.T])
                            ),
                        ]
                        dist_ = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions,
                        )
                    else:
                        dist_ = dist.Gamma(concentration=alpha, rate=beta)

                    y_ = pyro.sample(site.obs, dist_, obs=response)

                    if self.use_mixture:
                        log_probs = dist_.component_log_probs(y_)
                        pyro.deterministic(
                            "p",
                            log_probs - jax.nn.logsumexp(
                                log_probs, axis=-1, keepdims=True
                            ),
                        )

    def rectified_logistic(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        params = self._sample_priors(
            num_features=num_features,
            include_v=True,
        )

        mu = F.rectified_logistic(
            intensity,
            params[site.a][*features.T],
            params[site.b][*features.T],
            params[site.g][*features.T],
            params[site.h][*features.T],
            params[site.v][*features.T],
            EPS,
        )

        self._observe(
            mu=mu,
            response=response,
            mask_obs=mask_obs,
            num_data=num_data,
            params=params,
            features=features,
        )

    def logistic5(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        params = self._sample_priors(
            num_features=num_features,
            include_v=True,
        )

        mu = F.logistic5(
            intensity,
            params[site.a][*features.T],
            params[site.b][*features.T],
            params[site.g][*features.T],
            params[site.h][*features.T],
            params[site.v][*features.T],
        )

        self._observe(
            mu=mu,
            response=response,
            mask_obs=mask_obs,
            num_data=num_data,
            params=params,
            features=features,
        )

    def logistic4(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        params = self._sample_priors(
            num_features=num_features,
            include_v=False,
        )

        mu = F.logistic4(
            intensity,
            params[site.a][*features.T],
            params[site.b][*features.T],
            params[site.g][*features.T],
            params[site.h][*features.T],
        )

        self._observe(
            mu=mu,
            response=response,
            mask_obs=mask_obs,
            num_data=num_data,
            params=params,
            features=features,
        )
