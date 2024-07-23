import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep.nn import functional as F
from hbmep.model import GammaModel
from hbmep.model.utils import Site as site


class RectifiedLogistic(GammaModel):
    NAME = "rectified_logistic"

    def __init__(self, config: Config):
        super(RectifiedLogistic, self).__init__(config=config)

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]

        # Hyper Priors
        a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 50., low=0))
        a_scale = numpyro.sample("a_scale", dist.HalfNormal(50.))

        b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))

        L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))
        ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(10.))
        H_scale = numpyro.sample("H_scale", dist.HalfNormal(5.))

        c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
        c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5.))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[0], n_features[0]):
                # Priors
                a = numpyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
                L = numpyro.deterministic(site.L, jnp.multiply(L_scale, L_raw))

                ell_raw = numpyro.sample("ell_raw", dist.HalfNormal(scale=1))
                ell = numpyro.deterministic(site.ell, jnp.multiply(ell_scale, ell_raw))

                H_raw = numpyro.sample("H_raw", dist.HalfNormal(scale=1))
                H = numpyro.deterministic(site.H, jnp.multiply(H_scale, H_raw))

                c_1_raw = numpyro.sample("c_1_raw", dist.HalfNormal(scale=1))
                c_1 = numpyro.deterministic(site.c_1, jnp.multiply(c_1_scale, c_1_raw))

                c_2_raw = numpyro.sample("c_2_raw", dist.HalfNormal(scale=1))
                c_2 = numpyro.deterministic(site.c_2, jnp.multiply(c_2_scale, c_2_raw))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                # Model
                mu = numpyro.deterministic(
                    site.mu,
                    F.rectified_logistic(
                        x=intensity,
                        a=a[feature0],
                        b=b[feature0],
                        L=L[feature0],
                        ell=ell[feature0],
                        H=H[feature0]
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    self.rate(
                        mu,
                        c_1[feature0],
                        c_2[feature0]
                    )
                )
                alpha = numpyro.deterministic(
                    site.alpha,
                    self.concentration(mu, beta)
                )

                # Observation
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=alpha, rate=beta),
                    obs=response_obs
                )


class Logistic5(GammaModel):
    NAME = "logistic5"

    def __init__(self, config: Config):
        super(Logistic5, self).__init__(config=config)

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]

        # Hyper Priors
        a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 50., low=0))
        a_scale = numpyro.sample("a_scale", dist.HalfNormal(50.))

        b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))
        v_scale = numpyro.sample("v_scale", dist.HalfNormal(5.))

        L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))
        H_scale = numpyro.sample("H_scale", dist.HalfNormal(5.))

        c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
        c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5.))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[0], n_features[0]):
                # Priors
                a = numpyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                v_raw = numpyro.sample("v_raw", dist.HalfNormal(scale=1))
                v = numpyro.deterministic(site.v, jnp.multiply(v_scale, v_raw))

                L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
                L = numpyro.deterministic(site.L, jnp.multiply(L_scale, L_raw))

                H_raw = numpyro.sample("H_raw", dist.HalfNormal(scale=1))
                H = numpyro.deterministic(site.H, jnp.multiply(H_scale, H_raw))

                c_1_raw = numpyro.sample("c_1_raw", dist.HalfNormal(scale=1))
                c_1 = numpyro.deterministic(site.c_1, jnp.multiply(c_1_scale, c_1_raw))

                c_2_raw = numpyro.sample("c_2_raw", dist.HalfNormal(scale=1))
                c_2 = numpyro.deterministic(site.c_2, jnp.multiply(c_2_scale, c_2_raw))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                # Model
                mu = numpyro.deterministic(
                    site.mu,
                    F.logistic5(
                        x=intensity,
                        a=a[feature0],
                        b=b[feature0],
                        v=v[feature0],
                        L=L[feature0],
                        H=H[feature0]
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    self.rate(
                        mu,
                        c_1[feature0],
                        c_2[feature0]
                    )
                )
                alpha = numpyro.deterministic(
                    site.alpha,
                    self.concentration(mu, beta)
                )

                # Observation
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=alpha, rate=beta),
                    obs=response_obs
                )


class Logistic4(GammaModel):
    NAME = "logistic4"

    def __init__(self, config: Config):
        super(Logistic4, self).__init__(config=config)

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]

        # Hyper Priors
        a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 50., low=0))
        a_scale = numpyro.sample("a_scale", dist.HalfNormal(50.))

        b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))
        L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))
        H_scale = numpyro.sample("H_scale", dist.HalfNormal(5.))

        c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
        c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5.))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[0], n_features[0]):
                # Priors
                a = numpyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
                L = numpyro.deterministic(site.L, jnp.multiply(L_scale, L_raw))

                H_raw = numpyro.sample("H_raw", dist.HalfNormal(scale=1))
                H = numpyro.deterministic(site.H, jnp.multiply(H_scale, H_raw))

                c_1_raw = numpyro.sample("c_1_raw", dist.HalfNormal(scale=1))
                c_1 = numpyro.deterministic(site.c_1, jnp.multiply(c_1_scale, c_1_raw))

                c_2_raw = numpyro.sample("c_2_raw", dist.HalfNormal(scale=1))
                c_2 = numpyro.deterministic(site.c_2, jnp.multiply(c_2_scale, c_2_raw))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                # Model
                mu = numpyro.deterministic(
                    site.mu,
                    F.logistic4(
                        x=intensity,
                        a=a[feature0],
                        b=b[feature0],
                        L=L[feature0],
                        H=H[feature0]
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    self.rate(
                        mu,
                        c_1[feature0],
                        c_2[feature0]
                    )
                )
                alpha = numpyro.deterministic(
                    site.alpha,
                    self.concentration(mu, beta)
                )

                # Observation
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=alpha, rate=beta),
                    obs=response_obs
                )


class RectifiedLinear(GammaModel):
    NAME = "rectified_linear"

    def __init__(self, config: Config):
        super(RectifiedLinear, self).__init__(config=config)

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]

        # Hyper Priors
        a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 50., low=0))
        a_scale = numpyro.sample("a_scale", dist.HalfNormal(50.))

        b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))
        L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))

        c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
        c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5.))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[0], n_features[0]):
                # Priors
                a = numpyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
                L = numpyro.deterministic(site.L, jnp.multiply(L_scale, L_raw))

                c_1_raw = numpyro.sample("c_1_raw", dist.HalfNormal(scale=1))
                c_1 = numpyro.deterministic(site.c_1, jnp.multiply(c_1_scale, c_1_raw))

                c_2_raw = numpyro.sample("c_2_raw", dist.HalfNormal(scale=1))
                c_2 = numpyro.deterministic(site.c_2, jnp.multiply(c_2_scale, c_2_raw))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                # Model
                mu = numpyro.deterministic(
                    site.mu,
                    F.rectified_linear(
                        x=intensity,
                        a=a[feature0],
                        b=b[feature0],
                        L=L[feature0]
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    self.rate(
                        mu,
                        c_1[feature0],
                        c_2[feature0]
                    )
                )
                alpha = numpyro.deterministic(
                    site.alpha,
                    self.concentration(mu, beta)
                )

                # Observation
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=alpha, rate=beta),
                    obs=response_obs
                )


class MixtureModel(GammaModel):
    NAME = "mixture_model"

    def __init__(self, config: Config):
        super(MixtureModel, self).__init__(config=config)

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]

        # Hyper Priors
        a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 50., low=0))
        a_scale = numpyro.sample("a_scale", dist.HalfNormal(50.))

        b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))

        L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))
        ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(10.))
        H_scale = numpyro.sample("H_scale", dist.HalfNormal(5.))

        c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
        c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5.))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[0], n_features[0]):
                # Priors
                a = numpyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
                L = numpyro.deterministic(site.L, jnp.multiply(L_scale, L_raw))

                ell_raw = numpyro.sample("ell_raw", dist.HalfNormal(scale=1))
                ell = numpyro.deterministic(site.ell, jnp.multiply(ell_scale, ell_raw))

                H_raw = numpyro.sample("H_raw", dist.HalfNormal(scale=1))
                H = numpyro.deterministic(site.H, jnp.multiply(H_scale, H_raw))

                c_1_raw = numpyro.sample("c_1_raw", dist.HalfNormal(scale=1))
                c_1 = numpyro.deterministic(site.c_1, jnp.multiply(c_1_scale, c_1_raw))

                c_2_raw = numpyro.sample("c_2_raw", dist.HalfNormal(scale=1))
                c_2 = numpyro.deterministic(site.c_2, jnp.multiply(c_2_scale, c_2_raw))

        # Outlier Distribution
        outlier_prob = numpyro.sample(site.outlier_prob, dist.Uniform(0., .01))
        outlier_scale = numpyro.sample(site.outlier_scale, dist.HalfNormal(10))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                # Model
                mu = numpyro.deterministic(
                    site.mu,
                    F.rectified_logistic(
                        x=intensity,
                        a=a[feature0],
                        b=b[feature0],
                        L=L[feature0],
                        ell=ell[feature0],
                        H=H[feature0]
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    self.rate(
                        mu,
                        c_1[feature0],
                        c_2[feature0]
                    )
                )
                alpha = numpyro.deterministic(
                    site.alpha,
                    self.concentration(mu, beta)
                )

                # Mixture
                q = numpyro.deterministic(
                    site.q, outlier_prob * jnp.ones((n_data, self.n_response))
                )
                bg_scale = numpyro.deterministic(
                    site.bg_scale,
                    outlier_scale * jnp.ones((n_data, self.n_response))
                )

                mixing_distribution = dist.Categorical(
                    probs=jnp.stack([1 - q, q], axis=-1)
                )
                component_distributions = [
                    dist.Gamma(concentration=alpha, rate=beta),
                    dist.HalfNormal(scale=bg_scale)
                ]

                Mixture = dist.MixtureGeneral(
                    mixing_distribution=mixing_distribution,
                    component_distributions=component_distributions
                )

                # Observation
                y_ = numpyro.sample(
                    site.obs,
                    Mixture,
                    obs=response_obs
                )

                # Thanks to https://dfm.io/posts/intro-to-numpyro/
                # Until here, where we can track the membership probability of each sample
                log_probs = Mixture.component_log_probs(y_)
                numpyro.deterministic(
                    site.p,
                    (
                        1
                        - jnp.exp(
                            log_probs
                            - jax.nn.logsumexp(log_probs, axis=-1, keepdims=True)
                        )[..., 0]
                    )
                )
