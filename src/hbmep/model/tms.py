import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from hbmep import smooth_functional as SF
from hbmep.model import BaseModel
from hbmep.util import Site as site

EPS = 1e-3


class RectifiedLogistic(BaseModel):
    def __init__(self, *args, **kwargs):
        super(RectifiedLogistic, self).__init__(*args, **kwargs)

    def _model(self, intensity, features, response_obs=None, **kwargs):
        use_mixture = kwargs.get("use_mixture", False)
        mask_obs = np.invert(np.isnan(response_obs)) if response_obs is not None else True
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]

        # Hyper Priors
        a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 50., low=0))
        a_scale = numpyro.sample("a_scale", dist.HalfNormal(50.))

        b_scale = numpyro.sample("b_scale", dist.HalfNormal(1.))
        L_scale = numpyro.sample("L_scale", dist.HalfNormal(.1))
        ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(1.))
        H_scale = numpyro.sample("H_scale", dist.HalfNormal(5.))

        c1_scale = numpyro.sample("c1_scale", dist.HalfNormal(5.))
        c2_scale = numpyro.sample("c2_scale", dist.HalfNormal(.5))

        with numpyro.plate(site.num_response, self.num_response):
            with numpyro.plate(site.num_features[0], num_features[0]):
                # Priors
                a = numpyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                b = numpyro.deterministic(site.b, b_scale * b_raw)

                L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
                L = numpyro.deterministic(site.L, L_scale * L_raw)

                ell_raw = numpyro.sample("ell_raw", dist.HalfNormal(scale=1))
                ell = numpyro.deterministic(site.ell, ell_scale * ell_raw)

                H_raw = numpyro.sample("H_raw", dist.HalfNormal(scale=1))
                H = numpyro.deterministic(site.H, H_scale * H_raw)

                c1_raw = numpyro.sample("c1_raw", dist.HalfNormal(scale=1))
                c1 = numpyro.deterministic(site.c1, c1_scale * c1_raw)

                c2_raw = numpyro.sample("c2_raw", dist.HalfNormal(scale=1))
                c2 = numpyro.deterministic(site.c2, c_2_scale * c_2_raw)

        # Outlier Distribuion
        if use_mixture: q = numpyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

        with numpyro.handlers.mask(mask=mask_obs):
            with numpyro.plate(site.num_response, self.num_response):
                with numpyro.plate(site.num_data, num_data):
                    # Model
                    mu = numpyro.deterministic(
                        site.mu,
                        SF.rectified_logistic(
                            x=intensity,
                            a=a[feature0],
                            b=b[feature0],
                            L=L[feature0],
                            ell=ell[feature0],
                            H=H[feature0],
                            eps=EPS
                        )
                    )
                    beta = numpyro.deterministic(
                        site.beta,
                        self.gamma_rate(mu, c1[feature0], c2[feature0])
                    )
                    alpha = numpyro.deterministic(
                        site.alpha,
                        self.gamma_concentration(mu, beta)
                    )

                    # Mixture
                    if use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Gamma(concentration=alpha, rate=beta),
                            dist.HalfNormal(scale=L[feature0] + H[feature0])
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    numpyro.sample(
                        site.obs,
                        (
                            Mixture if use_mixture
                            else dist.Gamma(concentration=alpha, rate=beta)
                        ),
                        obs=response_obs
                    )
