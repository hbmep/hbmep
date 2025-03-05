import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from hbmep import functional as F
from hbmep.model import NonHierarchicalBaseModel
from hbmep.util import Site as site

from hbmep.notebooks.rat.util import get_subname


class nHB(NonHierarchicalBaseModel):
    def __init__(self, use_mixture, *args, **kw):
        super(nHB, self).__init__(*args, **kw)
        self.name = "nHB"
        self.use_mixture = use_mixture
        self.mcmc_params = {
            "num_warmup": 4000,
            "num_samples": 1000,
            "num_chains": 4,
            "thinning": 1,
        }
        self.nuts_params = {
            "max_tree_depth": (20, 20),
            "target_accept_prob": .95,
        }
        self.name = f"{self.name}__{get_subname(self)}"

    def _model(self, intensity, features, response_obs=None, **kw):
        num_data = intensity.shape[0]

        a = numpyro.sample(site.a, dist.Normal(5., 5.))
        b = numpyro.sample(site.b, dist.HalfNormal(scale=5.))

        L = numpyro.sample(site.L, dist.HalfNormal(scale=.1))
        ell = numpyro.sample(site.ell, dist.HalfNormal(scale=1.))
        H = numpyro.sample(site.H, dist.HalfNormal(scale=5.))

        c1 = numpyro.sample(site.c1, dist.HalfNormal(scale=5.))
        c2 = numpyro.sample(site.c2, dist.HalfNormal(scale=.5))

        if self.use_mixture:
            q = numpyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

        with numpyro.plate(site.num_data, num_data):
            # Model
            mu = numpyro.deterministic(
                site.mu,
                F.rectified_logistic(intensity, a, b, L, ell, H)
            )
            beta = numpyro.deterministic(
                site.beta,
                self.gamma_rate(mu, c1, c2)
            )
            alpha = numpyro.deterministic(
                site.alpha,
                self.gamma_concentration(mu, beta)
            )

            if self.use_mixture:
                mixing_distribution = dist.Categorical(
                    probs=jnp.stack([1 - q, q], axis=-1)
                )
                component_distributions=[
                    dist.Gamma(concentration=alpha, rate=beta),
                    dist.HalfNormal(scale=(L + H))
                ]
                Mixture = dist.MixtureGeneral(
                    mixing_distribution=mixing_distribution,
                    component_distributions=component_distributions
                )

            numpyro.sample(
                site.obs,
                (
                    Mixture if self.use_mixture
                    else dist.Gamma(concentration=alpha, rate=beta)
                ),
                obs=response_obs
            )
