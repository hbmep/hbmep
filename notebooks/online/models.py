import logging

import numpy as np
import numpyro
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
from hbmep.util import Site as site, timing

logger = logging.getLogger(__name__)
EPS = 1e-3


class HB(BaseModel):
    def __init__(self, *args, **kw):
        super(HB, self).__init__(*args, **kw)
        self.name = "hb"

    def _model(self, intensity, features, response_obs=None, **kwargs):
        num_data = intensity.shape[0]

        with numpyro.plate(site.num_response, self.num_response):
            # Priors
            a = numpyro.sample(site.a, dist.TruncatedNormal(50., 50., low=0))
            b = numpyro.sample(site.b, dist.HalfNormal(1.))

            L = numpyro.sample(site.L, dist.HalfNormal(.1))
            ell = numpyro.sample(site.ell, dist.HalfNormal(1.))
            H = numpyro.sample(site.H, dist.HalfNormal(5.))

            c1 = numpyro.sample(site.c1, dist.HalfNormal(5.))
            c2 = numpyro.sample(site.c2, dist.HalfNormal(.5))

        with numpyro.plate(site.num_response, self.num_response):
            with numpyro.plate(site.num_data, num_data):
                # Model
                mu = numpyro.deterministic(
                    site.mu,
                    SF.rectified_logistic(intensity, a, b, L, ell, H, eps=EPS)
                )
                beta = numpyro.deterministic(
                    site.beta, self.gamma_rate(mu, c1, c2)
                )
                alpha = numpyro.deterministic(
                    site.alpha, self.gamma_concentration(mu, beta)
                )

                # Observation
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=alpha, rate=beta),
                    obs=response_obs
                )

    @timing
    def run_svi(self, df, lr=1e-2, steps=2000, PROGRESS_BAR=False):
        optimizer = optim.ClippedAdam(step_size=lr)
        _guide = autoguide.AutoLowRankMultivariateNormal(self._model)
        svi = SVI(
            self._model,
            _guide,
            optimizer,
            loss=Trace_ELBO(num_particles=20)
        )
        logger.info(f"Running {self.name}...")
        svi_result = svi.run(
            self.key,
            steps,
            *self._get_regressors(df=df),
            *self._get_response(df=df),
            progress_bar=PROGRESS_BAR
        )
        predictive = Predictive(
            _guide,
            params=svi_result.params,
            num_samples=4000
        )
        posterior_samples = predictive(self.key, *self._get_regressors(df=df))
        posterior_samples = {u: np.array(v) for u, v in posterior_samples.items()}
        return svi_result, posterior_samples


class Simulator(BaseModel):
    def __init__(self, *args, **kw):
        super(Simulator, self).__init__(*args, **kw)
        self.name = "simulator"

    def _model(self, intensity, features, response_obs=None):
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

        c1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
        c2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(.5))

        with numpyro.plate(site.num_response, self.num_response):
            with numpyro.plate(site.num_features[0], num_features[0]):
                # Priors
                a = numpyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )
                b = numpyro.sample(site.b, dist.HalfNormal(b_scale))

                L = numpyro.sample(site.L, dist.HalfNormal(L_scale))
                ell = numpyro.sample(site.ell, dist.HalfNormal(ell_scale))
                H = numpyro.sample(site.H, dist.HalfNormal(H_scale))

                c1 = numpyro.sample(site.c1, dist.HalfNormal(c1_scale))
                c2 = numpyro.sample(site.c2, dist.HalfNormal(c2_scale))

        with numpyro.plate(site.num_response, self.num_response):
            with numpyro.plate(site.num_data, num_data):
                # Model
                mu = numpyro.deterministic(
                    site.mu,
                    F.rectified_logistic(
                        x=intensity,
                        a=a[feature0],
                        b=b[feature0],
                        L=L[feature0],
                        ell=ell[feature0],
                        H=H[feature0],
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

                # Observation
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=alpha, rate=beta),
                    obs=response_obs
                )
