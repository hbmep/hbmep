import os
import shutil
import pickle
from operator import attrgetter

import pandas as pd
import numpy as np
from numpyro.infer import MCMC
from joblib import Parallel, delayed

from hbmep.config import Config
from hbmep.model import BaseModel
from hbmep.model.utils import abstractvariables
from hbmep.utils import timing


@abstractvariables(
    ("n_jobs", "Number of parallel jobs not specified.")
)
class NonHierarchicalBaseModel(BaseModel):
    NAME = "non_hierarchical_base_model"

    def __init__(self, config: Config):
        super(NonHierarchicalBaseModel, self).__init__(config=config)

    @timing
    def run(
        self,
        df: pd.DataFrame,
        mcmc: MCMC = None,
        extra_fields: list | tuple = [],
        **kwargs
    ):
        response_ = self.response
        combinations = self._get_combinations(df, self.features)
        temp_dir = os.path.join(self.build_dir, "optimize_results")


        def body_run_inference(combination, response, destination_path):
            ind = df[self.features].apply(tuple, axis=1).isin([combination])
            df_ = df[ind].reset_index(drop=True).copy()
            df_, _ = self.load(df=df_)

            self.response = [response]
            self.n_response = len(self.response)

            mcmc, posterior_samples = BaseModel.run(
                self, df_, **kwargs
            )
            dest = os.path.join(temp_dir, destination_path)
            with open(dest, "wb") as f: pickle.dump((mcmc, posterior_samples,), f)

            ind, df_, _, posterior_samples = None, None, None, None
            del ind, df_, _, posterior_samples
            return


        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=False)

        with Parallel(n_jobs=self.n_jobs) as parallel:
            parallel(
                delayed(body_run_inference)(
                    combination,
                    response,
                    os.path.join(temp_dir, f"{'_'.join(map(str, combination))}_{response}.pkl")
                )
                for combination in combinations
                for response in response_
            )

        self.response = response_    # Not sure if this is necessary to reset
        self.n_response = len(self.response)
        n_features = (
            df[self.features].max().astype(int).to_numpy() + 1
        )
        n_features = n_features.tolist()

        posterior_samples = None
        for combination in combinations:
            for response_ind, response in enumerate(self.response):
                src = os.path.join(
                    temp_dir,
                    f"{'_'.join(map(str, combination))}_{response}.pkl"
                )
                with open(src, "rb") as f: mcmc, samples, = pickle.load(f)

                if posterior_samples is None:
                    sample_sites = list(attrgetter(mcmc._sample_field)(mcmc._last_state).keys())
                    sample_sites_shapes = [samples[u].shape for u in sample_sites]
                    deterministic_sites = [
                        u for u in samples.keys()
                        if (u not in sample_sites) and (samples[u].shape in sample_sites_shapes)
                    ]
                    self.sample_sites = sample_sites; self.deterministic_sites = deterministic_sites
                    num_samples = samples[sample_sites[0]].shape[0]

                    posterior_samples = {
                        u: np.full(
                            (num_samples, *n_features, self.n_response), np.nan
                        )
                        for u in samples.keys()
                        if samples[u].shape in [samples[v].shape for v in self.sample_sites]
                    }

                for named_param in posterior_samples.keys():
                    posterior_samples[named_param][:, *combination, response_ind] = (
                        samples[named_param].reshape(-1,)
                    )

        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        return None, posterior_samples
