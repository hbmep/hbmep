import os
import shutil
import pickle

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from numpyro.infer.mcmc import MCMCKernel

from hbmep.config import Config
from hbmep.model import BaseModel
from hbmep.model.bounded_optimization import abstractvariables
from hbmep.utils import timing


@abstractvariables(
    ("n_jobs", "Number of parallel jobs not specified.")
)
class NonHierarchicalBaseModel(BaseModel):
    NAME = "non_hierarchical_base_model"

    def __init__(self, config: Config):
        super(NonHierarchicalBaseModel, self).__init__(config=config)

    @timing
    def run_inference(
        self,
        df: pd.DataFrame,
        sampler: MCMCKernel = None,
        **kwargs
    ):
        response = self.response
        combinations = self._get_combinations(df, self.features)
        temp_dir = os.path.join(self.build_dir, "optimize_results")

        def body_run_inference(combination, response, destination_path):
            nonlocal df
            ind = df[self.features].apply(tuple, axis=1).isin([combination])
            df = df[ind].reset_index(drop=True).copy()
            df, _ = self.load(df=df)

            self.response = [response]
            self.n_response = len(self.response)

            _, posterior_samples = BaseModel.run_inference(
                self, df, sampler, **kwargs
            )
            dest = os.path.join(temp_dir, destination_path)
            with open(dest, "wb") as f:
                pickle.dump((posterior_samples,), f)

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
                for response in self.response
            )

        self.response = response    # Not sure if this is necessary to reset
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
                with open(src, "rb") as f:
                    samples, = pickle.load(f)

                if posterior_samples is None:
                    named_param = list(samples.keys())[0]
                    num_samples = samples[named_param].shape[0]

                    posterior_samples = {
                        u: np.full(
                            (num_samples, *n_features, self.n_response), np.nan
                        )
                        for u in samples.keys()
                        if np.array([dim in [num_samples, 1] for dim in samples[u].shape]).all()
                    }

                for named_param in posterior_samples.keys():
                    posterior_samples[named_param][:, *combination, response_ind] = samples[named_param].reshape(-1,)

        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        return None, posterior_samples
