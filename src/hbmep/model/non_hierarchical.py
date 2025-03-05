import os
import shutil
import pickle
from operator import attrgetter

import pandas as pd
import numpy as np
from numpyro.infer import MCMC
from joblib import Parallel, delayed

from hbmep.model import BaseModel
from hbmep.util import timing


class NonHierarchicalBaseModel(BaseModel):
    def __init__(self, *args, n_jobs=-1, **kw):
        super(NonHierarchicalBaseModel, self).__init__(*args, **kw)
        self.name = "non_hierarchical_base_model"
        self.n_jobs = n_jobs


    @staticmethod
    def _get_subdir(temp_dir, combination, response):
        return os.path.join(
            temp_dir,
            f"{response}__{'_'.join(map(str, combination))}.pkl"
        )

    def _combine_samples(self, df_features, temp_dir):
        combinations = df_features.unique().tolist()
        features_max = df_features.apply(pd.Series).max().values + 1
        combined_samples = None

        for combination in combinations:
            for r, response in enumerate(self.response):
                src = self._get_subdir(temp_dir, combination, response)
                try:
                    with open(src, "rb") as f:
                        mcmc, samples, = pickle.load(f)
                except ValueError:
                    try:
                        with open(src, "rb") as f:
                            samples, = pickle.load(f)
                    except Exception as e:
                        raise Exception
                except Exception as e:
                    raise Exception

                if combined_samples is None:
                    if self.sample_sites is None:
                        sample_sites = list(attrgetter(mcmc._sample_field)(mcmc._last_state).keys())
                        sample_sites_shapes = [samples[u].shape for u in sample_sites]
                        deterministic_sites = [
                            u for u in samples.keys()
                            if (u not in sample_sites) and (samples[u].shape in sample_sites_shapes)
                        ]
                        self.sample_sites = sample_sites
                        self.deterministic_sites = deterministic_sites
                    combined_samples = {
                        u: np.full((v.shape[0], *features_max, self.num_response), np.nan)
                        if v.ndim == 1
                        else np.full((v.shape[0], df_features.shape[0], self.num_response), np.nan)
                        for u, v in samples.items()
                    }

                for u, v in combined_samples.items():
                    if v.ndim == self.num_features + 2:
                        combined_samples[u][..., *combination, r] = samples[u]
                    else:
                        idx = df_features.isin([combination])
                        combined_samples[u][:, idx, r] = samples[u]

        return combined_samples

    @timing
    def run(self, df: pd.DataFrame, **kw):
        var_response = [r for r in self.response]
        var_features = [f for f in self.features]
        df_features = df[var_features].apply(tuple, axis=1)
        combinations = df_features.unique().tolist()
        temp_dir = os.path.join(self.build_dir, "temp_run_dir")


        def body_run(combination, response, output_path):
            idx = df_features.isin([combination])
            ccdf = df[idx].reset_index(drop=True).copy()
            self.features = []
            self.response = response
            mcmc, posterior = BaseModel.run(self, ccdf, **kw)
            with open(output_path, "wb") as f:
                pickle.dump((mcmc, posterior,), f)
            return


        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=False)
        with Parallel(n_jobs=self.n_jobs) as parallel:
            parallel(
                delayed(body_run)(
                    combination,
                    response,
                    self._get_subdir(temp_dir, combination, response)
                )
                for combination in combinations
                for response in var_response
            )

        self.features = var_features
        self.response = var_response
        posterior = self._combine_samples(df_features, temp_dir)
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        return None, posterior

    @timing
    def predict(self, df: pd.DataFrame, posterior: dict | None = None, **kw):
        var_response = [r for r in self.response]
        var_features = [f for f in self.features]
        df_features = df[var_features].apply(tuple, axis=1)
        combinations = df_features.unique().tolist()
        temp_dir = os.path.join(self.build_dir, "temp_predict_dir")


        def body_predict(combination, response_idx, output_path):
            idx = df_features.isin([combination])
            ccdf = df[idx].reset_index(drop=True).copy()
            self.features = []
            self.response = var_response[response_idx]
            ccposterior = {
                u: v[..., *combination, response_idx]
                for u, v in posterior.items() if u in self.sample_sites
            }
            predictive = BaseModel.predict(self, ccdf, posterior=ccposterior, **kw)
            with open(output_path, "wb") as f:
                pickle.dump((predictive,), f)
            return


        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=False)
        with Parallel(n_jobs=self.n_jobs) as parallel:
            parallel(
                delayed(body_predict)(
                    combination,
                    response_idx,
                    self._get_subdir(temp_dir, combination, var_response[response_idx])
                )
                for combination in combinations
                for response_idx in range(len(var_response))
            )


        self.features = var_features
        self.response = var_response
        predictive = self._combine_samples(df_features, temp_dir)
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        return predictive
