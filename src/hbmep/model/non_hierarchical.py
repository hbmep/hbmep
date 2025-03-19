import os
import gc
import shutil
import pickle
import logging

import pandas as pd
import numpy as np
from numpyro.infer import MCMC
from joblib import Parallel, delayed

import hbmep as mep
from hbmep.model import BaseModel
from hbmep.util import timing

logger = logging.getLogger(__name__)


class NonHierarchicalBaseModel(BaseModel):
    def __init__(self, *args, n_jobs=-1, **kw):
        super(NonHierarchicalBaseModel, self).__init__(*args, **kw)
        self.name = "non_hierarchical_base_model"
        self.n_jobs = n_jobs
        self.pre_dispatch="1*n_jobs"
        self.prefer = "threads"

    @property
    def joblib_params(self):
        attributes = ["n_jobs", "pre_dispatch"]
        return {attr: getattr(self, attr) for attr in attributes}

    @staticmethod
    def _get_output_path(folder, combination, response):
        return os.path.join(
            folder,
            f"{response}__{'_'.join(map(str, combination))}.pkl"
        )

    def _combine_samples(self, df_features, combinations, temp_folder):
        mcmc = None
        max_features = df_features.apply(pd.Series).max().values + 1
        combined_samples = None

        for combination_idx, combination in enumerate(combinations):
            for response_idx, response in enumerate(self.response):
                src = os.path.join(temp_folder, f"{response_idx}__{combination_idx}.pkl")
                with open(src, "rb") as f: samples, = pickle.load(f)

                if not self.sample_sites:
                    if not (combination_idx or response_idx):
                        src = os.path.join(temp_folder, "mcmc.pkl")
                        with open(src, "rb") as f:
                            mcmc, = pickle.load(f)
                        self._update_sites(mcmc, samples)

                if combined_samples is None:
                    combined_samples = {
                        u: np.full((v.shape[0], *max_features, self.num_response), np.nan)
                        if u in self.sample_sites + self.reparam_sites
                        else np.full((v.shape[0], df_features.shape[0], self.num_response), np.nan)
                        for u, v in samples.items()
                    }

                for u in combined_samples.keys():
                    if u in self.sample_sites + self.reparam_sites:
                        combined_samples[u][..., *combination, response_idx] = samples[u]
                    else:
                        idx = df_features.isin([combination])
                        combined_samples[u][:, idx, response_idx] = samples[u]

        return mcmc, combined_samples

    @timing
    def run(
        self,
        df: pd.DataFrame,
        mcmc: MCMC = None,
        extra_fields: list | tuple = (),
        init_params = None,
        **kw
):
        df_features = df[self.features].apply(tuple, axis=1)
        combinations = df_features.unique().tolist()
        num_combinations = len(combinations)
        temp_folder = os.path.join(
            self.build_dir, f"hbmep_temp_folder_run"
        )


        def body_run(combination_idx, response_idx):
            ccdf = (
                df[df_features.isin([combinations[combination_idx]])]
                .reset_index(drop=True)
                .copy()
            )
            mcmc, posterior = mep.run(
                self.key,
                self._model,
                *mep.get_regressors(ccdf, self.intensity, []),
                *mep.get_response(ccdf, self.response[response_idx]),
                nuts_params=self.nuts_params,
                mcmc_params=self.mcmc_params,
                **kw
            )
            output_path = os.path.join(
                temp_folder, f"{response_idx}__{combination_idx}.pkl"
            )
            with open(output_path, "wb") as f: 
                pickle.dump((posterior,), f)
            if not (combination_idx or response_idx):
                output_path = os.path.join(temp_folder, "mcmc.pkl")
                with open(output_path, "wb") as f:
                    pickle.dump((mcmc,), f)
            ccdf, mcmc, posterior, output_path = None, None, None, None
            del ccdf, mcmc, posterior, output_path
            gc.collect()


        try:
            if os.path.exists(temp_folder): shutil.rmtree(temp_folder)
            os.makedirs(temp_folder, exist_ok=False)
            logger.info(f"Created temporary folder {temp_folder}")
            with Parallel(**self.joblib_params) as parallel:
                parallel(
                    delayed(body_run)(combination_idx, response_idx)
                    for combination_idx in range(num_combinations)
                    for response_idx in range(self.num_response)
                )

        except Exception as e:
            logger.info(f"Exception {e} occured in run")
            raise e

        else:
            mcmc, posterior = self._combine_samples(df_features, combinations, temp_folder)
            return mcmc, posterior

        finally:
            if os.path.exists(temp_folder): shutil.rmtree(temp_folder)
            logger.info(f"Removed temporary folder {temp_folder}")

    @timing
    def predict(self, df: pd.DataFrame, posterior: dict | None = None, **kw):
        df_features = df[self.features].apply(tuple, axis=1)
        combinations = df_features.unique().tolist()
        num_combinations = len(combinations)
        temp_folder = os.path.join(
            self.build_dir, f"hbmep_temp_folder_predict"
        )


        def body_predict(combination_idx, response_idx):
            ccdf = (
                df[df_features.isin([combinations[combination_idx]])]
                .reset_index(drop=True)
                .copy()
            )
            predictive = mep.predict(
                self.key,
                self._model,
                *mep.get_regressors(ccdf, self.intensity, []),
                posterior={
                    u: v[..., *combinations[combination_idx], response_idx]
                    for u, v in posterior.items() if u in self.sample_sites
                },
                **kw
            )
            output_path = os.path.join(
                temp_folder, f"{response_idx}__{combination_idx}.pkl"
            )
            with open(output_path, "wb") as f:
                pickle.dump((predictive,), f)
            ccdf, predictive, output_path = None, None, None
            del ccdf, predictive, output_path
            gc.collect()


        try:
            n_jobs = self.n_jobs
            self.n_jobs = 8
            if os.path.exists(temp_folder): shutil.rmtree(temp_folder)
            os.makedirs(temp_folder, exist_ok=False)
            logger.info(f"Created temporary folder {temp_folder}")
            with Parallel(**self.joblib_params) as parallel:
                parallel(
                    delayed(body_predict)(combination_idx, response_idx)
                    for combination_idx in range(num_combinations)
                    for response_idx in range(self.num_response)
                )

        except Exception as e:
            logger.info(f"Exception {e} occured in predict")
            raise e

        else:
            _, predictive = self._combine_samples(df_features, combinations, temp_folder)
            return predictive

        finally:
            self.n_jobs = n_jobs
            if os.path.exists(temp_folder): shutil.rmtree(temp_folder)
            logger.info(f"Removed temporary folder {temp_folder}")
