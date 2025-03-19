import os
import sys
import pickle
import logging
from collections import OrderedDict

import pandas as pd
import numpy as np
import numpyro.infer.util as infer_util
from numpyro import distributions as dist
from hbmep.util import timing, setup_logging, site

from hbmep.notebooks.rat.model import nHB
from hbmep.notebooks.rat.util import run
from models import HB

logger = logging.getLogger(__name__)
HOME = os.getenv("HOME")


def get_init_params_l4(model, df, encoder, nhb, posterior):
    model_args = (*nhb.get_regressors(df), *nhb.get_response(df))
    init_params, _, _, constraints = infer_util.initialize_model(
        model.key,
        model._model,
        model_args=model_args,
    )
    z = getattr(init_params, "z")
    if site.outlier_prob in z: z.pop(site.outlier_prob)

    contraints_dict = OrderedDict()
    for u, v in constraints.items():
        if v["type"] == "sample" and not v["is_observed"]:
            contraints_dict[u] = v["fn"].support
    print(contraints_dict)

    transform_dict = OrderedDict()
    for u, v in contraints_dict.items():
        transform_dict[u] = dist.biject_to(v)
    print(transform_dict)
    print(transform_dict.keys())

    num_chains = nhb.mcmc_params["num_chains"]
    num_samples = nhb.mcmc_params["num_samples"]
    assert posterior[site.a].shape[0] == num_chains * num_samples
    posterior.keys()

    named_params = [site.a, site.b, site.g, site.h, site.c1, site.c2]
    params = {u: posterior[u].reshape(num_chains, num_samples, *posterior[u].shape[1:]) for u in named_params}
    params = {u: np.nanmean(v, axis=1) for u, v in params.items()}
    params[site.b].shape
    params = {
        u: np.where(np.isnan(v), np.nanmean(v, axis=(-2, -1), keepdims=True), v)
        for u, v in params.items()
    }
    subset = [site.b, site.g, site.c1, site.c2]
    dims = tuple([i for i in range(params[site.a].ndim)][1:])
    for u in subset:
        params[site(u).scale] = params[u].mean(axis=dims)
    params[site.b.scale].shape
    a_loc = params[site.a].mean(axis=dims, keepdims=True)
    a_scale = params[site.a].std(axis=dims, keepdims=True)
    a_raw = (params[site.a] - a_loc) / a_scale
    params[site.a.loc] = a_loc.reshape(num_chains,)
    params[site.a.scale] = a_scale.reshape(num_chains,)
    params[site.a.raw] = a_raw
    h_log = np.log(params[site.h])
    h_log_loc = h_log.mean(axis=dims, keepdims=True)
    h_log_scale = h_log.std(axis=dims, keepdims=True)
    h_raw = (h_log - h_log_loc) / h_log_scale
    params[site.h.log.loc] = h_log_loc.reshape(num_chains,)
    params[site.h.log.scale] = h_log_scale.reshape(num_chains,)
    params[site.h.log.raw] = h_raw
    params = {u: v for u, v in params.items() if u in transform_dict.keys()}
    if site.outlier_prob in transform_dict:
        params[site.outlier_prob] = (
            posterior[site.outlier_prob]
            .reshape(num_chains, -1)
            .mean(axis=-1)
        )

    constrained_params = OrderedDict()
    for u in transform_dict.keys():
        constrained_params[u] = params[u]
    constrained_params.keys()

    assert z.keys() <= constrained_params.keys()
    for u, v in constrained_params.items():
        print(u, v.shape)
        if u == site.outlier_prob:
            assert v.shape == (num_chains,)
            continue
        assert not np.isnan(v).any()
        assert v[0].shape == z[u].shape

    unconstrained_params = {
        u: transform_dict[u].inv(v)
        for u, v in constrained_params.items()
    }
    unconstrained_params.keys()
    return unconstrained_params


def get_init_params_rl(model, df, encoder, nhb, posterior):
    model_args = (*nhb.get_regressors(df), *nhb.get_response(df))
    init_params, _, _, constraints = infer_util.initialize_model(
        model.key,
        model._model,
        model_args=model_args,
    )
    z = getattr(init_params, "z")
    if site.outlier_prob in z: z.pop(site.outlier_prob)

    contraints_dict = OrderedDict()
    for u, v in constraints.items():
        if v["type"] == "sample" and not v["is_observed"]:
            contraints_dict[u] = v["fn"].support
    print(contraints_dict)

    transform_dict = OrderedDict()
    for u, v in contraints_dict.items():
        transform_dict[u] = dist.biject_to(v)
    print(transform_dict)
    print(transform_dict.keys())

    num_chains = nhb.mcmc_params["num_chains"]
    num_samples = nhb.mcmc_params["num_samples"]
    assert posterior[site.a].shape[0] == num_chains * num_samples
    posterior.keys()

    named_params = [site.a, site.b, site.g, site.h, site.v, site.c1, site.c2]
    params = {u: posterior[u].reshape(num_chains, num_samples, *posterior[u].shape[1:]) for u in named_params}
    params = {u: np.nanmean(v, axis=1) for u, v in params.items()}
    params[site.b].shape
    params = {
        u: np.where(np.isnan(v), np.nanmean(v, axis=(-2, -1), keepdims=True), v)
        for u, v in params.items()
    }
    subset = [site.b, site.g, site.h, site.v, site.c1, site.c2]
    dims = tuple([i for i in range(params[site.a].ndim)][1:])
    for u in subset:
        params[site(u).scale] = params[u].mean(axis=dims)
    params[site.b.scale].shape
    a_loc = params[site.a].mean(axis=dims, keepdims=True)
    a_scale = params[site.a].std(axis=dims, keepdims=True)
    a_raw = (params[site.a] - a_loc) / a_scale
    params[site.a.loc] = a_loc.reshape(num_chains,)
    params[site.a.scale] = a_scale.reshape(num_chains,)
    params[site.a.raw] = a_raw
    params = {u: v for u, v in params.items() if u in transform_dict.keys()}
    if site.outlier_prob in transform_dict:
        params[site.outlier_prob] = (
            posterior[site.outlier_prob]
            .reshape(num_chains, -1)
            .mean(axis=-1)
        )

    constrained_params = OrderedDict()
    for u in transform_dict.keys():
        constrained_params[u] = params[u]
    constrained_params.keys()

    assert z.keys() <= constrained_params.keys()
    for u, v in constrained_params.items():
        print(u, v.shape)
        if u == site.outlier_prob:
            assert v.shape == (num_chains,)
            continue
        assert not np.isnan(v).any()
        assert v[0].shape == z[u].shape

    unconstrained_params = {
        u: transform_dict[u].inv(v)
        for u, v in constrained_params.items()
    }
    unconstrained_params.keys()
    return unconstrained_params


@timing
def main(model, data_path):
    # Load data
    data = pd.read_csv(data_path)
    idx = data[model.intensity] > 0
    data = data[idx].reset_index(drop=True).copy()
    data[model.intensity] = np.log2(data[model.intensity])

    # idx = data[model.features[0]].isin(["amap01"])
    # data = data[idx].reset_index(drop=True).copy()
    # model.response = model.response[:1]

    extra_fields = ["num_steps"]
    run(data, model, extra_fields=extra_fields)
    return


if __name__ == "__main__":
    experiment = "lcirc"
    match experiment:
        case "lcirc": exp = "L_CIRC"
        case "lshie": exp = "L_SHIE"
        case "csmalar": exp = "C_SMA_LAR"
        case _: raise ValueError("Invalid experiment")

    toml_path = f"{HOME}/repos/refactor/hbmep/configs/rat/{exp}.toml"
    model = HB(toml_path=toml_path)

    build_dir = f"{HOME}/reports/hbmep/notebooks/rat/lognhb/{model.name}/{experiment}/"
    model.build_dir = os.path.join(build_dir, model._model.__name__)
    setup_logging(model.build_dir)

    data_path = f"{HOME}/data/hbmep-processed/rat/{exp}/data.csv"
    main(model, data_path)
