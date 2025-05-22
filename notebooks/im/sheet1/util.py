import os
import pickle
import logging 
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from hbmep.util import site
from hbmep.util.site import SiteAttribute

from constants import EPS, MIN_CONC, MAX_CONC

logger = logging.getLogger(__name__)


@dataclass
class Site(site):
    b1 = SiteAttribute("β₁")
    b2 = SiteAttribute("β₂")
    b3 = SiteAttribute("β₃")
    b4 = SiteAttribute("β₄")

    sigma = SiteAttribute("σ")
    alpha = SiteAttribute("α")


def load(df, log_conc=False):
    df = df.copy()
    # if range_restricted:
    #     conc = sorted(df.conc.unique().tolist())
    #     logger.info("Restricting range...")
    #     logger.info(f"Original df.shape: {df.shape}")
    #     logger.info(f"Original conc: {conc}")
    #     idx = (df.conc < 223) & (df.conc > 2)
    #     df = df[idx].reset_index(drop=True).copy()
    #     conc = sorted(df.conc.unique().tolist())
    #     logger.info(f"Restricted df.shape: {df.shape}")
    #     logger.info(f"Restricted conc: {conc}")

    if log_conc:
        idx = df.conc > 0
        df = df[idx].reset_index(drop=True).copy()
        df.conc = np.log(df.conc)

    # df.conc = df.conc.replace({0: EPS})
    idx = df.conc > 0
    df = df[idx].reset_index(drop=True).copy()
    return df


def predict(df, encoder, posterior, model, mcmc, **kw):
    # Predictions
    prediction_df = model.make_prediction_dataset(df=df)
    if site.outlier_prob in posterior.keys():
        posterior[site.outlier_prob] = 0 * posterior[site.outlier_prob]
    predictive = model.predict(prediction_df, posterior=posterior, **kw)
    # model.plot_curves(
    #     df=df,
    #     encoder=encoder,
    #     prediction_df=prediction_df,
    #     predictive=predictive,
    #     posterior=posterior,
    #     posterior_var=posterior_var
    # )
    model.plot_curves(
        df,
		prediction_df=prediction_df,
		predictive=predictive,
		encoder=encoder,
		prediction_prob=.95
    )

    if site.outlier_prob in posterior.keys():
        posterior.pop(site.outlier_prob)
    summary_df = model.summary(posterior)
    logger.info(f"Summary:\n{summary_df.to_string()}")
    dest = os.path.join(model.build_dir, "summary.csv")
    summary_df.to_csv(dest)
    logger.info(f"Saved summary to {dest}")
    logger.info(f"Finished running {model._model.__name__}")
    try:
        divergences = mcmc.get_extra_fields()["diverging"].sum().item()
        logger.info(f"No. of divergences {divergences}")
        num_steps = mcmc.get_extra_fields()["num_steps"]
        tree_depth = np.floor(np.log2(num_steps)).astype(int)
        logger.info(f"Tree depth statistics:")
        logger.info(f"Min: {tree_depth.min()}")
        logger.info(f"Max: {tree_depth.max()}")
        logger.info(f"Mean: {tree_depth.mean()}")
    except: pass
    logger.info(f"Saved results to {model.build_dir}")
    return


def run(data, model, encoder=None, **kw):
    # Run
    if encoder is None: df, encoder = model.load(df=data)
    else: df = data.copy()
    logger.info(f"df.shape {df.shape}")
    model.plot(df)
    mcmc, posterior = model.run(df=df, **kw)

    # Save
    output_path = os.path.join(model.build_dir, "inf.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((df, encoder, posterior,), f)
    logger.info(f"Saved to {output_path}")

    output_path = os.path.join(model.build_dir, "model.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((model,), f)
    logger.info(f"Saved to {output_path}")

    output_path = os.path.join(model.build_dir, "model_dict.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((model.__dict__,), f)
    logger.info(f"Saved to {output_path}")

    if mcmc is not None:
        output_path = os.path.join(model.build_dir, "mcmc.pkl")
        with open(output_path, "wb") as f:
            pickle.dump((mcmc,), f)
        logger.info(f"Saved to {output_path}")

    predict(df, encoder, posterior, model, mcmc, **kw)
    return


def bs(x, knots, degree=3, include_intercept=False):
    """
    Mimic R's bs() function for B-spline basis creation.
    
    Args:
        x: Input values (1D array)
        knots: Interior knot positions (must be sorted)
        degree: Degree of spline (3 = cubic)
        include_intercept: Whether to include first basis function (False mimics R's bs)
        
    Returns:
        Basis matrix of shape (len(x), num_basis)
    """
    # Pad knots at boundaries
    padded_knots = np.hstack(
        [[x.min()] * (degree + 1), knots, [x.max()] * (degree + 1)]
    )
    
    num_basis = len(padded_knots) - degree - 1
    eye = np.eye(num_basis)
    func = BSpline(padded_knots, eye, degree)
    out = func(x)
    return out


def load_model(model_dir):
    src = os.path.join(model_dir, "inf.pkl")
    with open(src, "rb") as f:
        df, encoder, posterior = pickle.load(f)
    src = os.path.join(model_dir, "model.pkl")
    with open(src, "rb") as f:
       model, = pickle.load(f)
    return (
        df,
		encoder,
		model,
		posterior,
    )


def make_serial(
	dilution_factor,
	min_conc=MIN_CONC,
    max_conc=MAX_CONC,
):
    x = [max_conc]
    for _ in range(100):
        curr_conc = x[-1] / dilution_factor
        if curr_conc < min_conc: break
        else: x.append(curr_conc)
    return sorted(np.array(x))
