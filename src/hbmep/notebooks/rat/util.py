import os
import pickle
import logging 

from hbmep.util import Site as site
from hbmep.notebooks.constants import INFERENCE_FILE, MODEL_FILE

logger = logging.getLogger(__name__)


def run(data, model):
    # Run
    df, encoder = model.load(df=data)
    logger.info(f"df.shape {df.shape}")
    _, posterior = model.run(df=df)

    # Predictions
    prediction_df = model.make_prediction_dataset(df=df)
    if site.outlier_prob in posterior.keys():
        posterior[site.outlier_prob] = 0 * posterior[site.outlier_prob]
    predictive = model.predict(prediction_df, posterior=posterior)
    model.plot_curves(
        df=df,
        encoder=encoder,
        posterior=posterior,
        prediction_df=prediction_df,
        predictive=predictive,
    )

    # Save
    output_path = os.path.join(model.build_dir, INFERENCE_FILE)
    with open(output_path, "wb") as f:
        pickle.dump((df, encoder, posterior, model.__dict__), f)

    output_path = os.path.join(model.build_dir, MODEL_FILE)
    with open(output_path, "wb") as f:
        pickle.dump((model,), f)

    logger.info(f"Finished running {model.name}")
    logger.info(f"Saved results to {model.build_dir}")
    return


def get_subname(model):
    return (
        f'{model.mcmc_params["num_warmup"]}W'
        f'_{model.mcmc_params["num_samples"]}S'
        f'_{model.mcmc_params["num_chains"]}C'
        f'_{model.mcmc_params["thinning"]}T'
        f'_{model.nuts_params["max_tree_depth"][0]}D'
        f'_{model.nuts_params["target_accept_prob"] * 100:.0f}A'
        f'_mixture{"True" if model.use_mixture else "False"}'
    )
