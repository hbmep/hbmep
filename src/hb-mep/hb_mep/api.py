import os
import logging

import numpyro

from hb_mep.config import HBMepConfig
from hb_mep.data_access import DataClass
from hb_mep.models.baseline import Baseline
from hb_mep.utils import timing

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)

logger = logging.getLogger(__name__)


@timing
def run_inference(
    config: HBMepConfig = HBMepConfig,
    model: Baseline = Baseline
) -> None:
    data = DataClass(config)
    data.make_dirs()

    # Load data and preprocess
    df, encoder_dict = data.build()

    # Run MCMC inference
    mcmc, posterior_samples = model.sample(df=df)

    # Plots
    fit_fig = model.plot_fit(df=df, posterior_samples=posterior_samples)
    kde_fig = model.plot_kde(df=df, posterior_samples=posterior_samples)

    # Save artefacts
    fit_fig.savefig(
        os.path.join(data.reports_path, config.PLOT_FIT),
        facecolor='w'
    )
    kde_fig.savefig(
        os.path.join(data.reports_path, config.PLOT_KDE),
        facecolor='w'
    )
    logger.info(f'Saved artefacts to {data.reports_path}')


@timing
def run_experiment(config: HBMepConfig = HBMepConfig):
    data = DataClass(config)
    data.make_dirs()

    n_participant = 20
    n_segment = 2

    num_random_seeds = 20           # 20
    num_sparse_factors = 8          # 10

    sparse_factor_space = [i/10 for i in range(num_sparse_factors)]
    random_seed_space = random.choice(
        random.PRNGKey(128), np.array(range(1000)), shape=(num_random_seeds,), replace=False
    )

    res = None
    columns = [
        "sparse_factor",
        "random_seed",
        "n_participant",
        "n_segment",
        "hb_MAE",
        "nhb_MAE",
        "mle_MAE"
    ]

    for sparse_factor in sparse_factor_space:
        for j, random_seed in enumerate(random_seed_space):
            _, a = data.simulate(
                random_seed=random_seed,
                n_participant=n_participant,
                n_segment=n_segment,
                sparse_factor=sparse_factor
            )
            df, _ = data.build()

            # HB Model
            hb = SaturatedReLU_HB(config)
            _, posterior_samples_hb = hb.sample(df=df)

            # NHB Model
            nhb = SaturatedReLU_NHB(config)
            _, posterior_samples_nhb = nhb.sample(df=df)

            # MLE Model
            mle = SaturatedReLU_MLE(config)
            _, posterior_samples_mle = mle.sample(df=df)

            hb_error = np.abs(posterior_samples_hb["a"].mean(axis=0).reshape(-1,) - a.reshape(-1,)).sum()
            nhb_error = np.abs(posterior_samples_nhb["a"].mean(axis=0).reshape(-1,) - a.reshape(-1,)).sum()
            mle_error = np.abs(posterior_samples_mle["a"].mean(axis=0).reshape(-1,) - a.reshape(-1,)).sum()
            logger.info(f" === sparse_factor: {sparse_factor} ===")
            logger.info(f" === seed: {j + 1}/{num_random_seeds} ===")
            logger.info(f" === HB: {hb_error}, NHB: {nhb_error}, MLE: {mle_error} ===")
            logger.info(f" === NHB-HB: {nhb_error - hb_error} ===")
            logger.info(f" === MLE-HB: {mle_error - hb_error} ===")
            logger.info(f" === MLE-NHB: {mle_error - nhb_error} ===")

            arr = np.array([
                sparse_factor,
                random_seed,
                n_participant,
                n_segment,
                hb_error,
                nhb_error,
                mle_error
            ]).reshape(-1, 1).T

            if res is None:
                res = pd.DataFrame(arr, columns=columns)
            else:
                temp = pd.DataFrame(arr, columns=columns)
                res = pd.concat([res, temp], ignore_index=True).copy()

            res.n_participant = res.n_participant.astype(int)
            res.n_segment = res.n_segment.astype(int)
            res.random_seed = res.random_seed.astype(int)

            save_path = os.path.join(data.reports_path, f"sparse_data_experiment.csv")
            res.to_csv(save_path, index=False)