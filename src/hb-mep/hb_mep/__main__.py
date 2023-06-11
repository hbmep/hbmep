import sys
import logging
import argparse

from hb_mep.config import HBMepConfig
from hb_mep.models import Baseline
from hb_mep.models.rats import RectifiedLogistic, GammaRegression
# from hb_mep.experiments import Experiment, SparseDataExperiment
# from hb_mep.experiments.models import BayesianHierarchical
from hb_mep.api import run_inference, run_experiment, run_inference_rats

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


def main(args):
    config = HBMepConfig()
    models = [Baseline, RectifiedLogistic, GammaRegression]

    try:
        model = args.model
        assert model in [m(config).name for m in models]

        Model = [m for m in models if m(config).name == model][0]

    except AssertionError:
        raise AssertionError(f"Invalid model {model} for inference job")

    run_inference_rats(config, Model)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run HB-MEP"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model to run"
    )

    try:
        args = parser.parse_args(sys.argv[2:])
    except IndexError:
        IndexError("Call to API requires an endpoint")

    main(args)
