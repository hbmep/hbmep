from pyexpat import model
import sys
import logging
import argparse

from hb_mep.config import HBMepConfig
from hb_mep.models import Baseline
from hb_mep.models.human import RectifiedLogistic
from hb_mep.experiments import Experiment, SparseDataExperiment
from hb_mep.experiments.models import BayesianHierarchical
from hb_mep.api import run_inference, run_experiment

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


def main(args):
    config = HBMepConfig()
    models = [Baseline, RectifiedLogistic]

    try:
        job = args.job
        name = args.name

        if job == "inference":
            assert name in [model.name for model in models]

            model = [model for model in models if model.name == ]

        if job == "experiment":
            assert name in ["sparse-data"]
            if name == "sparse-data":
                experiment = SparseDataExperiment(config)

    except AssertionError:
        raise AssertionError(f"Invalid instance {name} for {job} job")

    if job == "inference":
        run_inference(config, model)

    if job == "experiment":
        run_experiment(config, experiment)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run HB-MEP"
    )
    parser.add_argument(
        "--job",
        choices=["inference", "experiment"],
        required=True,
        help="Job needed to run"
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Job to run"
    )

    try:
        args = parser.parse_args(sys.argv[2:])
    except IndexError:
        IndexError("Call to API requires an endpoint")

    main(args)
