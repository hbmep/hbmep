from pyexpat import model
import sys
import logging
import argparse

from hb_mep.config import HBMepConfig
from hb_mep.models import Baseline, MixedEffects
from hb_mep.experiments import Experiment, SparseDataExperiment
from hb_mep.api import run_inference, run_experiment

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


def main(args):
    config = HBMepConfig()
    try:
        job = args.job
        name = args.name

        if job == "inference":
            assert name in ["baseline", "mixed-effects"]
            if name == "baseline":
                model = Baseline(config)
            elif name == "mixed-effects":
                model = MixedEffects(config)

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
        help="Job to run"
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
