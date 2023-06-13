import sys
import logging
import argparse

from hb_mep.config import HBMepConfig
from hb_mep.data_access import DataClass
from hb_mep.models import Baseline
from hb_mep.models.rats import (
    RectifiedLogistic as RRectifiedLogistic,
    GammaRegression as RGammaRegression
)
from hb_mep.models.human import (
    RectifiedLogistic as HRectifiedLogistic
)
from hb_mep.models.rats.utils import load_data as load_data_rats
from hb_mep.models.human.utils import load_data as load_data_human
from hb_mep.api import run_inference

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


def main(args):
    config = HBMepConfig()
    data = DataClass(config)

    base_models = [Baseline]
    rats_models = base_models + [RRectifiedLogistic, RGammaRegression]
    human_models = base_models + [HRectifiedLogistic]

    if args.dataset == "Human":
        models = human_models

        subset = ["scapptio001"]
        df = load_data_human(data=data, muscle="Triceps", subset=subset)

    elif args.dataset == "Rats":
        models = rats_models

        a, b = 1, 4
        subset = range(a, b)
        df, _, _ = load_data_rats(subset=subset, data=data)

    assert args.model in [m(config).name for m in models]

    Model = [m for m in models if m(config).name == args.model][0]

    run_inference(
        df=df, config=config, data=data, Model=Model, id=args.dataset
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run HB-MEP"
    )
    parser.add_argument(
        "--job",
        required=True,
        choices=["Inference", "Experiment"],
        help="Job to run"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model to run"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["Rats", "Human"],
        help="Dataset to use"
    )

    try:
        args = parser.parse_args(sys.argv[2:])
    except IndexError:
        IndexError("Call to API requires an endpoint")

    main(args)
