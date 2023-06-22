import os
import sys
import logging
import argparse
from pathlib import Path

from hb_mep.config import HBMepConfig
from hb_mep.data_access import DataClass
from hb_mep.models import Baseline
from hb_mep.models.rats import (
    ReLU as RReLU,
    SaturatedReLU as RSaturatedReLU,
    RectifiedLogistic as RRectifiedLogistic
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
    mat, time = None, None

    """ Load config """
    config = HBMepConfig()
    data = DataClass(config)

    """ Available models """
    base_models = [Baseline]
    rats_models = base_models + [RReLU, RSaturatedReLU, RRectifiedLogistic]
    human_models = base_models + [HRectifiedLogistic]

    """ Load data """
    if args.dataset == "Human":
        models = human_models

        # subset = ["scapptio001"]
        # df = load_data_human(data=data, muscle="Triceps", subset=subset)
        df = load_data_human(data=data, muscle="Triceps")

    elif args.dataset == "Rats":
        models = rats_models

        dir_name = "physio2"
        dir = os.path.join(data.data_path, dir_name)
        participants = range(1, 9)

        df, mat, time = load_data_rats(dir=dir, participants=participants)

    """ Initialize model """
    assert args.model in [m(config).name for m in models]

    Model = [m for m in models if m(config).name == args.model][0]
    model = Model(config)

    """ Artefacts directory """
    data.make_dirs()
    postfix = f"{model.name}_{args.dataset}_{args.job}_{args.id}_{args.tag}"
    reports_path = Path(os.path.join(data.reports_path, postfix))
    reports_path.mkdir(parents=False, exist_ok=False)

    """ Run inference """
    run_inference(
        df=df, data=data, model=model, reports_path=reports_path, mat=mat, time=time
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
        "--id",
        required=True,
        help="Job id"
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Job tag"
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
