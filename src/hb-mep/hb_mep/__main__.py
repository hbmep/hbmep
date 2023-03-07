from pyexpat import model
import sys
import logging
import argparse

from hb_mep.config import HBMepConfig
from hb_mep.models.baseline import Baseline
from hb_mep.models.saturated_exponential import SaturatedExponential
# from hb_mep.models.logistic_regression import LogisticRegression
from hb_mep.api import run_inference

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


def main(args):
    config = HBMepConfig()
    try:
        model_name = args.model
        if sys.argv[1] == "run":
            if model_name == 'baseline':
                model = Baseline(config)
            elif model_name == 'saturated-exponential':
                model = SaturatedExponential(config)
            elif model_name == 'logistic-regression':
                config.ZERO_ONE = True
                model = LogisticRegression(config)
            else:
                raise ValueError
    except IndexError:
        raise IndexError("Call to API requires an endpoint")
    except AttributeError:
        raise AttributeError("Required model name")
    except ValueError:
        raise ValueError(f"Model {model_name} does not exist")
    config.PLOT_FIT = model.name + '_' + config.RESPONSE_MUSCLES[0] + '_' + config.PLOT_FIT
    config.PLOT_KDE = model.name + '_' + config.RESPONSE_MUSCLES[0] + '_' + config.PLOT_KDE
    run_inference(model=model, config=config)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HB MEP Inference"
    )
    parser.add_argument(
        "--model",
        default='baseline',
        help="Model name"
    )
    args = parser.parse_args(sys.argv[2:])
    main(args)
