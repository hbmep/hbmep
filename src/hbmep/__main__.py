import sys
import argparse

from hbmep.config import Config
from hbmep.api import run_inference


def main(args):
    toml_path = args.config
    config = Config(toml_path=toml_path)
    run_inference(config=config)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run HB-MEP"
    )
    parser.add_argument(
        "config",
        help="Path to TOML configuration"
    )

    try:
        args = parser.parse_args(sys.argv[1:])
        print(args)
    except IndexError:
        IndexError("Call to API requires an endpoint")

    main(args)
