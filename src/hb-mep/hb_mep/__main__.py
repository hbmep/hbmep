import sys
import logging

from hb_mep.api import run_inference


if __name__ == "__main__":
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    try:
        if sys.argv[1] == "run":
            run_inference()
    except IndexError:
        raise ValueError("Call to API requires an endpoint")