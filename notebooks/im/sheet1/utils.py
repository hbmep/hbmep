from dataclasses import dataclass
from hbmep.util import site


@dataclass
class Site(site):
    b1 = "β₁"
    b2 = "β₂"
    b3 = "β₃"
    b4 = "β₄"

    @staticmethod
    def log(named_param):
        return named_param + "_log"
