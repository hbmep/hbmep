from dataclasses import dataclass
import numpy as np
from hbmep.util import site
from hbmep.util.site import SiteAttribute


@dataclass
class Site(site):
    b1 = SiteAttribute("β₁")
    b2 = SiteAttribute("β₂")
    b3 = SiteAttribute("β₃")
    b4 = SiteAttribute("β₄")


def load(df):
    df = df.copy()
    df["dilution_temp"] = (
        df["dilution"].replace({"0": "1:0"})
        .apply(lambda x: x.replace(",", ""))
        .apply(lambda x: x.split(":")[1])
        .apply(lambda x: int(x))
        .apply(lambda x: np.log10(x) if x > 0 else x)
        .apply(lambda x: int(x))
        .apply(lambda x: f"1e{x}" if x > 0 else f"{x}")
    )
    df["contam"] = (
        df[["plate", "dilution_temp"]].apply(
            lambda x: f"{x[1]}({x[0]})" if x[1] == "0" else x[1],
            axis=1
        )
    )
    return df[[col for col in df.columns if col != "dilution_temp"]]
