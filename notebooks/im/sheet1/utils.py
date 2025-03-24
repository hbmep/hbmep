import logging 

from dataclasses import dataclass
import numpy as np
from hbmep.util import site
from hbmep.util.site import SiteAttribute

logger = logging.getLogger(__name__)


@dataclass
class Site(site):
    b1 = SiteAttribute("β₁")
    b2 = SiteAttribute("β₂")
    b3 = SiteAttribute("β₃")
    b4 = SiteAttribute("β₄")

    sigma = SiteAttribute("sigma")


def load(df, range_restricted):
    df = df.copy()
    mapping = {
        '1:10': '8_1:10',
        '1:100': '7_1:100',
        '1:1,000': '6_1:1K',
        '1:10,000': '5_1:10K',
        '1:100,000': '4_1:100K',
        '1:1,000,000': '3_1:1M',
        '1:10,000,000': '2_1:10M',
        '1:100,000,000': '1_1:100M',
        '0:1': '0_0:1',
    }
    df["contam"] = (
        df["dilution"]
        .replace({"0": "0:1"})
        .map(mapping)
    )
    df["contam"] = df[["contam", "plate"]].apply(
        lambda x: x.contam + f"({x.plate})" if x.contam == "0_0:1" else x.contam,
        axis=1
    )
    if range_restricted:
        conc = sorted(df.conc.unique().tolist())
        logger.info("Restricting range...")
        logger.info(f"Original df.shape: {df.shape}")
        logger.info(f"Original conc: {conc}")
        idx = (df.conc < 223) & (df.conc > 2)
        df = df[idx].reset_index(drop=True).copy()
        conc = sorted(df.conc.unique().tolist())
        logger.info(f"Restricted df.shape: {df.shape}")
        logger.info(f"Restricted conc: {conc}")
    return df
