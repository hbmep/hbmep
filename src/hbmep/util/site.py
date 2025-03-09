from dataclasses import dataclass


@dataclass
class Site:
    # Priors
    a = "a"
    b = "b"
    g = "g"
    h = "h"
    v = "v"

    c1 = "c₁"
    c2 = "c₂"

    # Deterministic
    mu = "µ"
    alpha = "α"
    beta = "β"

    # Plates
    num_features = [f"num_feature{i}" for i in range(10)]
    num_response = "num_response"
    num_data = "num_data"


    # Observation
    obs = "obs"

    # Mixture
    outlier_prob = "p_outlier"

    @staticmethod
    def raw(site):
        return site + "_raw"

    @staticmethod
    def loc(site):
        return site + "_loc"
    
    @staticmethod
    def scale(site):
        return site + "_scale"
