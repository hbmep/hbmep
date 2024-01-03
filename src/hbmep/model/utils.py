class Site:
    """ Priors """
    a = "a"
    b = "b"

    L = "L"
    ell = "ℓ"
    H = "H"
    v = "v"

    c_1 = "c₁"
    c_2 = "c₂"

    """ Deterministic """
    mu = "µ"
    beta = "β"

    """ Plates """
    n_features = [f"n_feature{i}" for i in range(10)]
    n_response = "n_response"
    n_data = "n_data"

    """ Observation """
    obs = "obs"

    """ Outlier distribution """
    outlier_prob = "p_outlier"
    outlier_scale = "σ_outlier"
