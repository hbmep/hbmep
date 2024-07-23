class Site:
    # Priors
    a = "a"
    b = "b"
    v = "v"
    L = "L"
    ell = "ℓ"
    H = "H"

    c_1 = "c₁"
    c_2 = "c₂"

    # Deterministic
    mu = "µ"
    alpha = "α"
    beta = "β"

    # Plates
    n_features = [f"n_feature{i}" for i in range(10)]
    n_response = "n_response"
    n_data = "n_data"

    # Observation
    obs = "obs"

    # Mixture
    outlier_prob = "p_outlier"
    outlier_scale = "σ_outlier"
    q = "q"
    bg_scale = "σ_bg"

    # S50
    s50 = "S50"

    # Outlier classifier
    p = "p"
