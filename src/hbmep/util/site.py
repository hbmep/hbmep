class SiteAttribute(str):
    @property
    def raw(self):
        return SiteAttribute(f"{self}_raw")

    @property
    def log(self):
        return SiteAttribute(f"{self}_log")

    @property
    def loc(self):
        return SiteAttribute(f"{self}_loc")

    @property
    def scale(self):
        return SiteAttribute(f"{self}_scale")

    def __getitem__(self, key):
        # Handle indexing like site.num_features[0]
        if isinstance(key, int) and key >= 0:
            return SiteAttribute(f"{self}_{key}")
        raise KeyError(f"Key must be a non-negative integer, got {key}")


class SiteMeta(type):
    def __call__(cls, value):
        return SiteAttribute(value)


class Site(metaclass=SiteMeta):
    # Priors
    a = SiteAttribute("a")
    b = SiteAttribute("b")
    g = SiteAttribute("g")
    h = SiteAttribute("h")
    v = SiteAttribute("v")

    c1 = SiteAttribute("c₁")
    c2 = SiteAttribute("c₂")

    # Deterministic
    mu = SiteAttribute("µ")
    alpha = SiteAttribute("α")
    beta = SiteAttribute("β")

    # Plates
    num_features = SiteAttribute("num_features")
    num_response = SiteAttribute("num_response")
    num_data = SiteAttribute("num_data")

    # Observation
    obs = SiteAttribute("obs")

    # Mixture
    outlier_prob = SiteAttribute("p_outlier")
