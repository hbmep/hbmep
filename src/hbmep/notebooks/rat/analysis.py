import numpy as np

ignore_warnings = np.errstate(divide='ignore', invalid='ignore')


def cap_response(*, model, df, y_unnorm, func, func_params):
    y = y_unnorm.copy()
    num_features = df[model.features].max().to_numpy() + 1

    # Get max at the highest tested intensity
    xcap = np.full((*num_features, model.num_response), np.nan)
    xcap.shape

    gdf = df.groupby(model.features, as_index=False)[model.intensity].max()
    features, max_intensity = gdf[model.features].values, gdf[model.intensity].values
    xcap[*features.T] = max_intensity[:, None]
    xcap.shape

    # Sanity check
    for combination in features:
        idx = gdf[model.features].apply(tuple, axis=1).isin([tuple(combination)])
        assert idx.sum() == 1
        temp_intensity = gdf[idx].reset_index(drop=True)[model.intensity].values.item()
        assert np.all(xcap[*combination] == temp_intensity)

    ycap = np.array(func(
        xcap[None, None, ...], *func_params
    ))
    y = np.where(y < ycap, y, ycap)
    return y


def evaluate_response(
    *,
    func,
    named_params,
    x,
    posterior,
):
    params = [posterior[param][None, ...] for param in named_params]
    g = params[2].copy()
    params[2] = params[2] * 0

    a_shape = params[0].shape
    a_ndim = len(a_shape)
    print(a_shape, a_ndim)

    # Response
    print("Evaluating y_unnorm...")
    y = np.array(func(
        x[:, *(None for _ in range(a_ndim - 1))], *params
    ))
    return y, g, params


def evaluate_entropy(
    *, x, y_norm
):
    y = y_norm.copy()
    # Calculate selectivity
    p = np.nansum(y, axis=-1, keepdims=True)

    with ignore_warnings:
        p = np.where(p, y / p, 1 / y.shape[-1])
        print(p.shape)

    with ignore_warnings:
        plogp = np.where(p, p * np.log(p), 0)
        print(plogp.shape)

    entropy = 1 + (plogp.sum(axis=-1) / np.log(y.shape[-1]))
    print(f"entropy.shape {entropy.shape}")
    print(f"entropy.isnan.sum: {np.isnan(entropy).sum()}")

    auc = np.trapz(y=entropy, x=x, axis=0)
    print(f"auc.shape {auc.shape}")

    return (
        p,
        plogp,
        entropy,
        auc
    )
