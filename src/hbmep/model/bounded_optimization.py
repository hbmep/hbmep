import os
import pickle
import shutil
import logging
from abc import abstractmethod

import pandas as pd
import numpy as np
from jax import random
from scipy.optimize import minimize
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from hbmep.config import Config
from hbmep.model import BaseModel
from hbmep.model.utils import Site as site
from hbmep.utils import timing, floor, ceil
from hbmep.utils.constants import BOUNDED_OPTIMIZATION, RECRUITMENT_CURVES

logger = logging.getLogger(__name__)


# Courtesy of https://stackoverflow.com/a/56997348/6937963
def abstractvariables(*args):
    class av:
        def __init__(self, error_message):
            self.error_message = error_message

        def __get__(self, *args, **kwargs):
            raise NotImplementedError(self.error_message)

    def f(cls):
        for arg, message in args:
            setattr(cls, arg, av(f"Descendants must set variable `{arg}`. {message}"))
        return cls

    return f


@abstractvariables(
    ("solver", "Solver not implemented."),
    ("functional", "Parametric curve function not implemented."),
    ("named_params", "Named parameters for functional not specified."),
    ("bounds", "Bounds not specified."),
    ("informed_bounds", "Informed bounds not specified."),
    ("num_points", "Number of points for grid search space not specified."),
    ("num_iters", "Number of iterations for re-initializing solver not specified."),
    ("n_jobs", "Number of parallel jobs not specified.")
)
class BoundedOptimization(BaseModel):
    NAME = BOUNDED_OPTIMIZATION

    def __init__(self, config: Config):
        super(BoundedOptimization, self).__init__(config=config)

    def _get_search_space(self):
        rng_keys = random.split(self.rng_key, num=len(self.bounds))
        grid = [np.linspace(lo, hi, self.num_points) for lo, hi in self.informed_bounds]
        grid = [
            random.choice(key=rng_key, a=arr, shape=(self.num_iters,), replace=True)
            for arr, rng_key in zip(grid, rng_keys)
        ]
        grid = [np.array(arr).tolist() for arr in grid]
        grid = list(zip(*grid))
        return grid

    def _get_named_params(self, params):
        return {
            u: dict(zip(self.named_params, v))
            for u, v in params.items()
        }

    def cost_function(self, x, y, *args):
        y_pred = self.functional(x, *args)
        return np.sum((y - y_pred) ** 2)

    @timing
    def run_inference(self, df: pd.DataFrame):
        grid = self._get_search_space()
        combinations = self._get_combinations(df, self.features)
        temp_dir = os.path.join(self.build_dir, "optimize_results")

        def body_fn_optimize(x, y, x0, destination_path):
            res = minimize(
                lambda coeffs: self.cost_function(x, y, *coeffs),
                x0=x0,
                bounds=self.bounds,
                method=self.solver
            )
            with open(destination_path, "wb") as f:
                pickle.dump((res,), f)

        params = {}
        for combination in combinations:
            for response_ind, response in enumerate(self.response):
                ind = (
                    df[self.features]
                    .apply(tuple, axis=1)
                    .isin([combination])
                )
                df_ = df[ind].reset_index(drop=True).copy()
                x, y = df_[self.intensity].values, df_[response].values

                if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
                assert not os.path.exists(temp_dir)
                self._make_dir(temp_dir)

                with Parallel(n_jobs=self.n_jobs) as parallel:
                    parallel(
                        delayed(body_fn_optimize)(
                            x, y, x0, os.path.join(temp_dir, f"param{i}.pkl")
                        )
                        for i, x0 in enumerate(grid)
                    )

                res = []
                for i, _ in enumerate(grid):
                    src = os.path.join(temp_dir, f"param{i}.pkl")
                    with open(src, "rb") as g:
                        res.append(pickle.load(g)[0])

                estimated_params = [r.x for r in res]
                errors = [r.fun for r in res]
                argmin = np.argmin(errors)
                params[(*combination, response_ind)] = estimated_params[argmin]

        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

        named_params = self._get_named_params(params)
        _, features = self._get_regressors(df)
        n_features = np.max(features, axis=0) + 1

        params = {}
        for named_param in self.named_params:
            params[named_param] = np.full((*n_features, self.n_response), np.nan)

        for u, v in named_params.items():
            for named_param, value in v.items():
                params[named_param][*u] = value

        return params

    @timing
    def predict(
        self,
        df: pd.DataFrame,
        params: dict
    ):
        df = df.copy()
        combinations = self._get_combinations(df=df, columns=self.features)

        for response_ind, response in enumerate(self.response):
            df[response] = 0

            for combination in combinations:
                ind = (
                    df[self.features]
                    .apply(tuple, axis=1)
                    .isin([combination])
                )
                temp_df = df[ind].reset_index(drop=True).copy()
                y_pred = self.functional(
                    temp_df[self.intensity].values,
                    *(
                        params[named_param][*combination, response_ind]
                        for named_param in self.named_params
                    )
                )
                df.loc[ind, response] = y_pred

        return df

    @timing
    def render_recruitment_curves(
        self,
        df: pd.DataFrame,
        encoder_dict: dict[str, LabelEncoder] | None = None,
        params: dict | None = None,
        prediction_df: pd.DataFrame | None = None,
        mep_matrix: np.ndarray | None = None,
        destination_path: str | None = None,
        **kwargs
    ):
        # TODO: Integrate this within Plotter class (from hbmep.plotter.core)
        if destination_path is None: destination_path = os.path.join(
            self.build_dir, RECRUITMENT_CURVES
        )
        logger.info("Rendering recruitment curves ...")
        combination_columns = kwargs.get("combination_columns", self.features)
        orderby = kwargs.get("orderby")
        intensity = kwargs.get("intensity", self.intensity)
        response = kwargs.get("response", self.response)
        response_colors = kwargs.get("response_colors", self.response_colors)
        base = kwargs.get("base", self.base)
        subplot_cell_width = kwargs.get("subplot_cell_width", self.subplot_cell_width)
        subplot_cell_height = kwargs.get("subplot_cell_height", self.subplot_cell_height)
        recruitment_curve_props = kwargs.get("recruitment_curve_props", self.recruitment_curve_props)

        if mep_matrix is not None:
            assert mep_matrix.shape[0] == df.shape[0]
            a, b = self.mep_window
            time = np.linspace(a, b, mep_matrix.shape[1])
            is_within_mep_size_window = (time > self.mep_size_window[0]) & (time < self.mep_size_window[1])

        if params is not None:
            assert prediction_df is not None

        # Setup pdf layout
        combinations = self._get_combinations(df=df, columns=combination_columns, orderby=orderby)
        n_combinations = len(combinations)
        n_response = len(response)

        n_columns_per_response = 1
        if mep_matrix is not None: n_columns_per_response += 1
        if params is not None: n_columns_per_response += 2

        n_fig_rows = 10
        n_fig_columns = n_columns_per_response * n_response
        n_pdf_pages = n_combinations // n_fig_rows
        if n_combinations % n_fig_rows: n_pdf_pages += 1

        # Iterate over pdf pages
        pdf = PdfPages(destination_path)
        combination_counter = 0

        for page in range(n_pdf_pages):
            n_rows_current_page = min(
                n_fig_rows,
                n_combinations - page * n_fig_rows
            )
            fig, axes = plt.subplots(
                nrows=n_rows_current_page,
                ncols=n_fig_columns,
                figsize=(
                    n_fig_columns * subplot_cell_width,
                    n_rows_current_page * subplot_cell_height
                ),
                constrained_layout=True,
                squeeze=False
            )

            # Iterate over combinations
            for i in range(n_rows_current_page):
                curr_combination = combinations[combination_counter]
                curr_combination_inverse = ""

                if encoder_dict is not None:
                    curr_combination_inverse = self._get_combination_inverse(
                        combination=curr_combination,
                        columns=combination_columns,
                        encoder_dict=encoder_dict
                    )
                    curr_combination_inverse = ", ".join(map(str, curr_combination_inverse))
                    curr_combination_inverse += "\n"

                # Filter dataframe based on current combination
                df_ind = df[combination_columns].apply(tuple, axis=1).isin([curr_combination])
                curr_df = df[df_ind].reset_index(drop=True).copy()

                if params is not None:
                    # Filter prediction dataframe based on current combination
                    prediction_df_ind = prediction_df[combination_columns].apply(tuple, axis=1).isin([curr_combination])
                    curr_prediction_df = prediction_df[prediction_df_ind].reset_index(drop=True).copy()

                # Tickmarks
                min_intensity, max_intensity_ = curr_df[intensity].agg([min, max])
                min_intensity = floor(min_intensity, base=base)
                max_intensity = ceil(max_intensity_, base=base)
                if max_intensity == max_intensity_:
                    max_intensity += base
                curr_x_ticks = np.arange(min_intensity, max_intensity, base)

                axes[i, 0].set_xlabel(intensity)
                axes[i, 0].set_xticks(ticks=curr_x_ticks)
                axes[i, 0].set_xlim(left=min_intensity - (base // 2), right=max_intensity + (base // 2))

                # Iterate over responses
                j = 0
                for r, response_muscle in enumerate(response):
                    if params is not None:
                        curr_threshold = params[site.a][*curr_combination, r]

                    # Labels
                    prefix = f"{tuple(list(curr_combination) + [r])}: {response_muscle} - MEP"
                    if not j: prefix = curr_combination_inverse + prefix

                    # MEP data
                    if mep_matrix is not None:
                        postfix = " - MEP"
                        ax = axes[i, j]
                        mep_response_ind = [
                            i
                            for i, _response_muscle in enumerate(self.mep_response)
                            if _response_muscle == response_muscle
                        ]
                        mep_response_ind = mep_response_ind[0]
                        curr_mep_matrix = mep_matrix[df_ind, :, mep_response_ind]
                        max_amplitude = curr_mep_matrix[..., is_within_mep_size_window].max()
                        curr_mep_matrix = (curr_mep_matrix / max_amplitude) * (base // 2)

                        ax = self.mep_plot(
                            ax,
                            curr_mep_matrix,
                            curr_df[intensity],
                            time,
                            color=response_colors[r],
                            alpha=.4
                        )
                        ax.axhline(
                            y=self.mep_size_window[0], color="r", linestyle="--", alpha=.4, label="MEP Size Window"
                        )
                        ax.axhline(
                            y=self.mep_size_window[1], color="r", linestyle="--", alpha=.4
                        )
                        ax.set_ylim(bottom=-0.001, top=self.mep_size_window[1] + (self.mep_size_window[0] - (-0.001)))
                        ax.set_ylabel("Time")
                        ax.set_title(prefix + postfix)
                        ax.sharex(axes[i, 0])
                        ax.tick_params(axis="x", rotation=90)
                        if j > 0 and ax.get_legend() is not None: ax.get_legend().remove()
                        j += 1

                    # MEP Size scatter plot
                    postfix = " - MEP Size"
                    ax = axes[i, j]
                    sns.scatterplot(data=curr_df, x=intensity, y=response_muscle, color=response_colors[r], ax=ax)
                    ax.set_ylabel(response_muscle)
                    ax.set_title(prefix + postfix)
                    ax.sharex(axes[i, 0])
                    ax.tick_params(axis="x", rotation=90)
                    j += 1

                    if params is not None:
                        # MEP Size scatter plot and recruitment curve
                        postfix = "Recruitment Curve Fit"
                        ax = axes[i, j]
                        sns.scatterplot(data=curr_df, x=intensity, y=response_muscle, color=response_colors[r], ax=ax)
                        sns.lineplot(
                            x=curr_prediction_df[intensity],
                            y=curr_prediction_df[response_muscle],
                            ax=ax,
                            **recruitment_curve_props,
                        )
                        ax.axvline(
                            curr_threshold,
                            linestyle="--",
                            color=response_colors[r],
                            label="Threshold"
                        )
                        ax.set_title(postfix)
                        ax.sharex(axes[i, 0])
                        ax.sharey(axes[i, j - 1])
                        ax.tick_params(axis="x", rotation=90)
                        j += 1

                        # Threshold
                        ax = axes[i, j]
                        postfix = "Threshold Estimate"
                        ax.axvline(
                            curr_threshold,
                            linestyle="--",
                            color=response_colors[r],
                            label="Threshold"
                        )
                        ax.set_xlabel(intensity)
                        ax.set_title(postfix)
                        if j > 0 and ax.get_legend(): ax.get_legend().remove()
                        j += 1

                combination_counter += 1

            pdf.savefig(fig)
            plt.close()

        pdf.close()
        plt.show()

        logger.info(f"Saved to {destination_path}")
        return
