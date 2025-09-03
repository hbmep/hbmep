# import os
# import pickle
# import shutil
# import logging

# import pandas as pd
# import numpy as np
# from jax import random
# from scipy.optimize import minimize
# from joblib import Parallel, delayed
# from sklearn.preprocessing import LabelEncoder

# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

# from hbmep.model import BaseModel
# from hbmep.util import timing, site, abstractvariables, floor, ceil
# # from hbmep.utils.constants import BOUND_CONSTRAINED_OPTIMIZATION, RECRUITMENT_CURVES

# logger = logging.getLogger(__name__)


# @abstractvariables(
#     ("method", "str: Type of `scipy.optimize.minimize` solver used to minimize the cost function."),
#     ("named_args", "list[str]: Named arguments of the cost function, for which it is minimized."),
#     ("bounds", "list[tuple(float, float)]: Bounds for the arguments of `named_args`."),
#     ("informed_bounds", "list[tuple(float, float)]: Informed bounds for the arguments of `named_args`."),
#     ("num_reinit", "int: Number of reinitializations for the optimization algorithm."),
#     ("n_jobs", "int: Number of parallel jobs to run with `joblib.Parallel`."),
# )
# class BoundConstrainedOptimization(BaseModel):
#     def __init__(self, *args, **kw):
#         super(BoundConstrainedOptimization, self).__init__(*args, **kw)

#     def functional(self, x, *args):
#         """
#         This method should return the predicted values of the response variable given
#         the input intensity values `x` and the arguments `args` for which the
#         cost function is minimized.
#         """
#         raise NotImplementedError

#     def cost_function(self, x, y_obs, *args):
#         """
#         This method should return the cost function to be minimized for arguments `args`.
#         The return value should be a scalar.

#         Note that the same arguments in `args` are passed to both the `functional` and `cost_function`.

#         E.g.: Suppose the cost function is the mean squared error between the observed
#         and predicted values of the response variable. Then, the cost function would be:

#         ```python
#         def cost_function(self, x, y_obs, *args):
#             y_pred = self.functional(x, *args)
#             return np.mean((y_obs - y_pred) ** 2)
#         ```

#         while, the `functional` method could be:

#         ```python
#         def functional(self, x, *args):
#             return args[0] + args[1] * x + args[2] * (x ** 2)
#         ```
#         """
#         raise NotImplementedError

#     @staticmethod
#     def _get_search_space(
#         rng_keys, bounds, informed_bounds, num_grid_points, num_reinit
#     ):
#         if informed_bounds is None:
#             if bounds is not None: informed_bounds = bounds
#             grid = [
#                 random.uniform(key=rng_key, shape=(num_reinit,))
#                 for rng_key in rng_keys
#             ]

#         else:
#             grid = [
#                 np.linspace(lo, hi, num_grid_points)
#                 for lo, hi in informed_bounds
#             ]
#             grid = [
#                 random.choice(
#                     key=rng_key,
#                     a=arr,
#                     shape=(num_reinit,),
#                     replace=True
#                 )
#                 for arr, rng_key in zip(grid, rng_keys)
#             ]

#         grid = [np.array(arr).tolist() for arr in grid]
#         grid = list(zip(*grid))
#         return grid

#     @staticmethod
#     def _get_named_args(args, named_args):
#         # TODO: Rename this method to make more sense
#         return {u: dict(zip(named_args, v)) for u, v in args.items()}

#     def optimize(self, x, y_obs, build_dir, **kwargs):
#         rng_keys = random.split(self.rng_key, num=len(self.named_args))
#         grid = BoundConstrainedOptimization._get_search_space(
#             rng_keys=rng_keys,
#             bounds=self.bounds,
#             informed_bounds=self.informed_bounds,
#             num_grid_points=10 * self.num_reinit,
#             num_reinit=self.num_reinit
#         )

#         temp_dir = os.path.join(build_dir, f"optimize_results")
#         if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
#         os.makedirs(temp_dir, exist_ok=False)


#         def _body_fn_optimize(x, y_obs, x0, destination_path):
#             res = minimize(
#                 lambda coeffs: self.cost_function(x, y_obs, *coeffs),
#                 x0=x0,
#                 bounds=self.bounds,
#                 method=self.method,
#                 **kwargs
#             )
#             with open(destination_path, "wb") as f:
#                 pickle.dump((res,), f)


#         with Parallel(n_jobs=self.n_jobs) as parallel:
#             parallel(
#                 delayed(_body_fn_optimize)(
#                     x, y_obs, x0, os.path.join(temp_dir, f"param{i}.pkl")
#                 )
#                 for i, x0 in enumerate(grid)
#             )

#         result = []
#         for i, _ in enumerate(grid):
#             src = os.path.join(temp_dir, f"param{i}.pkl")
#             with open(src, "rb") as g:
#                 result.append(pickle.load(g)[0])

#         if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

#         estimated_args = [r.x for r in result]
#         errors = [r.fun for r in result]
#         argmin = np.argmin(errors)
#         estimated_args = estimated_args[argmin]

#         dest = os.path.join(build_dir, "estimated_args.pkl")
#         with open(dest, "wb") as f:
#             pickle.dump(estimated_args, f)

#         return

#     @timing
#     def run(self, df: pd.DataFrame, **kwargs):
#         combinations = self._get_combinations(df, self.features)


#         def _body_fn_run(combination, response_ind):
#             nonlocal df
#             ind = (
#                 df[self.features]
#                 .apply(tuple, axis=1)
#                 .isin([combination])
#             )
#             df_ = df[ind].reset_index(drop=True).copy()
#             x, y_obs = (
#                 df_[self.intensity].values,
#                 df_[self.response[response_ind]].values
#             )

#             build_dir = os.path.join(
#                 self.build_dir, f"{combination}__{response_ind}"
#             )
#             self.optimize(x, y_obs, build_dir, **kwargs)
#             return


#         with Parallel(n_jobs=self.n_jobs) as parallel:
#             parallel(
#                 delayed(_body_fn_run)(combination, response_ind)
#                 for combination in combinations
#                 for response_ind in range(self.n_response)
#             )

#         args = {}
#         for combination in combinations:
#             for response_ind, _ in enumerate(self.response):
#                 build_dir = os.path.join(
#                     self.build_dir, f"{combination}__{response_ind}"
#                 )

#                 src = os.path.join(build_dir, "estimated_args.pkl")
#                 with open(src, "rb") as f:
#                     estimated_args = pickle.load(f)

#                 args[(*combination, response_ind)] = estimated_args
#                 if os.path.exists(build_dir): shutil.rmtree(build_dir)

#         named_params = BoundConstrainedOptimization._get_named_args(
#             args=args, named_args=self.named_args
#         )

#         params = {}
#         n_features = df[self.features].max().values + 1

#         for named_arg in self.named_args:
#             params[named_arg] = np.full((*n_features, self.n_response), np.nan)

#         for u, v in named_params.items():
#             for named_param, value in v.items():
#                 params[named_param][*u] = value

#         return params

#     @timing
#     def predict(
#         self,
#         df: pd.DataFrame,
#         params: dict
#     ):
#         df = df.copy()
#         combinations = self._get_combinations(df=df, columns=self.features)

#         for response_ind, response in enumerate(self.response):
#             df[response] = 0

#             for combination in combinations:
#                 ind = (
#                     df[self.features]
#                     .apply(tuple, axis=1)
#                     .isin([combination])
#                 )
#                 temp_df = df[ind].reset_index(drop=True).copy()
#                 y_pred = self.functional(
#                     temp_df[self.intensity].values,
#                     *(
#                         params[named_param][*combination, response_ind]
#                         for named_param in self.named_args
#                     )
#                 )
#                 df.loc[ind, response] = y_pred

#         return df

#     @timing
#     def render_recruitment_curves(
#         self,
#         df: pd.DataFrame,
#         encoder_dict: dict[str, LabelEncoder] | None = None,
#         params: dict | None = None,
#         prediction_df: pd.DataFrame | None = None,
#         mep_matrix: np.ndarray | None = None,
#         destination_path: str | None = None,
#         **kwargs
#     ):
#         # TODO: Integrate this within Plotter class (from hbmep.plotter.core)
#         if destination_path is None: destination_path = os.path.join(
#             self.build_dir, RECRUITMENT_CURVES
#         )
#         logger.info("Rendering recruitment curves ...")
#         combination_columns = kwargs.get("combination_columns", self.features)
#         orderby = kwargs.get("orderby")
#         intensity = kwargs.get("intensity", self.intensity)
#         response = kwargs.get("response", self.response)
#         response_colors = kwargs.get("response_colors", self.response_colors)
#         base = kwargs.get("base", self.base)
#         subplot_cell_width = kwargs.get(
#             "subplot_cell_width", self.subplot_cell_width
#         )
#         subplot_cell_height = kwargs.get(
#             "subplot_cell_height", self.subplot_cell_height
#         )
#         recruitment_curve_props = kwargs.get(
#             "recruitment_curve_props", self.recruitment_curve_props
#         )

#         if mep_matrix is not None:
#             assert mep_matrix.shape[0] == df.shape[0]
#             a, b = self.mep_window
#             time = np.linspace(a, b, mep_matrix.shape[1])
#             is_within_mep_size_window = (
#                 (time > self.mep_size_window[0])
#                 & (time < self.mep_size_window[1])
#             )

#         if params is not None:
#             assert prediction_df is not None

#         # Setup pdf layout
#         combinations = self._get_combinations(
#             df=df, columns=combination_columns, orderby=orderby
#         )
#         n_combinations = len(combinations)
#         n_response = len(response)

#         n_columns_per_response = 1
#         if mep_matrix is not None: n_columns_per_response += 1
#         if params is not None: n_columns_per_response += 2

#         n_fig_rows = 10
#         n_fig_columns = n_columns_per_response * n_response
#         n_pdf_pages = n_combinations // n_fig_rows
#         if n_combinations % n_fig_rows: n_pdf_pages += 1

#         # Iterate over pdf pages
#         pdf = PdfPages(destination_path)
#         combination_counter = 0

#         for page in range(n_pdf_pages):
#             n_rows_current_page = min(
#                 n_fig_rows,
#                 n_combinations - page * n_fig_rows
#             )
#             fig, axes = plt.subplots(
#                 nrows=n_rows_current_page,
#                 ncols=n_fig_columns,
#                 figsize=(
#                     n_fig_columns * subplot_cell_width,
#                     n_rows_current_page * subplot_cell_height
#                 ),
#                 constrained_layout=True,
#                 squeeze=False
#             )

#             # Iterate over combinations
#             for i in range(n_rows_current_page):
#                 curr_combination = combinations[combination_counter]
#                 curr_combination_inverse = ""

#                 if encoder_dict is not None:
#                     curr_combination_inverse = self._get_combination_inverse(
#                         combination=curr_combination,
#                         columns=combination_columns,
#                         encoder_dict=encoder_dict
#                     )
#                     curr_combination_inverse = ", ".join(map(str, curr_combination_inverse))
#                     curr_combination_inverse += "\n"

#                 # Filter dataframe based on current combination
#                 df_ind = df[combination_columns].apply(tuple, axis=1).isin([curr_combination])
#                 curr_df = df[df_ind].reset_index(drop=True).copy()

#                 if params is not None:
#                     # Filter prediction dataframe based on current combination
#                     prediction_df_ind = prediction_df[combination_columns].apply(tuple, axis=1).isin([curr_combination])
#                     curr_prediction_df = prediction_df[prediction_df_ind].reset_index(drop=True).copy()

#                 # Tickmarks
#                 min_intensity, max_intensity_ = curr_df[intensity].agg([min, max])
#                 min_intensity = floor(min_intensity, base=base)
#                 max_intensity = ceil(max_intensity_, base=base)
#                 if max_intensity == max_intensity_:
#                     max_intensity += base
#                 curr_x_ticks = np.arange(min_intensity, max_intensity, base)

#                 axes[i, 0].set_xlabel(intensity)
#                 axes[i, 0].set_xticks(ticks=curr_x_ticks)
#                 axes[i, 0].set_xlim(left=min_intensity - (base // 2), right=max_intensity + (base // 2))

#                 # Iterate over responses
#                 j = 0
#                 for r, response_muscle in enumerate(response):
#                     if params is not None:
#                         curr_threshold = params[site.a][*curr_combination, r]

#                     # Labels
#                     prefix = f"{tuple(list(curr_combination) + [r])}: {response_muscle} - MEP"
#                     if not j: prefix = curr_combination_inverse + prefix

#                     # MEP data
#                     if mep_matrix is not None:
#                         postfix = " - MEP"
#                         ax = axes[i, j]
#                         mep_response_ind = [
#                             i
#                             for i, _response_muscle in enumerate(self.mep_response)
#                             if _response_muscle == response_muscle
#                         ]
#                         mep_response_ind = mep_response_ind[0]
#                         curr_mep_matrix = mep_matrix[df_ind, :, mep_response_ind]
#                         max_amplitude = curr_mep_matrix[..., is_within_mep_size_window].max()
#                         curr_mep_matrix = (curr_mep_matrix / max_amplitude) * (base // 2)

#                         ax = self.mep_plot(
#                             ax,
#                             curr_mep_matrix,
#                             curr_df[intensity],
#                             time,
#                             color=response_colors[r],
#                             alpha=.4
#                         )
#                         ax.axhline(
#                             y=self.mep_size_window[0], color="r", linestyle="--", alpha=.4, label="MEP Size Window"
#                         )
#                         ax.axhline(
#                             y=self.mep_size_window[1], color="r", linestyle="--", alpha=.4
#                         )
#                         ax.set_ylim(bottom=-0.001, top=self.mep_size_window[1] + (self.mep_size_window[0] - (-0.001)))
#                         ax.set_ylabel("Time")
#                         ax.set_title(prefix + postfix)
#                         ax.sharex(axes[i, 0])
#                         ax.tick_params(axis="x", rotation=90)
#                         if j > 0 and ax.get_legend() is not None: ax.get_legend().remove()
#                         j += 1

#                     # MEP Size scatter plot
#                     postfix = " - MEP Size"
#                     ax = axes[i, j]
#                     sns.scatterplot(data=curr_df, x=intensity, y=response_muscle, color=response_colors[r], ax=ax)
#                     ax.set_ylabel(response_muscle)
#                     ax.set_title(prefix + postfix)
#                     ax.sharex(axes[i, 0])
#                     ax.tick_params(axis="x", rotation=90)
#                     j += 1

#                     if params is not None:
#                         # MEP Size scatter plot and recruitment curve
#                         postfix = "Recruitment Curve Fit"
#                         ax = axes[i, j]
#                         sns.scatterplot(data=curr_df, x=intensity, y=response_muscle, color=response_colors[r], ax=ax)
#                         sns.lineplot(
#                             x=curr_prediction_df[intensity],
#                             y=curr_prediction_df[response_muscle],
#                             ax=ax,
#                             **recruitment_curve_props,
#                         )
#                         ax.axvline(
#                             curr_threshold,
#                             linestyle="--",
#                             color=response_colors[r],
#                             label="Threshold"
#                         )
#                         ax.set_title(postfix)
#                         ax.sharex(axes[i, 0])
#                         ax.sharey(axes[i, j - 1])
#                         ax.tick_params(axis="x", rotation=90)
#                         j += 1

#                         # Threshold
#                         ax = axes[i, j]
#                         postfix = "Threshold Estimate"
#                         ax.axvline(
#                             curr_threshold,
#                             linestyle="--",
#                             color=response_colors[r],
#                             label="Threshold"
#                         )
#                         ax.set_xlabel(intensity)
#                         ax.set_title(postfix)
#                         if j > 0 and ax.get_legend(): ax.get_legend().remove()
#                         j += 1

#                 combination_counter += 1

#             pdf.savefig(fig)
#             plt.close()

#         pdf.close()
#         plt.show()

#         logger.info(f"Saved to {destination_path}")
#         return
