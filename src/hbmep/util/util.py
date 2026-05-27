import os
import sys
import logging
from time import time
from functools import wraps
from collections.abc import Iterable, Callable

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        time_taken = te - ts
        hours_taken = time_taken // (60 * 60)
        time_taken %= (60 * 60)
        minutes_taken = time_taken // 60
        time_taken %= 60
        seconds_taken = time_taken % 60
        if hours_taken:
            message = \
                f"func:{f.__name__} took: {hours_taken:0.0f} hr and " + \
                f"{minutes_taken:0.0f} min"
        elif minutes_taken:
            message = \
                f"func:{f.__name__} took: {minutes_taken:0.0f} min and " + \
                f"{seconds_taken:0.2f} sec"
        else:
            message = f"func:{f.__name__} took: {seconds_taken:0.2f} sec"
        logger.info(message)
        return result
    return wrap


def enable_logging(output=None, *, level=logging.INFO, format=FORMAT):
    handlers = [
        logging.StreamHandler(stream=sys.__stderr__),
    ]

    output_file = None

    if output is not None:
        root, ext = os.path.splitext(output)
        output_file = os.path.join(output, "logs.log") if not ext else output

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        handlers = [logging.FileHandler(output_file, mode="w")] + handlers

    logging.basicConfig(
        format=format,
        level=level,
        handlers=handlers,
        force=True
    )

    if output_file is not None:
        logger.info(f"Logging to {output_file}")

    return


def invert_combination(
    combination: tuple[int],
    columns: list[str],
    encoder: dict[str, LabelEncoder],
) -> tuple:
    return tuple(
        encoder[column].inverse_transform(np.array([value]))[0]
        for (column, value) in zip(columns, combination)
    )


def generate_response_colors(n: int, palette="rainbow", low=0, high=1):
    return sns.color_palette(palette=palette, as_cmap=True)(np.linspace(low, high, n))


def make_pdf(figures: list[Figure], output_path: str, dpi=100):
    """
    Save a list of matplotlib figures to a multi-page PDF.

    Args:
        figures (List[Figure]): List of figures to save.
        output_path (str): Path to the output PDF file.
    """
    logger.info(f"Saving pdf...")
    with PdfPages(output_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight', dpi=dpi)
            plt.close(fig)
    logger.info(f"Saved to {output_path}")
    return


def _record_would_be_handled_elsewhere(
    record: logging.LogRecord,
    this_handler: logging.Handler
) -> bool:
    """
    Walk up the logger hierarchy and see whether any other handler would handle
    this record (based on handler levels + filters).
    """
    logger = logging.getLogger(record.name)
    while logger is not None:
        for h in logger.handlers:
            if h is this_handler:
                continue
            if record.levelno >= h.level and h.filter(record):
                return True

        if not logger.propagate:
            break
        logger = logger.parent

    return False


class _HBMEPFallbackHandler(logging.StreamHandler):
    """
    Emits log records ONLY if they would otherwise be dropped (no handler in the
    logger chain would handle them at this level).

    This lets the library be verbose by default without double-printing when the
    user configures logging.
    """
    def emit(self, record: logging.LogRecord) -> None:
        if os.environ.get("HBMEP_DISABLE_FALLBACK_LOGGING") == "1":
            return

        if _record_would_be_handled_elsewhere(record, self):
            return

        super().emit(record)


def _enable_fallback_logging(*, level=logging.INFO, format=FORMAT, stream=None):
    """
    Install a fallback console handler for the hbmep logger so INFO logs show up
    even if the user never calls logging.basicConfig / enable_logging().
    It auto-disables itself when another handler would handle the record.
    """
    base = logging.getLogger("hbmep")
    if base.level == logging.NOTSET:
        base.setLevel(level)
    if any(isinstance(h, _HBMEPFallbackHandler) for h in base.handlers):
        return
    handler = _HBMEPFallbackHandler(stream or sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(format))
    base.addHandler(handler)


def run_batched(
    fn: Callable,
    tasks: Iterable,
    *,
    batch_size: int = 8,
    n_jobs: int | None = None,
    skip_none: bool = True,
    verbose: bool = True,
):
    """
    Run fn(*task) for tasks in parallel batches.
    """
    tasks = list(tasks)

    if n_jobs is None:
        n_jobs = -1

    out = []

    for start in range(0, len(tasks), batch_size):
        stop = min(start + batch_size, len(tasks))

        if verbose:
            print(f"Processing batch {start} to {stop}...")

        results = Parallel(n_jobs=n_jobs)(
            delayed(fn)(*task)
            for task in tasks[start:stop]
        )

        for r in results:
            if skip_none and r is None:
                continue
            out.append(r)

        del results

    return out
