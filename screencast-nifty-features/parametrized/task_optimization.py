import numpy as np
import pandas as pd
import plotly.express as px
import pytask
from estimagic import minimize
from estimagic.logging.read_log import read_optimization_histories

from config import BLD

_ALGOS = ["scipy_lbfgsb", "scipy_neldermead"]
PRODUCES_ALGOS = [
    [
        {
            "result": BLD / a / "optimization.pkl",
            "log": BLD / a / "optimization.db",
        },
        algo,
    ]
    for algo in _ALGOS
]


@pytask.mark.parametrize("produces, algo", PRODUCES_ALGOS, ids=_ALGOS)
def task_run_optimization(produces, algo):
    start_params = pd.DataFrame(1, index=["a", "b", "c"], columns=["value"])
    res = minimize(
        criterion=_sphere,
        params=start_params,
        algorithm=algo,
        logging=produces["log"],
    )
    pd.to_pickle(res, produces["result"])


def _sphere(params):
    return (params["value"] ** 2).sum()


@pytask.mark.depends_on(pa[0]["log"] for pa in PRODUCES_ALGOS)
@pytask.mark.produces(BLD / "history_plot.png")
def task_plot_histories(depends_on, produces):
    to_concat = []
    for algo, path in depends_on.items():
        history = read_optimization_histories(path)["values"].to_frame(name="value")
        history["eval"] = np.arange(len(history))
        history["algorithm"] = algo
        to_concat.append(history)

    data = pd.concat(to_concat)

    fig = px.line(data, x="eval", y="value", color="algorithm", template="plotly_dark")
    fig.write_image(produces)
