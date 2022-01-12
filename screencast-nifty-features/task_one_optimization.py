import pandas as pd
import pytask
from estimagic import minimize

from config import BLD


PRODUCES = {
    "result": BLD / "scipy_lbfgsb" / "optimization.pkl",
    "log": BLD / "scipy_lbfgsb" / "optimization.db",
}


@pytask.mark.produces(PRODUCES)
def task_run_optimization(produces):
    start_params = pd.DataFrame(1, index=["a", "b", "c"], columns=["value"])
    res = minimize(
        criterion=_sphere,
        params=start_params,
        algorithm="scipy_lbfgsb",
        logging=produces["log"],
    )
    pd.to_pickle(res, produces["result"])


def _sphere(params):
    return (params["value"] ** 2).sum()
