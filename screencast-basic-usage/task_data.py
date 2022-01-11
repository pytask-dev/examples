import pandas as pd
import pytask


@pytask.mark.depends_on("auto.dta")
@pytask.mark.produces("bld/auto_final.dta")
def task_data_management(depends_on, produces):
    raw_data = pd.read_stata(depends_on)
    data = raw_data[["price", "mpg", "weight"]]
    data.to_stata(produces)
