# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import pandas
from IPython.display import HTML


def pandas_render(df: pandas.DataFrame) -> HTML:
    """
    Render a dataframe, hiding the index,
    and set ID to minimize diffs for notebook regression tests
    """
    return HTML(df.style.hide().set_uuid("my_id").to_html())
