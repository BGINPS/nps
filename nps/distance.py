from dtaidistance import dtw
import numpy as np
import pandas as pd
from .tools import extract_signals_in_windows_for_one_obj, down_sampling


def cal_dtw(
    obj_dict,
    down_sample_to: int = 0
)->pd.DataFrame:
    """Calculate dtw distance for any two reads in obj

    Args:
        obj_dict (_type_): obj
        down_sample_to (int): down sample signals in window to this size. If is 0, no down sampling.
    Returns:
        pd.DataFrame: distance matrix
    """
    signals_in_windows = extract_signals_in_windows_for_one_obj(obj_dict)
    if down_sample_to > 0:
        for read_id, signal in signals_in_windows.items():
            signals_in_windows[read_id] = down_sampling(signal, down_sample_to=down_sample_to)
    timeseries = list(signals_in_windows.values())
    ds = dtw.distance_matrix_fast(timeseries)
    max_dis = np.max(ds[(ds!=np.inf) & (~np.isnan(ds))])
    ds[(ds==np.inf) | (np.isnan(ds))] = max_dis
    ds_df = pd.DataFrame(ds, columns=obj_dict.keys(), index=obj_dict.keys())
    return ds_df

