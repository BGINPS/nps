import pickle
import numpy as np
import copy
import os
from typing import Dict, Optional, Tuple, Union, Literal, List


def read_pickle(
    pickle_file_path: str,
) -> dict:
    """Read a pickled file and return a dict

    Args:
        pickle_file_path (str): the file path of pickled file

    Returns:
        dict: the content of pickled file
    """
    data_dict = pickle.load(open(pickle_file_path, 'rb'))
    return data_dict


def read_obj_or_pickle(
    obj: Union[dict, str],
):
    if isinstance(obj, str):
        obj = read_pickle(obj)
    elif isinstance(obj, dict):
        obj = copy.deepcopy(obj)
    return obj

def save_pickle(
    obj,
    save_to_file: str,
):
    """Save an obj to a pickle file.

    Args:
        obj (_type_): the obj to be saved
        save_to_file (str): the file name (with or with path) for saving obj
    """
    assert save_to_file[-4:] == ".pkl", "please provide file name ends with .pkl"
    save_dir = os.path.dirname(os.path.abspath(save_to_file))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    o = open(save_to_file, 'wb')
    pickle.dump(obj, o)
    o.close()



def save_np_as_npy(
    a: np.ndarray,
    npy_file_name: str
):
    assert npy_file_name[-4:] == '.npy'
    with open(npy_file_name, "wb") as f:
        np.save(f, arr=a)

def load_npy_as_np(
    npy_file_name: str
) -> np.ndarray:
    assert npy_file_name[-4:] == '.npy'
    with open(npy_file_name, 'rb') as f:
        a = np.load(f)
    return a