import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from . import preprocessing as pp
from typing import Optional, Literal, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from . import tools as tl
from scipy.spatial.distance import pdist, squareform

def get_density_matrix_from_an_obj(
    obj: dict,
    y_len: Optional[int] = 1000,
    x_len: Optional[int] = 1000,
    target: Literal['dna1', 'window', 'dna2'] = 'window',
) -> np.ndarray:
    """Generate density of signal of an obj.

       I0 -> 1

    y

       0  -> 0          1000
                   x

    Args:
        obj (dict): obj
        y_len (Optional[int], optional): the height of matrix, indicates I/I0. Defaults to 1000.
        x_len (Optional[int], optional): the width of matrix, indicates signal length. Defaults to 1000.
        target (Literal['dna1', 'window', 'dna2']): extract this part of the read. Defaults to 'window'.

    Returns:
        np.ndarray: density matrix. The first row indicates I0, the last indicates current with 0.
    """   
    all_read_ids = list(obj.keys())
    X = pp.get_signals_for_reads_in_an_obj(obj, all_read_ids, scale_to=x_len, normalize_by_openpore=True, target=target)
    X = np.array(X, dtype=np.float32)
    X = y_len - 1 - np.round(X*y_len).astype(np.int16)
    X[X<0] = 0

    data = np.ones(X.size)
    ys = X.flatten()
    xs = list(range(x_len)) * len(X)

    den_arr = csr_matrix((data, (ys, xs)), shape=(y_len, x_len)).A

    den_arr = den_arr/den_arr.sum(axis=0)

    return den_arr


def get_density_matrix_for_each_label_from_an_obj(
    obj: dict,
    labels: list,
    y_len: Optional[int] = 1000,
    x_len: Optional[int] = 1000,
    target: Literal['dna1', 'window', 'dna2'] = 'window',
) -> dict:
    density_matrix_dict = {}
    for label in labels:
        one_obj = pp.extract_reads_with_labels(obj, labels=[label])
        density_matrix_dict[label] = get_density_matrix_from_an_obj(one_obj, y_len=y_len, x_len=x_len, target=target)
    return density_matrix_dict
        

def _get_llikelihood_between_density_matrix_and_one_dealed_signal(
    density_matrix: np.ndarray,
    one_signal: np.ndarray,
    eps: Optional[float] = 1e-5,
):
    llikelihood = _get_llikelihood_between_density_matrix_and_case_signal(density_matrix=density_matrix, case_signal=one_signal, eps=eps, axis=0)
    return llikelihood


def get_llikelihood_between_reads_of_an_obj_to_density_matrix(
    obj: dict,
    density_matrix: np.ndarray,
    y_len: int = 1000,
    x_len: int = 1000,
    eps: float = 1e-5,
    target: Literal['dna1', 'window', 'dna2'] = 'window',
) -> pd.DataFrame:
    all_read_ids = list(obj.keys())
    X = pp.get_signals_for_reads_in_an_obj(obj, all_read_ids, scale_to=x_len, normalize_by_openpore=True, target=target)
    X = np.array(X, dtype=np.float32)
    X = y_len - 1 - np.round(X*y_len).astype(np.int16)
    X[X<0] = 0
    llikelihood = _get_llikelihood_between_density_matrix_and_dealed_signals(density_matrix=density_matrix, signals=X, eps=eps)
    llikelihood_df = pd.DataFrame({'llikelihood': llikelihood}, index=all_read_ids)
    return llikelihood_df


def get_llikelihood_between_reads_of_an_obj_to_density_matrix_dict(
    obj: dict,
    density_matrix_dict: dict, # key: label value: density matrix
    y_len: int = 1000,
    x_len: int = 1000,
    eps: float = 1e-5,
    target: Literal['dna1', 'window', 'dna2'] = 'window',
) -> pd.DataFrame:
    labels = list(density_matrix_dict.keys())
    all_dfs = []
    for label in labels:
        one_llikelihod_df = get_llikelihood_between_reads_of_an_obj_to_density_matrix(obj, density_matrix=density_matrix_dict[label],
                                                                                      y_len=y_len, x_len=x_len, eps=eps,
                                                                                      target=target)
        all_dfs.append(one_llikelihod_df)
    df = pd.concat(all_dfs, axis=1)
    df.columns = labels
    df['read_label'] = [obj[read_id]['label'] for read_id in list(obj.keys())]
    return df
    



def _get_llikelihood_between_density_matrix_and_dealed_signals(
    density_matrix: np.ndarray,
    signals: np.ndarray,
    eps: Optional[float] = 1e-5,
):
    llikelihood = _get_llikelihood_between_density_matrix_and_case_signal(density_matrix=density_matrix, case_signal=signals, eps=eps, axis=1)
    return llikelihood   


def _get_llikelihood_between_density_matrix_and_case_signal(
    density_matrix: np.ndarray,
    case_signal: np.ndarray,
    axis: Literal[0, 1],
    eps: Optional[float] = 1e-5,
):
    dens = density_matrix[case_signal,range(density_matrix.shape[1])]
    dens[dens==0] = eps
    llikelihood = np.sum(np.log(dens), axis=axis)
    return llikelihood



def find_density_cutoff_for_a_llikelihood_df_for_one_label(
    llikelihood_df: pd.DataFrame, 
    sd_fold: float = 1.0, 
    figsize: Tuple[float, float] = (5,4), 
    save_figure: bool = False, 
    density_figure_prefix: str = None,
    label: str = None,
) -> float:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.kdeplot(llikelihood_df, ax=ax)
    xs, ys = ax.lines[-1].get_data()
    ymax_index = np.argmax(ys)
    xmax = xs[ymax_index]

    data = llikelihood_df['llikelihood']
    norm_array = np.concatenate([
        2*xmax - data[data > xmax],
        data[data >= xmax]
    ])

    cutoff = xmax-norm_array.std()*sd_fold
    if cutoff < data.min():
        cutoff = data.min()
    
    ax.vlines(cutoff, 0, ys[ymax_index], ls='--', color='red')

    if save_figure and density_figure_prefix:
        figure_name = density_figure_prefix + "." + label + '.pdf'
        plt.savefig(figure_name)
    
    return cutoff



def find_density_cutoffs_for_each_label_in_an_obj(
    obj: dict,
    labels: list,
    density_matrix_dict: dict,
    sd_fold: float = 1.0, 
    y_len: int = 1000,
    x_len: int = 1000,
    eps: float = 1e-5,
    density_figure_prefix: str = None
):
    cutoffs = []
    for label in labels:
        one_obj = pp.extract_reads_with_labels(obj, [label])
        llikelihood_df = get_llikelihood_between_reads_of_an_obj_to_density_matrix(one_obj, density_matrix_dict[label], y_len=y_len, x_len=x_len, eps=eps)
        cutoff = find_density_cutoff_for_a_llikelihood_df_for_one_label(llikelihood_df=llikelihood_df, sd_fold=sd_fold, density_figure_prefix=density_figure_prefix, label=label, save_figure=True)
        cutoffs.append(cutoff)
    return dict(zip(labels, cutoffs))



def get_density_matirx_and_cutoff_for_each_label_in_two_objs(
    obj_3d: dict,
    obj_valid: dict,
    labels: list,
    sd_fold: float = 1.0, 
    y_len: int = 1000,
    x_len: int = 1000,
    eps: float = 1e-5,
    density_figure_prefix: str = None
) -> Tuple[dict, dict]:
    """get density matrix from 3d filtered obj and find cutoff by raw obj which is the one that filtered out 
    invalid windows or windows shorter than 1000, with current higher than I0 lower than 0.

    Args:
        obj_3d (dict): _description_
        obj_valid (dict): _description_
        labels (list): _description_
        sd_fold (float, optional): _description_. Defaults to 1.0.
        y_len (int, optional): _description_. Defaults to 1000.
        x_len (int, optional): _description_. Defaults to 1000.
        eps (float, optional): _description_. Defaults to 1e-5.
        density_figure_prefix (str, optional): _description_. Defaults to None.

    Returns:
        Tuple[dict, dict]: density_matrix_dict, cutoff_dict
    """    
    density_matrix_dict = get_density_matrix_for_each_label_from_an_obj(obj=obj_3d, labels=labels, y_len=y_len, x_len=x_len)
    cutoff_dict = find_density_cutoffs_for_each_label_in_an_obj(obj_valid, labels=labels, density_matrix_dict=density_matrix_dict, sd_fold=sd_fold,
                                                                y_len=y_len, x_len=x_len, eps=eps, density_figure_prefix=density_figure_prefix)
    return density_matrix_dict, cutoff_dict


# def filter_reads_by_density(
#     obj_for_density: dict,
#     obj_need_to_clean: dict,
#     sd_fold: float = 1.0, 
#     y_len: int = 1000,
#     x_len: int = 1000,
#     eps: float = 1e-5,
#     density_figure_prefix: str = None,
# ) -> dict:
#     """filter reads by density. Consider each label seperatelly.

#     Args:
#         obj_for_density (dict): the obj for calculating density matrix and finding the cutoff for each label
#         obj_need_to_clean (dict): the obj needs to be cleaned
#         sd_fold (float, optional): the sd fold of the constructed normal distribution. Defaults to 1.0.
#         y_len (int, optional): the height of density matrix. the first and last row indicate I0 and zero respectivelly. Defaults to 1000.
#         x_len (int, optional): the width of densith matrix indicating signal length. Defaults to 1000.
#         eps (float, optional): eps for log zero. Defaults to 1e-5.
#         density_figure_prefix (str, optional): file prefix of density plot. If it is not none, will save density figures with cutoff lines.

#     Returns:
#         dict: clean obj
#     """    
#     labels = np.unique(np.array([read_obj['label'] for read_id, read_obj in obj_for_density.items()]))
#     density_matrix_dict, cutoff_dict = get_density_matrix_and_cutoff_for_each_label_in_an_obj(obj=obj_for_density,
#                                                                                               labels=labels,
#                                                                                               sd_fold=sd_fold,
#                                                                                               y_len=y_len,
#                                                                                               x_len=x_len,
#                                                                                               eps=eps,
#                                                                                               density_figure_prefix=density_figure_prefix)
#     all_invalid_read_ids = []
#     for label in labels:
#         one_obj = pp.extract_reads_with_labels(obj_need_to_clean, [label])
#         llikelihood_df = get_llikelihood_between_reads_of_an_obj_to_density_matrix(one_obj, density_matrix_dict[label], y_len=y_len, x_len=x_len, eps=eps)
#         invalid_read_ids = llikelihood_df[llikelihood_df['llikelihood']<cutoff_dict[label]].index.to_list()
#         all_invalid_read_ids.extend(invalid_read_ids)
#     all_invalid_read_ids = np.unique(np.array(all_invalid_read_ids))

#     obj = tl.delete_reads_in_an_obj(obj_dict=obj_need_to_clean, read_ids_to_be_removed=all_invalid_read_ids, in_place=False)
#     return obj
    


def cal_distance_between_any_two_density_matrixs_in_a_density_matrix_dict(
    density_matrix_dict,
    metric: str = 'euclidean',
) -> pd.DataFrame:
    one_dim_matrix_dict = {}
    labels = []
    for label, matrix in density_matrix_dict.items():
        one_dim_matrix_dict[label] = matrix.flatten()
        labels.append(label)
    X = np.concatenate(list(one_dim_matrix_dict.values())).reshape(len(density_matrix_dict),-1)
    Y = pdist(X, metric=metric)
    distance_matrix = squareform(Y)
    distance_matrix_df = pd.DataFrame(distance_matrix, columns=labels, index=labels)
    return distance_matrix_df
        
