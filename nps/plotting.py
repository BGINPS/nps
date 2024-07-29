import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
import os
import re
from . import tools as tl
from . import preprocessing as pp



def draw_signal(signal, ax, color: str = '#99AB5F'):
    if ax:
        sns.lineplot(x=range(len(signal)),y=signal,ax=ax, color=color)
    else:
        ax = sns.lineplot(x=range(len(signal)),y=signal, color=color)
    ax.set_xlabel('time (1/5000 s)')
    ax.set_ylabel('current')


def draw_reads_randomly_for_one_label_in_obj(
    obj: dict,
    target_label: str,
    out_dir: str,
    read_num: int = 100,
    seed: int = 0,
):
    obj = pp.extract_reads_with_labels(obj, labels=target_label)
    draw_reads_randomly(obj=obj, read_num=read_num,
                        seed=seed, out_dir=out_dir)

def draw_reads_randomly(
    obj: dict,
    read_num: int = 100,
    seed: int = 0,
    out_dir: str = './out_figures',
):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    obj = tl.random_extract_reads(obj, extract_num=read_num, seed=seed)
    for read_id, read_obj in obj.items():
        draw_one_read(one_read=read_obj, save_figure=True, figure_name=f'{out_dir}/{read_id}.signal.pdf', title=read_id)


def draw_density_of_one_att_in_an_obj(
    obj: dict,
    att: str,
    figsize: Tuple[float, float] = (5,5), 
    ax = None,
    xlabel: str = None,
    color: str = '#99AB5F'
):
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)
    x = [read_obj[att] for read_id, read_obj in obj.items()]
    sns.kdeplot(x, ax=ax, color=color)
    if xlabel == None:
        xlabel = att
    ax.set_xlabel(xlabel)
    return ax



def draw_3d_for_atts(
    obj: dict,
    ax = None,
    color: str = '#99AB5F'
):
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    x = [read_obj['pd2rd'] for read_id, read_obj in obj.items()]
    y = [read_obj['window_i2i0_mean'] for read_id, read_obj in obj.items()]
    z = [read_obj['window_i2i0_std'] for read_id, read_obj in obj.items()]
    ax.scatter(x, y, z, marker='o', s=1, color=color)
    ax.set_xlabel('PD/RD')
    ax.set_ylabel('I/I0 mean')
    ax.set_zlabel('I/I0 std')
    return ax

def draw_combine_plot_of_atts_of_one_obj(
    raw_obj: dict,
    clean_obj: dict,
    representative_read_id: str,
    figure_name: str, 
    figsize: Tuple[float, float] = (15, 8),
    xlim_3d: Tuple[float, float] = (0, 0.6),
    ylim_3d: Tuple[float, float] = (0.1, 0.6),
    zlim_3d: Tuple[float, float] = (0, 0.2),

):
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4,6)

    ax1 = plt.subplot(gs[0,:])
    draw_one_read(raw_obj[representative_read_id], ax=ax1)

    ax2_1 = plt.subplot(gs[1,0:2])
    draw_density_of_one_att_in_an_obj(raw_obj, 'pd2rd', xlabel='PD/RD', ax=ax2_1)

    ax2_2 = plt.subplot(gs[1,2:4])
    draw_density_of_one_att_in_an_obj(raw_obj, 'window_i2i0_mean', xlabel="I/I0 mean", ax=ax2_2)

    ax2_3 = plt.subplot(gs[1,4:6])
    draw_density_of_one_att_in_an_obj(raw_obj, 'window_i2i0_std', xlabel="I/I0 std", ax=ax2_3)

    
    ax3_1 = plt.subplot(gs[2:4,0:3], projection='3d')
    draw_3d_for_atts(raw_obj, ax=ax3_1)
    ax3_1.set_xlim(xlim_3d)
    ax3_1.set_ylim(ylim_3d)
    ax3_1.set_zlim(zlim_3d)

    ax3_2 = plt.subplot(gs[2:4,3:6], projection='3d')
    draw_3d_for_atts(clean_obj, ax=ax3_2)
    ax3_2.set_xlim(xlim_3d)
    ax3_2.set_ylim(ylim_3d)
    ax3_2.set_zlim(zlim_3d)


    plt.tight_layout()
    plt.savefig(figure_name, bbox_inches='tight')






def draw_combine_plot_of_atts_of_one_label_in_obj(
    raw_obj: dict,
    clean_obj: dict,
    target_label: str,
    representative_read_id: str,
    figure_name: str, 
    figsize: Tuple[float, float] = (10,10),
    **kwargs
):
    raw_obj = pp.extract_reads_with_labels(raw_obj, labels=target_label)
    clean_obj = pp.extract_reads_with_labels(clean_obj, labels=target_label)
    draw_combine_plot_of_atts_of_one_obj(raw_obj=raw_obj, clean_obj=clean_obj, representative_read_id=representative_read_id,
                                         figure_name=figure_name, figsize=figsize, **kwargs)



def draw_one_read(
    one_read: dict, 
    save_figure: bool = False, 
    figure_name = None, 
    figsize: Tuple[float, float] = (15,8), 
    title = None, 
    ax = None,
    only_window: bool = False
):
    """Draw the sinnal of one read. If this read contains valid window, also draw out the window boundary.

    Args:
        one_read (dict): one read
        save_figure (bool, optional): Whether to save figure. Defaults to False.
        figure_name (_type_, optional): file name for saving figure. Defaults to None.
        figsize (Tuple, optional): figure size. Defaults to (15,8).
        title (_type_, optional): figure title. Defaults to None.
        ax (_type_, optional): axes. Defaults to None.
    """
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)
    
    x = one_read['signal']

    if one_read['window']:
        (start, end) = one_read['window']
        if only_window:
            x = x[start:end]
        if not only_window:
            draw_window(one_read, ax)

    draw_signal(x, ax=ax)
    
    ax.set_title(title)
        
    if save_figure:
        plt.savefig(figure_name)



def draw_window(
    one_read, 
    ax,
    color: str = 'red',
):
    if isinstance(one_read['window'], tuple):
        window_start, window_end = one_read['window']
        # ax.axvline(window_start, color='red', ls='--')
        # ax.axvline(window_end, color='red', ls='--')
        ax.plot([window_start, window_end], [one_read['signal'][window_start], one_read['signal'][window_end]], 'x',
                color=color)




def draw_clustermap_for_dtw_dis(
    ds_df: pd.DataFrame,
    color_by_prefix: bool = False,
):
    """Draw clustermap for dtw distance matrix

    Args:
        ds_df (pd.DataFrame): dtw distance matrix
        color_by_prefix (bool): whether to set row colors by prefix of each element.
    """
    X = np.array(ds_df)
    X = X[np.triu_indices(len(X), 1)]
    linkage_matrix = sch.average(X)

    row_colors = None
    if color_by_prefix:
        sample_name = pd.Series([re.sub('_.*', '', i) for i in ds_df.index], index=ds_df.index)
        lut = dict(zip(sample_name.unique(), "rb"))
        row_colors = sample_name.map(lut)
        
    sns.clustermap(ds_df, row_linkage=linkage_matrix, col_linkage=linkage_matrix, row_colors=row_colors)





def draw_density_heatmap(
    data: np.ndarray,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (5,4), 
    save_figure: bool = False, 
    figure_name = None,
    **kwargs,
):
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(data, cmap='YlGnBu', yticklabels=False, ax=ax, **kwargs)

    if save_figure:
        plt.savefig(figure_name)
    


def draw_stack_plot(obj, read_num: int = 3000, color: str = 'black', ax = None, alpha: float = 0.007):
    obj = tl.random_extract_reads(obj, extract_num=read_num)
    all_read_ids = list(obj.keys())
    window_sigs = pp.get_signals_for_reads_in_an_obj(obj_dict=obj, all_read_ids=all_read_ids)
    if ax == None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim((0, len(window_sigs[0])))
    ax.set_ylim(0,1)
    ax.set_ylabel('I/I0')
    ax.set_xlabel('scaled time')
    for i, sig in enumerate(window_sigs):
        ax.plot(sig, color=color, linewidth=1, alpha=alpha)
    return ax