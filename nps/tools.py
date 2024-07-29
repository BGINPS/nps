import numpy as np
import pandas as pd
import copy
import random
from typing import Dict, Optional, Tuple, Union
from . import preprocessing as pp
from . import plotting as pl
from . import io
import re
import glob


def random_extract_reads(
    obj_dict, 
    extract_num: int, 
    validated_read: bool = True, 
    seed: int = 1
) -> dict:
    """Extract reads randomly from an obj.

    Args:
        obj_dict (_type_): obj
        extract_num (int): the number of reads would be selected
        validated_read (bool, optional): Whether to only select reads with windows. Defaults to True.
        seed (int, optional): random seed. Defaults to 1.

    Returns:
        dict: _description_
    """
    obj_dict = copy.deepcopy(obj_dict)

    if validated_read:
        pp.filter_out_reads_without_widows(obj_dict)

    read_ids = [read_id for read_id, read_obj in obj_dict.items()]
    random.seed(seed)
    random.shuffle(read_ids)
    
    selected_dict = {}
    for read_id in read_ids[0:extract_num]:
        selected_dict[read_id] = obj_dict[read_id]
    return selected_dict


def get_window_start_end_for_one_read(
    one_read,
) -> Tuple[int, int]:
    """Get window start and end for one read.

    Args:
        one_read (_type_): one read, dict

    Returns:
        Tuple[int, int]: start, end. Zero based.
    """
    window_start, window_end = None, None
    if isinstance(one_read['window'], tuple):
        window_start, window_end = one_read['window']
    return window_start, window_end


def extract_signal_in_window_for_one_read_obj(
    read_obj,    
) -> np.array:
    win_start, win_end = get_window_start_end_for_one_read(read_obj)
    signal_in_window = read_obj['signal'][win_start:win_end+1]
    return signal_in_window


def extract_signals_in_windows_for_one_obj(
    obj_dict,
) -> dict:
    """Extract signals in windows for one obj. Return a dict with keys as read_ids and values as signals in windows.

    Args:
        ojb_dict (_type_): obj

    Returns:
        dict: keys as read_ids and values as signals in windows
    """
    signals_in_windows = {}
    for read_ids, read_obj in obj_dict.items():
        signal_in_window = extract_signal_in_window_for_one_read_obj(read_obj)
        signals_in_windows[read_ids] = signal_in_window
    return signals_in_windows


def colloct_info_for_one_exp(one_obj, func):
    infos = []
    for read_id, read_obj in one_obj.items():
        infos.append(func(read_obj))
    infos = pd.Series(infos, index=one_obj.keys()).dropna()
    return infos

def total_length_for_one_read(read_obj):
    signal = read_obj['signal']
    return len(signal)

def total_length_for_one_obj(one_obj):
    signal_lengths = colloct_info_for_one_exp(one_obj, total_length_for_one_read)
    return signal_lengths


def window_length_for_one_read(read_obj):
    win_start, win_end = get_window_start_end_for_one_read(read_obj)
    return win_end - win_start + 1

def window_length_for_one_obj(
    one_obj
) -> pd.Series:
    signal_lengths = colloct_info_for_one_exp(one_obj, total_length_for_one_read)
    return signal_lengths



def combine_objs_with_a_prefix_in_a_dir(
    in_dir: str,
    prefix: str,
) -> dict:
    """combine all pkl files with a specific prefix in a dir.

    Args:
        in_dir (str): the dir than contains pkl files
        prefix (str): the prefix of pkl files that would be combined. all pkl files with `prefix`_sample\d+ would be combined.

    Returns:
        dict: obj
    """
    all_pkl_files = glob.glob(f'{in_dir}/{prefix}_sample*')
    reg = re.compile(r'(' + prefix + '_sample\d+)')
    all_sample_names = [re.search(reg, i).group(1) for i in all_pkl_files]
    all_labels = [prefix for i in all_sample_names]
    all_objs = [io.read_pickle(i) for i in all_pkl_files]
    obj = combine_objs(obj_list=all_objs, sample_name_list=all_sample_names, label_list=all_labels)
    return obj


def combine_objs(
    obj_list: list, 
    sample_name_list: list = [],
    label_list: list = [],
    get_obj_stat: bool = True,
) -> dict:
    """Combine several objs into one by adding different name to reads in each obj

    Args:
        obj_list (list): a list of obj
        sample_name_list (list): a list of sample name, will be added as prefix of read ids with "_" as linking
        label_list: (list): a list of label which would be used as y in model input, such as pepetide name.
                            Would be add to 'label' key in reads.  

    Returns:
        dict: _description_
    """
    sample_name_list = copy.deepcopy(sample_name_list)
    label_list = copy.deepcopy(label_list)
    sample_name_list = np.array(sample_name_list)
    # assert len(sample_name_list) == len(np.unique(sample_name_list))
    if len(sample_name_list) > 0:
        assert len(sample_name_list) == len(obj_list)
    if label_list:
        assert len(label_list) == len(obj_list)
        pp.add_label(objs=obj_list, labels=label_list)
    
    for indx in range(len(obj_list)):
        if len(sample_name_list) > 0:
            obj_list[indx] = {sample_name_list[indx] + "_" + k: v for k, v in obj_list[indx].items()}

    com_obj = copy.deepcopy(obj_list[0])

    for indx in range(1, len(obj_list)):
        com_obj.update(obj_list[indx])
        # for read_id, read_obj in obj_list[indx].items():
        #     com_obj[read_id] = read_obj
    if get_obj_stat:
        state_obj(obj=com_obj)
    
    return com_obj


def down_sampling(
    array: np.array, 
    down_sample_to: int = 1000
) -> np.array:
    total_length = len(array)
    
    if total_length < down_sample_to:
        return np.concatenate((array,[0 for i in range(down_sample_to-total_length)]))
    
    sample_idx = np.round(np.linspace(start=0, stop=total_length-1, num=down_sample_to)).astype(np.int16)
    return array[sample_idx]


def delete_reads_in_an_obj(
    obj_dict,
    read_ids_to_be_removed: list,
    in_place: bool = False
) -> dict:
    if not in_place:
        obj_dict = copy.deepcopy(obj_dict)
    for read_id in read_ids_to_be_removed:
        del obj_dict[read_id]
    if not in_place:
        return obj_dict
    else:
        return None


def state_obj(
    obj,
    des: str = None,
    return_sample_num_for_classes: bool = False
):
    """Print out total read number of an obj and read numbers of all classes if 'label' in obj

    Args:
        obj (_type_): _description_
        des (str): _description_
    """
    if des:
        print(des + "(read number):", end='\t')

    total_reads_num = len(obj)
    if des:
        print(f'{total_reads_num}')
    else:
        print(f'read number: {total_reads_num}')

    if 'label' in list(obj.values())[0]:
        df = pd.DataFrame([[read_id, read_obj['label']] for read_id, read_obj in obj.items()], columns=['read_id', 'label'])
        print(df['label'].value_counts())
        if return_sample_num_for_classes:
            return df['label'].value_counts()


def combine_cutoffs_dicts(
    cutoffs_dict_list: list,
):
    all_dict = cutoffs_dict_list[0].copy()
    for one_dict in cutoffs_dict_list:
        all_dict.update(one_dict)
    cutoffs_df = pd.DataFrame.from_dict(all_dict, orient='index')

    return cutoffs_df


def get_peak_of_one_att_in_an_obj_by_density_plot(
    obj: dict,
    att: str,
):
    ax = pl.draw_density_of_one_att_in_an_obj(obj=obj, att=att)
    xs, ys = ax.get_lines()[0].get_xdata(), ax.get_lines()[0].get_ydata()
    peak_on_x = xs[np.argmax(ys)]
    return peak_on_x


def get_summary_stat_of_an_obj(
    obj: dict,
):
    pd2rd_mean, i2i0_mean_mean, i2i0_std_mean = np.array([[read_obj['pd2rd'], read_obj['window_i2i0_mean'], read_obj['window_i2i0_std']] for read_id, read_obj in obj.items()]).mean(axis=0)
    pd2rd_peak = get_peak_of_one_att_in_an_obj_by_density_plot(obj, 'pd2rd')
    i2i0_mean_peak = get_peak_of_one_att_in_an_obj_by_density_plot(obj, 'window_i2i0_mean')
    i2i0_std_peak = get_peak_of_one_att_in_an_obj_by_density_plot(obj, 'window_i2i0_std')

    df = pd.DataFrame({'pd2rd_mean': [pd2rd_mean], 'i2i0_mean_mean': [i2i0_mean_mean], 'i2i0_std_mean': [i2i0_std_mean],
                  'pd2rd_peak': [pd2rd_peak], 'i2i0_mean_peak': [i2i0_mean_peak], 'i2i0_std_peak': [i2i0_std_peak]})
    return df



def cal_read_nums_of_all_channels(
    obj: dict,
) -> pd.DataFrame:
    channels = []
    for read_id, read_obj in obj.items():
        channel = get_channel_from_one_read_id(read_id=read_id)
        channels.append(channel)
    channels = np.array(channels)
    channel_count_df = pd.DataFrame(channels).value_counts().to_frame().reset_index()
    channel_count_df.columns = ['channel', 'count']
    return channel_count_df


def get_reads_with_a_specific_channel_from_obj(
    obj: dict,
    channel_id: str,
) -> dict:
    sub_dict = {}
    for read_id, read_obj in obj.items():
        if get_channel_from_one_read_id(read_id) == channel_id:
            sub_dict[read_id] = read_obj
    return sub_dict



def get_channel_from_one_read_id(
    read_id: str,
) -> str:
    tmp = re.search(r'(channel\S+?)_', read_id)
    if tmp:
        return(tmp.group(1))
    else:
        return('channel')