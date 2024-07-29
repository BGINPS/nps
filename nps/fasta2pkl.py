import os
import pickle
import shutil
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
from . import io
import glob
import re

OPP_DICT = dict()

pool_num = 5

class Trend:
    RISE = 0
    FALL = 1
    OTHER = 2


class State:
    START = 0
    PEAK = 1
    PEPTIDE = 2


def get_trend(pair, threshold):
    left, right = pair
    if left < threshold <= right:
        return Trend.RISE
    elif left >= threshold > right:
        return Trend.FALL
    else:
        return Trend.OTHER


def find_peak(signal, y_cut):
    min_peak_len = 15
    min_peptide_len = 200
    current_state = State.START
    for i in range(1, len(signal)):
        trend = get_trend(signal[i-1:i+1], y_cut)
        if trend == Trend.RISE and np.all(signal[i:i+min_peak_len] >= y_cut):
            if current_state == State.START:
                current_state = State.PEAK
        elif trend == Trend.FALL and np.all(signal[i:i+min_peptide_len] < y_cut):
            if current_state == State.PEAK:
                return i
    return None


def do_slice(signal, openpore=220.0, offset=0.4698):
    y_cut = openpore * offset
    start_trim_len = round(0.3 * len(signal))
    end_trim_len = round(0.075 * len(signal))
    signal_trimmed = signal[start_trim_len:-end_trim_len]

    left = find_peak(signal_trimmed, y_cut)
    if left is None:
        return None
    else:
        left += start_trim_len

    right = find_peak(signal_trimmed[::-1], y_cut)
    if right is None:
        return None
    else:
        right = len(signal) - end_trim_len - right

    #slice_length = slice_start(signal[left:right][::-1], y_cut - 20) # 2024/01/08 zhuqh
    #right -= slice_length

    if left + 600 >= right:
        return None
    else:
        return left, right


def slice_start(signal, y_cut):
    # plot_x_y(signal, axis_off=False, hline=(y_cut,))
    min_peptide_len = 50
    for i in range(1, len(signal)):
        trend = get_trend(signal[i-1:i+1], y_cut)
        if trend == Trend.FALL and np.all(signal[i:i+min_peptide_len] < y_cut):
            return i
    return 0


def get_reads_dict_from_fast5(fast5_path, retry=2):
    while retry > 0:
        try:
            reads_dict = dict()
            with h5py.File(fast5_path, 'r') as f:
                if fast5_path.endswith('fast5'):
                    for read_id, val in f['Raw']['Reads'].items():
                        if val is not None:
                            fast5_fn = os.path.basename(fast5_path).replace('.fast5', '')
                            ch_read_id = f'{fast5_fn}_Read_{read_id[5:]}' if fast5_fn.startswith('channel') else read_id
                            signal = np.array(f['Raw']['Reads'][read_id]['Signal'])
                            openpore = np.array(f['Raw']['Reads'][read_id]['Openpore_mean']).item()
                            reads_dict[ch_read_id] = {'window': None, 'confidence': 0, 'signal': signal,
                                                      'openpore': openpore}
                elif fast5_path.endswith('ccf'):
                    for read_id, val in f.items():
                        if val is not None:
                            signal = np.array(f[read_id]['Raw']['Signal'], dtype=np.float16)
                            ccf_fn = os.path.basename(fast5_path).split('.')[0]
                            if ccf_fn.startswith('channel'):
                                ch = int(ccf_fn.replace('channel', ''))
                            else:
                                ch = [token for token in read_id.split('_') if token.startswith('channel')][0]
                                ch = int(ch.replace('channel', ''))
                            start_idx = int(read_id.split('_')[-1].split('-')[0])
                            openpore = OPP_DICT[(ch, start_idx)]
                            if 'channel' in read_id:
                                if read_id.startswith('read_'):
                                    ch_read_id = read_id[5:]
                                else:
                                    ch_read_id = read_id
                            else:
                                ch_read_id = f'channel{ch}_{read_id[5:]}'
                            reads_dict[ch_read_id] = {'window': None, 'confidence': 0, 'signal': signal,
                                                      'openpore': openpore}
                else:
                    raise Exception(f'Invalid file extension: {fast5_path}')
                f.close()
            if fast5_path.startswith('/tmp'):
                os.remove(fast5_path)
            return reads_dict
        except Exception as e:
            retry -= 1
            if retry == 1:
                shutil.copy(fast5_path, os.path.join('/tmp', os.path.basename(fast5_path)))
                fast5_path = os.path.join('/tmp', os.path.basename(fast5_path))
            print(f'[Warning] Failed reading reads file: {fast5_path}. Retry({retry})')
    return None


def load_reads_csv(data_name):
    global OPP_DICT
    OPP_DICT = dict()
    read_csv_path = f'{data_path}/{data_name}/data_for_analysing/read.csv'
    df = pd.read_csv(read_csv_path)
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        fn = int(row_dict['abf_file_name'])
        idx = int(row_dict['start'])
        OPP_DICT[(fn, idx)] = float(row_dict['openpore_value'])


def get_reads_dict(data_dir):
    fast5_dir = f'{data_dir}/results/Meta/'
    fast5_paths = [os.path.join(fast5_dir, fn) for fn in os.listdir(fast5_dir) if fn.endswith('.fast5') or fn.endswith('.ccf')]
    # if len(fast5_paths) > 1 and fast5_paths[0].endswith('.ccf'):
    #     load_reads_csv(data_name)
    with Pool(min(len(fast5_paths), pool_num)) as p:
        reads_dict_list = p.starmap(get_reads_dict_from_fast5, zip(fast5_paths))
    result = dict()
    for d in reads_dict_list:
        result.update(d)
    return result


def slice_peptide(read_dict, process_all_reads):
    keys = read_dict.keys()
    signal_openpore = [(v['signal'], v['openpore']) for v in read_dict.values()]
    with Pool(pool_num) as p:
        windows = p.starmap(do_slice, signal_openpore)
    sliced_count = 0
    for key, window in zip(keys, windows):
        if window is None:
            if not process_all_reads:
                del read_dict[key]
        else:
            sliced_count += 1
            read_dict[key]['window'] = window
            read_dict[key]['confidence'] = 1
    return sliced_count


def get_pkl_data(data_dir, process_all_reads=True, pkl_save_path=None):
    if os.path.isfile(pkl_save_path):
        print(f'[INFO] Already exist: {pkl_save_path}')
        return None
        
    print(f'[INFO] Extracting reads data from {data_dir}...')
    read_dict = get_reads_dict(data_dir)
    read_dict = {read_id: v for read_id, v in read_dict.items() if len(v['signal']) < 30000}
    if len(read_dict) > 0:
        print(f'[INFO] {len(read_dict)} reads loaded, start slicing...')
        sliced_count = slice_peptide(read_dict, process_all_reads)
        print(f'[INFO] {sliced_count} OPO detected.')
    
    if pkl_save_path:
        io.save_pickle(read_dict, pkl_save_path)
        return None
    return read_dict


def load_data(seg_data_path):
    with open(seg_data_path, 'rb') as f:
        return pickle.load(f)



def fast5s_in_a_dir_to_pkls(
    in_dir: str, 
    out_dir: str, 
    label: str
):
    """`in_dir` contains one or more sub-dirs that named as 20*_Mux (eg: 20231222130109_.*_Mux). This function
    convert each sub-dir as one pkl file, with windows be identified. Each pkl file name
    would be `label`_sample20* (eg: `label`_sample20231222130109.pkl)

    Args:
        in_dir (str): the dir contains one or more 20*_Mux dirs
        out_dir (str): the dir for saving pkl files
        label (str): class label
    """
    sample_dirs = glob.glob(f'{in_dir}/20*_Mux')
    sample_names = _get_sample_names_from_sample_paths(sample_dirs)
    pkl_basenames = [f'{label}_sample{sample_name}' for sample_name in sample_names]
    pkl_file_paths = [f'{out_dir}/{pkl_basename}.pkl' for pkl_basename in pkl_basenames]
    for one_sample_dir, one_pkl_file_path in zip(sample_dirs, pkl_file_paths):
        get_pkl_data(data_dir=one_sample_dir, pkl_save_path=one_pkl_file_path)

def _get_sample_names_from_sample_paths(sample_paths):
    sample_names = []
    for sample_path in sample_paths:
        sample_name = re.search(r'(20\d+)_', os.path.basename(sample_path)).group(1)
        sample_names.append(sample_name)
    return sample_names
