from skimage.restoration import denoise_wavelet
import numpy as np

def denoise_by_wavelet(
    array: np.array,
    sigma: float = 20,
):
    array = denoise_wavelet(array, sigma=sigma)
    return array


def denoise_window_signals_for_one_obj_by_wavelet(
    obj_dict,
    sigma: float = 20,
):
    """Denoise signals in window by wavelet

    Args:
        obj_dict (_type_): obj
        sigma (float, optional): sigma for wavelet. Defaults to 20.
    """
    for read_id, read_obj in obj_dict.items():
        read_obj['signal'] = denoise_by_wavelet(read_obj['signal'], sigma=sigma)


def smooth_by_conv(
    array: np.array,
    v_len: int = 50,
):
    return np.convolve(a=array, v=np.ones(v_len)/v_len, mode='same')


def denoise_window_signals_for_one_obj_by_conv(
    obj_dict,
    v_len: int = 50,
):
    """Denoise signals in window by np.convolve

    Args:
        obj_dict (_type_): obj
        v_len (int, optional): v len. Defaults to 50.
    """
    for read_id, read_obj in obj_dict.items():
        read_obj['signal'] = smooth_by_conv(read_obj['signal'], v_len=v_len)