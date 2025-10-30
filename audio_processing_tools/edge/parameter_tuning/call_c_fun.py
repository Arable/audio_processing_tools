import subprocess as subps
from ctypes import *
from typing import Any
from typing import Union as tUnion
import os, sys
import sys
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import librosa
import csv
from collections import Counter
import math
import copy
import binascii



class evmgr_sensor_data_t(Structure):
    _fields_ = [("sensor_id", c_uint8),
                ("len", c_uint8),
                ("reserved", c_uint16),
                ("buf", POINTER(c_float))]

class evmgr_data_input_t(Structure):
    _fields_ = [('audio_len', c_int),
                ('raw_audiop', c_char_p),
                ('image_len', c_int), ('imagep', c_char_p),
                ('sensor_data', evmgr_sensor_data_t)]

FREQ_BAND=6
class rain_cl_optional_data_t(Structure):
    _pack_ = 1
    _fields_ = [("len", c_uint16),
                ("version", c_uint32),
                ("timestamp", c_uint32),
                ("raindrops", c_uint32),
                ("mean_freq", c_float*FREQ_BAND),
                ("rain_threshold", c_float*FREQ_BAND),
                ("buf", c_uint8*2)]

class rain_cl_config_param_t(Structure):
    _pack_ = 1
    _fields_ = [("sample_rate", c_uint32),
                ("freq_resolution", c_uint16),
                ("time_resolution_ms", c_uint16),
                ("check_duration", c_float),
                ("op_freq_range", c_uint16*2),
                ("n_freq_range", c_uint16*2),
                ("harmonic_threshold", c_float*FREQ_BAND),
                ("fn", c_uint16),
                ("num_harmonics", c_uint16),
                ("max_peaks", c_uint16),
                ("log_factor", c_uint16),
                ("ns_duration_ms", c_uint16),
                ("nf", c_float),
                ("min_drop_count", c_float)]


def conditional_print(enabled=False):
    """
    A wrapper function around print() that allows conditional printing based on the 'enabled' flag.

    Args:
        enabled (bool): If True, print the message; if False, do not print.

    Returns:
        None
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if enabled:
                return func(*args, **kwargs)
        return wrapper
    return decorator

@conditional_print(enabled=False)  # Set 'enabled' to False to disable printing
def print_log(*args, **kwargs):
    print(*args, **kwargs)


def write_results(csv_file_name, csv_columns, data):

    with open(csv_file_name, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def twos_complement_hex(num):
    # Assuming 16-bit integers, you can adjust the bit length if needed
    return (num + (1 << 16)) % (1 << 16)

def read_audio_file(audio_file, read_size, read_offset):
    """

    Parameters
    ----------
    audio_file - path to file OR signal contents
    read_size
    read_offset

    Returns
    -------

    """
    if type(audio_file) == str:
        if audio_file.lower().endswith(".wav"):
            audio_data_in, sr_ref = librosa.load(audio_file, sr=None)

        else:
            with open(audio_file, "rb") as file:
                # skip the audio file header
                file.seek(HEADER_SIZE),
                # read the file in to an ndarray
                audio_data_in = file.read()
                # scale it to float values between -1.0 to 1.0
                scale_factor = 1 << (bytes_per_sample * 8 - 1)
                scale_factor = 32767
                audio_data_in = (
                    np.frombuffer(audio_data_in, dtype=np.int16) / scale_factor
                )
    elif type(audio_file) == np.ndarray:
        audio_data_in = audio_file
    else:
        raise Exception(
            "Did not recognize 'audio_file' type. Must be path to WAV, audio binary, or numpy signal array"
        )

    audio_data_in = audio_data_in[read_offset : read_offset + read_size]

    return audio_data_in

# def classifier_boolean_wrapper(
#     audio_signal: np.ndarray, **kwargs: Any
# ) -> tUnion[bool, float]:
#     """
#     Evaluates whether an audio signal contains rain.
#
#     Parameters:
#     - audio_signal (np.ndarray): The audio signal to evaluate.
#     - threshold (float): The threshold value for determining rain presence.
#     - **kwargs (Any): Additional keyword arguments passed to the rain detection algorithm.
#
#     Returns:
#     - Union[bool, float]: Returns True if rain is detected above the threshold, False if below or equal to the threshold,
#       and np.nan if the rain drop count is negative.
#     """
#
#     rain_drop_count, _ = rain_detection_algo(audio_signal, **kwargs)
#     if rain_drop_count > 0:
#         return True
#     elif 0 <= rain_drop_count:
#         return False
#     else:
#         return np.nan

def rain_detection_algo (audio_data, **kwargs):

    default_params = {
        "sample_rate": 11162,
        "freq_resolution": 45,
        "time_resolution_ms": 10,
        "check_duration": 5,
        "op_freq_range": [375, 3000],
        "n_freq_range": [400, 600],
        "fn": 400,
        "num_harmonics": 6,
        "harmonic_threshold": [4.25, 4, 4, 4, 4, 4],
        "max_peaks": 3,
        "log_factor": 0,
        "ns_duration_ms": 470,
        "nf": 0,
        "min_drop_count": 1,
    }

    merged_params = {**default_params, **kwargs}

    c_code_directory = os.path.dirname(os.path.abspath(__file__))
    so_file = os.path.join(c_code_directory, "libdsp_shared_lib.dylib")

    dsp_functions = CDLL(so_file)
    print_log(type(dsp_functions))

    # scale_factor = 32767
    # audio_data_in_t1 = audio_data * scale_factor
    # audio_data_in = np.array(audio_data_in_t1, dtype=np.int16)
    audio_data_in = audio_data.copy()

    char_array = c_char * len(audio_data_in) * 2

    init_func = dsp_functions.sample_classifier_to_evaluate_impl
    init_func.argtypes = [POINTER(evmgr_data_input_t), POINTER(rain_cl_optional_data_t), POINTER(rain_cl_config_param_t)]
    init_func.restype = c_int

    inp_data_ptr = evmgr_data_input_t()
    inp_data_ptr.audio_len = len(audio_data_in) * 2
    inp_data_ptr.raw_audiop = cast(char_array.from_buffer(audio_data_in), c_char_p)

    out_opt_data_ptr = rain_cl_optional_data_t()

    inp_config_ptr = rain_cl_config_param_t()
    inp_config_ptr.sample_rate = merged_params['sample_rate']
    inp_config_ptr.freq_resolution = merged_params['freq_resolution']
    inp_config_ptr.time_resolution_ms = merged_params['time_resolution_ms']
    inp_config_ptr.check_duration = merged_params['check_duration']
    inp_config_ptr.fn = merged_params['fn']
    inp_config_ptr.op_freq_range[0] = merged_params['op_freq_range'][0]
    inp_config_ptr.op_freq_range[1] = merged_params['op_freq_range'][1]
    inp_config_ptr.n_freq_range[0] = merged_params['n_freq_range'][0]
    inp_config_ptr.n_freq_range[1] = merged_params['n_freq_range'][1]
    inp_config_ptr.harmonic_threshold[0] = merged_params['harmonic_threshold'][0]
    for i in range(1, 6):
        inp_config_ptr.harmonic_threshold[i] = merged_params['harmonic_threshold'][i]
    inp_config_ptr.num_harmonics = merged_params['num_harmonics']
    inp_config_ptr.max_peaks = merged_params['max_peaks']
    inp_config_ptr.log_factor = merged_params['log_factor']
    inp_config_ptr.ns_duration_ms = merged_params['ns_duration_ms']
    inp_config_ptr.nf = merged_params['nf']
    inp_config_ptr.min_drop_count = merged_params['min_drop_count']

    class Convert(Union):
        _fields_ = (("my_bytes", c_char * sizeof(c_float)),
                    ("my_float", c_float))

    conv = Convert()

    # Convert negative numbers to two's complement representation
    #converted_data = np.vectorize(twos_complement_hex)(audio_data_in)
    #np.savetxt('aud_in_data.txt', converted_data, fmt='%#x')



    rain_drop_count = dsp_functions.sample_classifier_to_evaluate_impl(
                        byref(inp_data_ptr), byref(out_opt_data_ptr), byref(inp_config_ptr))

    conv.my_float = out_opt_data_ptr.mean_freq[0]
    print_log (''.join('{:02x}'.format(x) for x in conv.my_bytes))

    print_log(f"number of rain drops = {rain_drop_count}")
    print_log(f"Mean freq = {out_opt_data_ptr.mean_freq[0]}")
    for x in out_opt_data_ptr.mean_freq:
        print_log(x)

    return rain_drop_count, out_opt_data_ptr.mean_freq[0]

def rain_detection_algo_device (audio_data, **kwargs):

    default_params = {
        'sample_rate': 11162,
        'freq_resolution': 45,
        'time_resolution_ms': 10,
        'check_duration': 4,
        'op_freq_range': [400, 3000],
        'n_freq_range': [400, 600],
        'fn': 400,
        'num_harmonics': 6,
        'harmonic_threshold': [4.25, 4, 4, 4, 4, 4],
        'max_peaks': 3,
        'log_factor': 0,
        'ns_duration_ms': 470,
        'nf': 0,
        'min_drop_count': 1
    }

    merged_params = {**default_params, **kwargs}

    scale_factor = 32767
    audio_data_in_t1 = audio_data * scale_factor
    audio_data_in = np.array(audio_data_in_t1, dtype=np.int16)

    out_opt_data_ptr = rain_cl_optional_data_t()

    audio_data_in.astype(dtype = np.int16).tofile('/tmp/audtmp')

    cwd = "/home/santhosh/repo/mark3-firmware-trunk/Utilities/M3cli/"
    audio_path = '/tmp/'
    bin_f = cwd + 'm3cli'
    aud_data = audio_path+'audtmp'
    cmd_flash_model = [bin_f, 'dfu_model raincl.bin', 'quit']
    cmd_data = [bin_f, "model_input "+aud_data, 'quit']
    cmd_run = [bin_f, "cm7ctl modelrun RAINCL.BIN", 'quit']
    #cmd_flash_model = [bin_f, "help", "quit"]
    #cmd_flash_model = ["ls", "-l"]
    # Run the command

    ################################# Send model
    print_log(cmd_flash_model)
    process = subps.Popen(cmd_flash_model, shell=False, stdout=subps.PIPE, stderr=subps.PIPE, cwd=cwd)

    # Get the output and error (if any)
    output, error = process.communicate()

    # Decode the output
    output = output.decode("utf-8")

    # Print the output
    print_log("Output:", output)

    # If there's any error, print it
    if error:
        error = error.decode("utf-8")
        print_log("Error:", error)

    # Get the return code
    return_code = process.returncode
    print_log("Return Code:", return_code)
    ################################# Send audio
    print_log(cmd_data)
    process = subps.Popen(cmd_data, shell=False, stdout=subps.PIPE, stderr=subps.PIPE, cwd=cwd)

    # Get the output and error (if any)
    output, error = process.communicate()

    # Decode the output
    output = output.decode("utf-8")

    # Print the output
    print_log("Output:", output)

    # If there's any error, print it
    if error:
        error = error.decode("utf-8")
        print_log("Error:", error)

    # Get the return code
    return_code = process.returncode
    print_log("Return Code:", return_code)

    ################################# Run model
    print_log(cmd_run)
    process = subps.Popen(cmd_run, shell=False, stdout=subps.PIPE, stderr=subps.PIPE, cwd=cwd)

    # Get the output and error (if any)
    output, error = process.communicate()

    # Decode the output
    output = output.decode("utf-8")
    output1, response = output.split('Response = ')
    response_string = response.replace('Response = ', '').strip()

    # Convert string to bytearray
    response_bytearray = bytearray(eval(response_string))
    out_opt_data_ptr = rain_cl_optional_data_t()

    data = response_bytearray[4:]
    rain_drop_count = int.from_bytes(response_bytearray[:4], "little")

    out_opt_data_ptr = rain_cl_optional_data_t.from_buffer_copy(data)

    print_log(f"Rain Drops : {rain_drop_count} - {out_opt_data_ptr.raindrops}")
    print_log(f"Mean Freq : {out_opt_data_ptr.mean_freq[0]}")

    # Print the output
    print_log("Output:", output)

    # If there's any error, print it
    if error:
        error = error.decode("utf-8")
        print_log("Error:", error)

    # Get the return code
    return_code = process.returncode
    print_log("Return Code:", return_code)

    return rain_drop_count, out_opt_data_ptr.mean_freq[0]



def sample_classifier_to_evaluate (
    audio_data,
    threshold=2,
    **kwargs) -> bool:

    default_params = {
        'sample_rate': 11162,
        'freq_resolution': 45,
        'time_resolution_ms': 10,
        'check_duration': 4,
        'op_freq_range': [400, 3000],
        'n_freq_range': [400, 600],
        'fn': 400,
        'num_harmonics': 6,
        'harmonic_threshold': [4.25, 4, 4, 4, 4, 4],
        'max_peaks': 3,
        'log_factor': 0,
        'ns_duration_ms': 470,
        'nf': 0,
        'min_drop_count': 1
    }

    merged_params = {**default_params, **kwargs}

    so_file = "/home/santhosh/repo/edge/RainCL/TargetPC/Test/lib/libdsp_shared_lib.so"
    dsp_functions = CDLL(so_file)
    print_log(type(dsp_functions))

    scale_factor = 32767
    audio_data_in_t1 = audio_data * scale_factor
    audio_data_in = np.array(audio_data_in_t1, dtype=np.int16)

    char_array = c_char * len(audio_data_in)

    init_func = dsp_functions.sample_classifier_to_evaluate_impl
    init_func.argtypes = [POINTER(evmgr_data_input_t), POINTER(rain_cl_optional_data_t), POINTER(rain_cl_config_param_t)]
    init_func.restype = c_int

    inp_data_ptr = evmgr_data_input_t()
    inp_data_ptr.audio_len = len(audio_data_in)
    inp_data_ptr.raw_audiop = cast(char_array.from_buffer(audio_data_in), c_char_p)

    out_opt_data_ptr = rain_cl_optional_data_t()

    inp_config_ptr = rain_cl_config_param_t()
    inp_config_ptr.sample_rate = merged_params['sample_rate']
    inp_config_ptr.freq_resolution = merged_params['freq_resolution']
    inp_config_ptr.time_resolution_ms = merged_params['time_resolution_ms']
    inp_config_ptr.check_duration = merged_params['check_duration']
    inp_config_ptr.fn = merged_params['fn']
    inp_config_ptr.op_freq_range[0] = merged_params['op_freq_range'][0]
    inp_config_ptr.op_freq_range[1] = merged_params['op_freq_range'][1]
    inp_config_ptr.n_freq_range[0] = merged_params['n_freq_range'][0]
    inp_config_ptr.n_freq_range[1] = merged_params['n_freq_range'][1]
    inp_config_ptr.harmonic_threshold[0] = merged_params['harmonic_threshold'][0]
    for i in range(1, 6):
        inp_config_ptr.harmonic_threshold[i] = merged_params['harmonic_threshold'][i]
    inp_config_ptr.num_harmonics = merged_params['num_harmonics']
    inp_config_ptr.max_peaks = merged_params['max_peaks']
    inp_config_ptr.log_factor = merged_params['log_factor']
    inp_config_ptr.ns_duration_ms = merged_params['ns_duration_ms']
    inp_config_ptr.nf = merged_params['nf']
    inp_config_ptr.min_drop_count = merged_params['min_drop_count']

    class Convert(Union):
        _fields_ = (("my_bytes", c_char * sizeof(c_float)),
                    ("my_float", c_float))

    conv = Convert()

    # Convert negative numbers to two's complement representation
    #converted_data = np.vectorize(twos_complement_hex)(audio_data_in)
    #np.savetxt('aud_in_data.txt', converted_data, fmt='%#x')



    rain_drop_count = dsp_functions.sample_classifier_to_evaluate_impl(
                        byref(inp_data_ptr), byref(out_opt_data_ptr), byref(inp_config_ptr))

    conv.my_float = out_opt_data_ptr.mean_freq[0]
    print_log (''.join('{:02x}'.format(x) for x in conv.my_bytes))

    print_log(f"number of rain drops = {rain_drop_count}")
    print_log(f"Mean freq = {out_opt_data_ptr.mean_freq[0]}")
    for x in out_opt_data_ptr.mean_freq:
        print_log(x)

    if rain_drop_count > threshold:
        return True
    elif 0 <= rain_drop_count <= threshold:
        return False
    else:
        return np.nan


def get_version (dsp_functions):

    ver = bytearray(1024)
    char_array = c_char * len(ver)

    dsp_functions.get_version_info(char_array.from_buffer(ver), len(ver))

    print_log(str(ver, 'UTF-8'))


# def run_model():
#     so_file = "/home/santhosh/repo/edge/RainCL/TargetPC/Test/lib/libdsp_shared_lib.so"
#
#     dsp_functions = CDLL(so_file)
#
#     print_log(type(dsp_functions))
#
#     get_version(dsp_functions)
#
#     #typedef struct {
#     #    int                         audio_len;
#     #    uint8_t                     *raw_audiop;
#     #    int                         image_len;
#     #    uint8_t                     *imagep;
#     #    evmgr_sensor_data_t         sensor_data;
#     #} evmgr_data_input_t;
#
#
#     #typedef struct {
#     #    uint8_t    sensor_id;
#     #    uint8_t    len;
#     #    uint16_t   reserved;
#     #    float      buf[];
#     #} evmgr_sensor_data_t;
#
#     path = os.path.join('/','home', 'santhosh', 'Santhosh', 'DSP_model','datasets','manually_validated')
#
#     os.chdir(path)
#     print_log(os.listdir());
#     data = []
#     idx = 0
#     for filename in os.listdir():
#
#         if filename.endswith(".BIN") or filename.endswith(".wav"):
#
#             print_log(filename)
#
#             audio_data_in_f = read_audio_file(filename, 11162*2*2, 0)
#
#             scale_factor = 32767
#             audio_data_in_t1 = audio_data_in_f * scale_factor
#             audio_data_in = np.array(audio_data_in_t1, dtype=np.int16)
#
#             char_array = c_char * len(audio_data_in)
#
#             rain_detection_algo(audio_data_in_f, threshold=2 ,
#                                           op_freq_range=[400,3000], n_freq_range=[400, 600],
#                                           ns_duration_ms = 470)
#             """
#             rain_detection_algo_device(audio_data_in_f, threshold=2 ,
#                                           op_freq_range=[400,3000], n_freq_range=[400, 600],
#                                           ns_duration_ms = 470)
#             """
#
#             init_func = dsp_functions.sample_classifier_to_evaluate_impl
#             init_func.argtypes = [POINTER(evmgr_data_input_t), POINTER(rain_cl_optional_data_t), POINTER(rain_cl_config_param_t)]
#             init_func.restype = c_int
#
#             inp_data_ptr = evmgr_data_input_t()
#             inp_data_ptr.audio_len = len(audio_data_in)
#             inp_data_ptr.raw_audiop = cast(char_array.from_buffer(audio_data_in), c_char_p)
#
#             out_opt_data_ptr = rain_cl_optional_data_t()
#
#             inp_config_ptr = rain_cl_config_param_t()
#             inp_config_ptr.sample_rate = 11162
#             inp_config_ptr.freq_resolution = 45
#             inp_config_ptr.time_resolution_ms = 10
#             inp_config_ptr.check_duration = 4
#             inp_config_ptr.fn = 400
#             inp_config_ptr.op_freq_range[0] = 400
#             inp_config_ptr.op_freq_range[1] = 3000
#             inp_config_ptr.n_freq_range[0] = 400
#             inp_config_ptr.n_freq_range[1] = 600
#             inp_config_ptr.harmonic_threshold[0] = 4.25
#             for i in range(1, 6):
#                 inp_config_ptr.harmonic_threshold[i] = 4
#             inp_config_ptr.num_harmonics = 6
#             inp_config_ptr.max_peaks = 3
#             inp_config_ptr.log_factor = 0
#             inp_config_ptr.ns_duration_ms = 470
#             inp_config_ptr.nf = 0
#             inp_config_ptr.min_drop_count = 1
#
#             class Convert(Union):
#                 _fields_ = (("my_bytes", c_char * sizeof(c_float)),
#                             ("my_float", c_float))
#
#             conv = Convert()
#
#             # Convert negative numbers to two's complement representation
#             #converted_data = np.vectorize(twos_complement_hex)(audio_data_in)
#             #np.savetxt('aud_in_data.txt', converted_data, fmt='%#x')
#
#             """
#             Test sample_classifier_to_evaluate function
#             """
#
#
#             classifier_boolean_wrapper(audio_data_in_f, op_freq_range=[400,3000], n_freq_range=[400, 600],
#                                           ns_duration_ms = 470)
#
#             rain_drop_count = dsp_functions.sample_classifier_to_evaluate_impl(
#                                 byref(inp_data_ptr), byref(out_opt_data_ptr), byref(inp_config_ptr))
#
#             conv.my_float = out_opt_data_ptr.mean_freq[0]
#             print_log (''.join('{:02x}'.format(x) for x in conv.my_bytes))
#
#             print_log(f"number of rain drops = {rain_drop_count}")
#             print_log(f"Mean freq = {out_opt_data_ptr.mean_freq[0]}")
#             for x in out_opt_data_ptr.mean_freq:
#                 print_log(x)
#
#
#             # if (rain_drop_count < 5):
#             #     rain_drop_count = 0
#             if filename.startswith("True"):
#                 category = 1
#             else:
#                 category = 0
#
#             #print_log(f'{idx:5} {rain_drop_count:5} {int(out_opt_data_ptr.mean_freq[0]):5} {category:2}:{filename}')
#             row = {
#                 "file_name": filename,
#                 "mean_freq": out_opt_data_ptr.mean_freq[0],
#                 "rain_drops": rain_drop_count,
#                 "category": category,
#             }
#             data.append(row)
#
#             #if (idx % 25) == 0:
#             #    print(idx)
#             #idx = idx + 1
#             # if (count > 10):
#             #    break;
#     csv_filename = "test_results.csv"
#     csv_columns = ["file_name", "mean_freq", "rain_drops", "category"]
#     write_results(csv_filename, csv_columns, data)
#     print_log("done")
#
# def main():
#     run_model()
#
#
# if __name__ == "__main__":
#     main()
