import os
import csv
import math

import numpy as np
from scipy.signal import get_window
from scipy.fftpack import fft
from matplotlib import pyplot as plt
import librosa
import copy

class DsdProcessingEmualtor:
    def __init__(
        self,
        fs=11162,
        frame_length=512,
        hop_length=512,
        bwindow=False,
        ts=0,
        verbose=False,
    ):
        self.fs = fs  # sample rate
        self.frame_length = frame_length
        self.fft_n_bins = int(frame_length / 2)

        self.hop_length = hop_length
        self.apply_window = bwindow

        self.verbose = verbose

        # Frequency resolution
        self.dF = self.fs / self.frame_length

        # output
        self.loudness_bins = 32
        self.pft_bins = 30
        self.fft_bins = 38

        # rain detection parameters
        self.rain_chk_period_seconds = 60
        self.rain_chk_duration_seconds = 3

        self.rain_energy_threshold = 0.6
        self.rain_low_freq = 400
        self.rain_high_freq = 700
        self.rain_low_idx = int(self.rain_low_freq // self.dF) + 1
        self.rain_high_idx = int(self.rain_high_freq // self.dF)

        self.rain_log_base = 1.13
        self.rain_log_factor = 0.6

        # pft parameters
        self.pft_low_freq = 100
        self.pft_high_freq = 1500
        self.pft_low_idx = int(self.pft_low_freq // self.dF) + 1
        self.pft_high_idx = int(self.pft_high_freq // self.dF) - 1

        # fft parameters
        self.lwin_start = 300
        self.hwin_start = 1000
        self.lwin_start_idx = int(self.lwin_start // self.dF)
        self.lwin_end_idx = (
            int(self.lwin_start // self.dF) + int(self.fft_bins // 2) - 1
        )
        self.hwin_start_idx = int(self.hwin_start // self.dF)
        self.hwin_end_idx = (
            int(self.hwin_start // self.dF) + int(self.fft_bins // 2) - 1
        )
        # raw file header size
        self.hdr_size = 38

        # state variables of the class
        self.ts_start = 0
        self.ts_current = 0
        self.total_frames = 0
        self.frame_count = 0
        self.energy_histogram = np.zeros(
            self.loudness_bins + self.pft_bins + self.fft_bins
        )
        self.peak_histogram = np.zeros(self.fft_n_bins)
        self.freq_histogram = np.zeros(self.fft_n_bins)
        self.raining = True

    def __str__(self):
        ret = f"###### start of the config block #########\n"
        ret += f"Sampling frequency = {self.fs} \n"
        ret += f"FFT length = {self.frame_length}\n"
        ret += f"frequency resultion = {self.dF} Hz\n"
        ret += f"Rain Enegry Threshold = {self.rain_energy_threshold}\n"
        ret += f"Windowing Enabled = {self.apply_window}\n"
        ret += f"Rain Detection Band = {self.rain_low_freq} {self.rain_high_freq} Hz  indices = {self.rain_low_idx} {self.rain_high_idx} \n"
        ret += f"lwin = {self.lwin_start} {self.lwin_end_idx * self.dF} Hz  lwin indices = {self.lwin_start_idx} {self.lwin_end_idx} \n"
        ret += f"hwin = {self.hwin_start} {self.hwin_end_idx*self.dF} Hz  hwin indices = {self.hwin_start_idx} {self.hwin_end_idx} \n"
        ret += f"pft frequencies = {self.pft_low_freq} {self.pft_high_freq} Hz  pft indices =  {self.pft_low_idx} {self.pft_high_idx} \n"
        ret += f"###### end of the config block #########\n"
        return ret

    def clear_histogram (self):
        self.energy_histogram.fill(0)
        self.peak_histogram.fill(0)
        self.freq_histogram.fill(0)

    def set_audio_timestamp (self, ts, sample_count):
        self.ts_start = ts - (ts % self.rain_chk_period_seconds)
        self.ts_current = ts
        self.frame_count = int(
            (self.ts_current % self.rain_chk_period_seconds) * self.fs / self.hop_length
        )
        self.total_frames = int (sample_count/self.hop_length)
        if (sample_count - self.total_frames * self.hop_length) < self.frame_length:
            if self.total_frames > 1:
                self.total_frames -= 1
    
    def index_falls_in_lower_window(self, i):
        # does the index fall in the lower window
        return self.lwin_start_idx <= i <= self.lwin_end_idx

    def index_falls_in_upper_window(self, i):
        # check if the index fall in the upper window
        if self.hwin_start_idx == self.lwin_end_idx:
            return False
        return self.hwin_start_idx <= i <= self.hwin_end_idx

    def process_audio_frame(self, audio_data):
        # Extract one audio frame
        frame = audio_data[: self.frame_length]
        # calculate frequency spectrum
        if self.apply_window:
            window = get_window("hanning", self.frame_length)
            frame = frame * window
            
        spectrum = np.abs(fft(frame))

        # calculate peak energy and peak enegry index
        pft_spectrum = spectrum[self.pft_low_idx:self.pft_high_idx]
        peak_energy_index = np.argmax(pft_spectrum) + self.pft_low_idx
        peak_energy = spectrum[peak_energy_index]
        if peak_energy != 0:
            self.peak_histogram[int(peak_energy_index)] += 1
            self.freq_histogram[int(peak_energy_index)] += peak_energy
        
        # Assign the peak frequecy index 
        # here it is possible to check for the change of index for efficiency
        next_frame_time = self.ts_current + self.hop_length / self.fs
        next_pft_idx = int((next_frame_time % 60) / 2)
        pft_idx = int((self.ts_current % 60) / 2)
        peak_frequency_idx = np.argmax(self.peak_histogram)
        peak_frequency = self.peak_histogram[peak_frequency_idx]
        self.energy_histogram[self.loudness_bins + pft_idx] = peak_frequency_idx
        if next_pft_idx != pft_idx:
            self.peak_histogram.fill(0)

        # calculate raindrop histogram
        drop_energy_level = 0.0
        for i in range(self.rain_low_idx, self.rain_high_idx + 1):
            drop_energy_level += spectrum[i]

        if drop_energy_level > self.rain_energy_threshold:
            logbase = math.log(self.rain_log_base)
            rain_energy = (
                drop_energy_level - self.rain_energy_threshold
            ) * self.rain_log_factor
            histidx = math.floor(math.log(1 + rain_energy) / logbase)
            if histidx > self.loudness_bins - 1:
                histidx = self.loudness_bins - 1
            if histidx < 0:
                histidx = 0
            self.energy_histogram[histidx] += 1

        # Remove the processed samples from the audio_data
        audio_data = audio_data[self.hop_length :]

        self.frame_count += 1
        self.ts_current = self.ts_start + self.frame_count*self.hop_length/self.fs
        return audio_data

    def calculate_fft_energies(self):
        # calculate accumulated fft energies for peak frequencies
        exp_pow_one = 2.719
        scale_freq = 25.0
        upper_uint8_bound = 255

        for i in range(self.fft_n_bins):
            j = int(math.log(self.freq_histogram[i] + exp_pow_one) * scale_freq)
            if j > upper_uint8_bound:
                j = upper_uint8_bound

            if self.index_falls_in_lower_window(i):
                tmp_index = i - self.lwin_start_idx
                index = self.loudness_bins + self.pft_bins + tmp_index
                self.energy_histogram[index] = int(j)

            if self.index_falls_in_upper_window(i):
                tmp_index = (i - self.hwin_start_idx) + (self.fft_bins // 2)
                index = self.loudness_bins + self.pft_bins + tmp_index
                self.energy_histogram[index] = int(j)

    def check_histogram_for_rain(self):
        raining = False
        for idx in range(self.loudness_bins):
            if self.energy_histogram[idx] != 0:
                raining = True
                break
        self.raining = raining
        return self.raining

    def get_frames_to_next_interval (self, audio_data):
        #calculate time remaining till minute boundary 
        time_to_next_interval = self.get_time_to_next_interval()
        #frames to process till next interval     
        frames = int(time_to_next_interval*self.fs/self.hop_length)
        frames_remaining = int(len(audio_data) / self.hop_length)
        if frames_remaining < frames:
            frames = frames_remaining
        if len (audio_data) < self.frame_length:
            frames = 0
        return frames

    def get_time_to_next_interval(self):
        # time to the next interval to calculate rain statistics
        time_to_next_interval = self.rain_chk_period_seconds - (
            self.ts_current % self.rain_chk_period_seconds
        )

        # if the time remaining is less than a frame then add that partial frame to next
        # rain check period
        if time_to_next_interval < self.hop_length / self.fs:
            time_to_next_interval += self.rain_chk_period_seconds
        return time_to_next_interval

    def process_audio_upto_minute_boundary(self, audio_data):
        
        frames = self.get_frames_to_next_interval(audio_data)
        if self.verbose:
            print (f"frames to next raincheck {frames} ")
        
        for fc in range(frames):
            if len(audio_data) >= self.frame_length:
                audio_data = self.process_audio_frame(audio_data)
        # calculate and store 38 fft energies
        self.calculate_fft_energies()
        return audio_data

    def get_next_raincheck_time(self):
        #time to next raincheck
        time_to_next_interval = self.get_time_to_next_interval()
        raincheck_time = (
            self.ts_current + time_to_next_interval - self.rain_chk_duration_seconds
        )
        return raincheck_time

    def process_audio_data(self, audio_data, ts):
        # calculate audio data duration in minutes
        self.set_audio_timestamp (ts, len(audio_data))
        num_minutes = math.ceil(len(audio_data) / (self.fs * 60))
        output = []
        if len(audio_data) < self.frame_length:
            return output
        data_to_process = True
        for m in range(num_minutes):
            if self.verbose:
                print(f"process 1 min audio data at the offset of {m} minutes")
            self.clear_histogram()
            if self.raining == True:
                # if raining process all the audio frames this time onward
                audio_data = self.process_audio_upto_minute_boundary(audio_data)

            else:
                rain_check_time = self.get_next_raincheck_time()
                # skips audio frames till next rain check time
                while self.ts_current < rain_check_time:
                    audio_data = audio_data[self.hop_length :]
                    self.frame_count += 1
                    self.ts_current = (
                        self.ts_start + self.frame_count * self.hop_length / self.fs
                    )
                    if len(audio_data) < self.frame_length:
                        data_to_process = False
                        break

                if data_to_process == False:
                    break
                # clear the old histogram
                self.clear_histogram()
                if self.verbose:
                    print(
                        f"rain check duration = {rain_check_time + self.rain_chk_duration_seconds - self.ts_current}"
                    )
                while self.ts_current < (
                    rain_check_time + self.rain_chk_duration_seconds
                ):
                    if len(audio_data) >= self.frame_length:
                        audio_data = self.process_audio_frame(audio_data)
                    else :
                        data_to_process = False
                        break
                if data_to_process == False:
                    break

            self.check_histogram_for_rain()
            if self.verbose:
                print(f"Raining Status = {self.raining}")
            if self.verbose:
                print(self.energy_histogram)
            output.append (copy.deepcopy(self.energy_histogram))
            self.clear_histogram()
            if (data_to_process == False) or (len(audio_data) < self.frame_length):
                break

        return output


def read_audio_file(audio_file, read_size, read_offset):
    if audio_file.lower().endswith(".wav"):
        audio_data_in, sr_ref = librosa.load(audio_file, sr=11162)
        # audio_data_in = audio_data_in[read_offset:read_offset+read_size];

    else:
        with open(audio_file, "rb") as file:
            # skip the audio file header
            file.seek(HEADER_SIZE),
            #read the file in to an ndarray
            audio_data_in = file.read()
            # scale it to float values between -1.0 to 1.0
            scale_factor = 1 << (bytes_per_sample * 8 - 1)
            audio_data_in = np.frombuffer(audio_data_in, dtype=np.int16) / scale_factor

    audio_data_in = audio_data_in[read_offset : read_offset + read_size]

    return audio_data_in


def plot_data(sp_rows, sp_cols, sp_idx, ax1, val, duration, title):
    # ax = plt.subplot(sp_rows, sp_cols, sp_idx, sharex=ax1)
    ###
    # fig, ax1 = plt.subplots(sp_rows, sp_cols, figsize=(10, 10))
    if enable_plot_data == True:
        if sp_idx == 0:
            fig = plt.figure(figsize=(10, 15))
            fig.suptitle(title)
            gs = fig.add_gridspec(sp_rows)
            ax1 = gs.subplots(sharex=True)

            # fig, ax1 = plt.subplots(sp_rows, sp_cols, figsize=(10, 8), sharex=True)
            time = np.arange(len(val)) * duration / len(val)
            ax1[sp_idx].plot(time, val)

            plt.grid(which="both", axis="both")
            plt.tight_layout()

            return fig, ax1
        ####
        else:
            time = np.arange(len(val)) * duration / len(val)
            ax1[sp_idx].plot(time, val)
            ax1[sp_idx].set_title(title)  # , y=1.0, pad=-14
            plt.grid(which="both", axis="both")
            plt.tight_layout()

            return 0, ax1
    return 0, 0


def write_results(csv_file_name, csv_columns, data):
    with open(csv_filename, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


# sample file run:
if __name__ == "__main__":
    current_dir = "/home/vikrant/rain_testvecs/"
    os.chdir(current_dir)
    list1 = [
        (1, "63482897.RAW"),
        (1, "6348211C.RAW"),
        (1, "1678743960"),
        (0, "1678200300"),
        (1, "634D15FC.RAW"),
    ]

    HEADER_SIZE = 38
    enable_plot_data = False
    max_rows = 1
    Fs, bytes_per_sample, frame_size, duration = 11162, 2, 512, 120
    n_frames = int(duration * Fs / frame_size)

    # read audio data
    read_size = int(frame_size * n_frames)
    read_offset = HEADER_SIZE

    for x in list1:
        category = x[0]
        filename = x[1]
        print(filename, read_size, read_offset)
        audio_data_in = read_audio_file(filename, read_size, read_offset)
        sp_idx = 0
        print(f"file={filename}, {len(audio_data_in)}. {n_frames}")
        if enable_plot_data == True:
            title = filename + " waveform"
            # fig, ax1 = plot_data (max_rows, 1, sp_idx, 0, audio_data_in, duration, title)
            duration = len(audio_data_in) / Fs
            time = np.arange(len(audio_data_in)) * duration / len(audio_data_in)
            plt.rcParams["figure.figsize"] = (10, 4)
            plt.plot(time, audio_data_in)
            plt.grid(which="both", axis="both")
            plt.show()

        ts = 0

        rmodel = DsdProcessingEmualtor(11162, 512, 512, False, 0)
        print(rmodel)

        output_array = rmodel.process_audio_data(audio_data_in, 0)
        print(f"output array len = {len(output_array)}")
        index = 0
        for o in output_array:
            print(f"frame number = {index}")
            print(f"loudness bins = \n{o[:32]}")
            print(f"peak indices = \n{o[32:32+30]}")
            print(f"fft_bins = \n{o[32+30:32+30+38]}")
            index += 1
        # print (output_array[1])
        # print(output_array)
