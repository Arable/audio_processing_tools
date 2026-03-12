import datetime as dt
import hashlib
import threading
import time
from typing import Callable
from collections import deque

import pandas as pd
import plotly.graph_objects as go
import requests
from IPython.display import Audio, clear_output, display
from ipywidgets import Button, HBox, Output
from audio_processing_tools.db_tools import (
    get_db_data,
    upsert_df,
)
from audio_processing_tools.fetch import get_device_raw_audio_data
from audio_processing_tools.parse import parse_mark_audio_file, pcm_to_float
from audio_processing_tools.visualize_audio import plot_audio_signal, plot_audio_spectrogram


class TestVectorLabeller:
    def __init__(
        self,
        audio_df: pd.DataFrame,
        db_engine,
        db_engine_upsert=None,
        max_duration_seconds=15,
        local_audio_cache="./raw_audio_cache",
        normalize_audio: bool = True,
        autoplay: bool = True,
        visualize_device_context=False,
        context_window_days=5,
        add_ibm_data=True,
        visualize_time_series_signal=False,
        visualize_signal_spectrogram=False,
    ):
        self.audio_df = audio_df.copy()
        if "source_file" not in self.audio_df.columns:
            raise ValueError("audio_df must contain a 'source_file' column")
        if self.audio_df["source_file"].isnull().any():
            raise ValueError("audio_df contains null values in 'source_file'")
        if not self.audio_df["source_file"].is_unique:
            raise ValueError("audio_df must have unique source_file values")
        if not self.audio_df.index.equals(pd.Index(self.audio_df["source_file"])):
            self.audio_df = self.audio_df.set_index("source_file", drop=False)
        self.db_engine = db_engine
        self.db_engine_upsert = db_engine_upsert or db_engine
 
        self.max_duration_seconds = max_duration_seconds
        self.local_audio_cache = local_audio_cache
        self.normalize_audio = normalize_audio
        self.autoplay = autoplay
        self.index_list = self.audio_df.index
        self.index_iter = iter(self.index_list)
        self.main_output = Output()  # Main output widget for buttons and text
        self.audio_output = Output()  # Separate output for audio
        self.signal_output = Output()  # Output for audio signal plot
        self.spectrogram_output = Output()  # Output for spectrogram plot
        self.figure_output = Output()  # Output for device context figure
        self.visualize_device_context = visualize_device_context
        self.context_window_days = context_window_days
        self.add_ibm_data = add_ibm_data
        self.visualize_time_series_signal = visualize_time_series_signal
        self.visualize_signal_spectrogram = visualize_signal_spectrogram
        self.history_stack = deque()  # History stack to keep track of processed indices
        self.upsert_threads = []

    def reset(self) -> None:
        """Reset navigation state and recreate widget outputs for a fresh labeling run."""
        self.index_list = self.audio_df.index
        self.index_iter = iter(self.index_list)
        self.history_stack = deque()
        self.main_output = Output()
        self.audio_output = Output()
        self.signal_output = Output()
        self.spectrogram_output = Output()
        self.figure_output = Output()
        self.upsert_threads = []

    def label_vectors(self) -> None:
        """Start a fresh labeling session from the first file."""
        self.reset()
        display(self.main_output)
        display(self.audio_output)
        display(self.signal_output)
        display(self.spectrogram_output)
        display(self.figure_output)
        self.process_next_index()

    def process_next_index(self) -> None:
        try:
            next_index = next(self.index_iter)
            self.history_stack.append(next_index)  # Save the index to history stack
            self.process_index(next_index, self.process_next_index, self.main_output)
        except StopIteration:
            with self.main_output:
                clear_output(wait=True)
                print("All files have been processed.")

    def process_previous_index(self) -> None:
        if len(self.history_stack) > 1:
            self.history_stack.pop()  # Remove the current index
            previous_index = self.history_stack.pop()  # Get the previous index
            self.index_iter = iter(
                self.index_list[self.index_list.get_loc(previous_index):]
            )  # Reset iterator
            self.process_index(
                previous_index, self.process_next_index, self.main_output
            )
        else:
            with self.main_output:
                print("No previous file to go back to.")

    @staticmethod
    def str_to_bool(s: str) -> bool:
        return str(s).lower() == "true"

    @staticmethod
    def generate_uid(data):
        hash_obj = hashlib.sha256()
        hash_obj.update(data.encode())
        return hash_obj.hexdigest()

    @staticmethod
    def fetch_ibm_data(db_engine, start_date, end_date, lat, long):
        start_date_sql = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_date_sql = end_date.strftime("%Y-%m-%d %H:%M:%S")
        ibm_query = f"""
        SELECT
          time_utc as time, precip as ibm_precip
        FROM
          ext_weather.hist_local_hourly
        WHERE
          time_utc BETWEEN '{start_date_sql}' AND '{end_date_sql}' 
          AND lat BETWEEN {lat} - 0.005 AND {lat} + 0.005
          AND long BETWEEN {long} - 0.005 AND {long} + 0.005
        """
        return get_db_data(ibm_query, db_engine)

    @staticmethod
    def plot_device_context(db_engine, key_of_interest, audio_df, window_size, display_ibm_data):
        center_time = audio_df.loc[key_of_interest]["time"]
        start_time = center_time - pd.Timedelta(days=window_size / 2)
        end_time = center_time + pd.Timedelta(days=window_size / 2)
        device_id = audio_df.loc[key_of_interest]["device_id"]
        audio_df_for_plot = audio_df[
            (audio_df["device_id"] == device_id)
            & (audio_df["time"].between(start_time, end_time))
        ].copy()
        fig = go.Figure(
            [
                go.Scatter(
                    x=audio_df_for_plot["time"],
                    y=audio_df_for_plot["device_id"],
                    name="Adjacent Audio Recordings",
                    mode="markers",
                ),
                go.Scatter(
                    x=[audio_df.loc[key_of_interest]["time"]],
                    y=[audio_df.loc[key_of_interest]["device_id"]],
                    name="Current Audio File",
                    mode="markers",
                ),
            ],
            go.Layout(title=f"Audio Context For {device_id}"),
        )
        if display_ibm_data:
            lat = audio_df.loc[key_of_interest]["lat"]
            long = audio_df.loc[key_of_interest]["long"]

            if (lat == 0 and long == 0) or pd.isnull(lat) or pd.isnull(long):
                print("Could not get IBM data due to bad coordinates")
                fig.show()
                return

            try:
                ibm_plot_data = TestVectorLabeller.fetch_ibm_data(
                    db_engine, start_time, end_time, lat, long
                )
            except Exception as e:
                print(f"Could not fetch IBM data: {e}")
                fig.show()
                return

            if ibm_plot_data.empty:
                print(f"IBM data for {lat}, {long} not found in db")
                fig.show()
                return

            fig.update_layout(
                yaxis2={
                    "overlaying": "y",
                    "side": "right",
                    "title": "IBM rain (mm)",
                }
            )
            fig.add_trace(
                go.Scatter(
                    mode="lines",
                    x=ibm_plot_data["time"],
                    y=ibm_plot_data["ibm_precip"],
                    yaxis="y2",
                    name="IBM precip",
                )
            )
        fig.show()

    def process_index(
        self, index: str, next_index_callback: Callable, output_widget: Output
    ) -> None:
        with output_widget:
            audio_file_data = self.audio_df.loc[index].copy()
            clear_output(wait=True)
            source_file = audio_file_data["source_file"]
            current_position = self.index_list.get_loc(index) + 1
            total_files = len(self.index_list)
            print(f"File {current_position} of {total_files}")

            audio_data = get_device_raw_audio_data(
                keys=[source_file],
                local_cache_location=self.local_audio_cache,
                redownload=False,
                use_caching=True,
                header_only=False,
                verbose=False,
            )
            if source_file not in audio_data:
                raise KeyError(
                    f"Fetched audio data does not contain key {source_file!r}. Available keys: {list(audio_data.keys())[:10]}"
                )
            audio_binary = audio_data[source_file]
            sig, metadata = parse_mark_audio_file(audio_binary)
            sample_rate = metadata["sample_rate"]
            duration_seconds = len(sig) / sample_rate
            start_time = 0
            end_time = min(duration_seconds, self.max_duration_seconds)
            audio_file_data["segment_start_seconds"] = start_time
            audio_file_data["segment_end_seconds"] = end_time
            print(f"Working on {source_file} from {start_time}s to {end_time}s")

            start_frame = int(start_time * sample_rate)
            end_frame = int(end_time * sample_rate)
            sig_of_interest = sig[start_frame:end_frame]

            with self.audio_output:
                clear_output(wait=True)
                display(
                    Audio(
                        data=pcm_to_float(sig_of_interest),
                        rate=sample_rate,
                        normalize=self.normalize_audio,
                        autoplay=self.autoplay,
                    )
                )

            raining_button = Button(description="Raining")
            not_raining_button = Button(description="Not Raining")
            skip_button = Button(description="Skip")
            go_back_button = Button(description="Go Back")

            raining_button.on_click(
                self.make_button_handler(
                    audio_file_data, output_widget, True, next_index_callback
                )
            )
            not_raining_button.on_click(
                self.make_button_handler(
                    audio_file_data, output_widget, False, next_index_callback
                )
            )
            skip_button.on_click(lambda b: next_index_callback())
            go_back_button.on_click(
                lambda b: self.process_previous_index()
            )  # Go back action

            button_box = HBox(
                [raining_button, not_raining_button, skip_button, go_back_button]
            )
            display(button_box)

            if self.visualize_time_series_signal:
                with self.signal_output:
                    clear_output(wait=True)
                    plot_audio_signal(
                        pcm_to_float(sig_of_interest), sample_rate, title=source_file
                    )
            if self.visualize_signal_spectrogram:
                with self.spectrogram_output:
                    clear_output(wait=True)
                    plot_audio_spectrogram(pcm_to_float(sig_of_interest), sample_rate)

            if self.visualize_device_context:
                with self.figure_output:
                    clear_output(wait=True)
                    self.plot_device_context(
                        self.db_engine,
                        index,
                        self.audio_df,
                        self.context_window_days,
                        self.add_ibm_data,
                    )

    def make_button_handler(
        self,
        data: pd.Series,
        output_widget: Output,
        rain_status: bool,
        next_index_callback: Callable,
    ) -> Callable:
        def on_button_clicked(b):
            try:
                self.update_rain_label(data, rain_status, output_widget)
                time.sleep(0.5)
                next_index_callback()
            except Exception as e:
                print(f"Error in button handler: {e}")

        return on_button_clicked

    def update_rain_label(
        self, audio_file_data: pd.Series, rain_status: bool, output_widget: Output
    ) -> None:
        with output_widget:
            display(
                f"Rain label being updated to {'TRUE' if rain_status else 'FALSE'}..."
            )
            current_time = dt.datetime.utcnow()
            data_to_upload = pd.Series(dtype="object")
            data_to_upload["source_file"] = audio_file_data["source_file"]
            data_to_upload["device"] = audio_file_data["device_id"]
            data_to_upload["start_time"] = audio_file_data["time"]
            data_to_upload["segment_start_seconds"] = audio_file_data[
                "segment_start_seconds"
            ]
            data_to_upload["segment_end_seconds"] = audio_file_data[
                "segment_end_seconds"
            ]
            data_to_upload["site"] = None
            data_to_upload["source"] = "manually labeled"
            data_to_upload["raining"] = rain_status
            data_to_upload["corrected"] = False
            try:
                data_to_upload["creator"] = requests.get(
                    "https://api.ipify.org", timeout=5
                ).content.decode("utf8")
            except Exception:
                data_to_upload["creator"] = "unknown"
            data_to_upload["update_time"] = current_time
            data_to_upload["create_time"] = current_time
            data_to_upload["manually_labeled"] = True
            uid = (
                str(data_to_upload["source_file"])
                + str(data_to_upload["segment_start_seconds"])
                + str(data_to_upload["segment_end_seconds"])
            )
            data_to_upload["uid"] = self.generate_uid(uid)
            data = pd.DataFrame([data_to_upload])
            data.set_index("uid", inplace=True)
            # Fire-and-forget background upsert. Daemon thread allows notebook shutdown
            # without waiting, but an in-flight write may be interrupted on kernel exit.
            thread = threading.Thread(
                target=self.background_upsert,
                args=(data,),
                daemon=True,
            )
            thread.start()
            self.upsert_threads.append(thread)

    def background_upsert(self, data: pd.DataFrame) -> None:
        try:
            upsert_df(
                data, "device_audio_rain_classification", self.db_engine_upsert
            )
            print("Database upsert completed successfully.")
        except Exception as e:
            print(f"Error during database upsert: {e}")
