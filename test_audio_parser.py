import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import wave
import tempfile
import os
import subprocess
import platform

from audio_processing_tools.parse import parse_mark_audio_file

def save_and_play_audio(signal, sample_rate, temp_dir):
    """Save audio as WAV and open with system media player"""
    # Generate temporary WAV file path
    wav_path = os.path.join(temp_dir, "audio_preview.wav")
    
    print(f"\nSaving WAV file to: {wav_path}")
    
    # Scale the normalized signal to INT16 range for better volume
    signal_int16 = (signal * 32767).astype(np.int16)
    
    print(f"Signal stats before saving:")
    print(f"- Shape: {signal_int16.shape}")
    print(f"- Sample rate: {sample_rate}")
    print(f"- Min/Max: {signal_int16.min()}/{signal_int16.max()}")
    
    # Save as WAV file
    try:
        with wave.open(wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 2 bytes for 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(signal_int16.tobytes())
        
        print(f"WAV file saved successfully: {os.path.getsize(wav_path)} bytes")
        
        # Verify the saved WAV file
        with wave.open(wav_path, 'rb') as wav_file:
            print(f"\nWAV file verification:")
            print(f"- Channels: {wav_file.getnchannels()}")
            print(f"- Sample width: {wav_file.getsampwidth()}")
            print(f"- Frame rate: {wav_file.getframerate()}")
            print(f"- Frames: {wav_file.getnframes()}")
            
        # Open with system default player
        print(f"\nAttempting to play with system player...")
        if platform.system() == 'Darwin':       # macOS
            print("Detected macOS, using 'open' command")
            result = subprocess.run(['open', wav_path], capture_output=True, text=True)
            print(f"Open command result: {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr}")
        elif platform.system() == 'Windows':    # Windows
            print("Detected Windows, using os.startfile")
            os.startfile(wav_path)
        else:                                   # Linux
            print("Detected Linux, using 'xdg-open' command")
            result = subprocess.run(['xdg-open', wav_path], capture_output=True, text=True)
            print(f"xdg-open command result: {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr}")
        
        print(f"\nOpened audio file: {wav_path}")
        return wav_path
        
    except Exception as e:
        print(f"Error during WAV file operations: {str(e)}")
        raise

def visualize_and_play_audio(file_path, force_file_type=None):
    """
    Read, visualize and play an audio file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the audio file
    force_file_type : str, optional
        Override automatic format detection ('pcm' or 'alac')
    """
    print(f"\nProcessing file: {file_path}")
    print(f"Force file type: {force_file_type}")
    
    # Read the file
    with open(file_path, 'rb') as f:
        file_contents = f.read()
    print(f"Read {len(file_contents)} bytes from file")
    
    # Parse the audio file
    signal, metadata = parse_mark_audio_file(file_contents, force_file_type=force_file_type)
    
    # Print metadata
    print("\nMetadata:")
    for key, value in metadata.items():
        print(f"{key}: {value}")
    
    # Normalize signal for visualization and playback
    signal_max = np.max(np.abs(signal))
    if signal_max > 0:  # Avoid division by zero
        signal_normalized = signal / signal_max
    else:
        signal_normalized = signal
    
    # Create time axis
    duration = len(signal) / metadata['sample_rate']
    time = np.linspace(0, duration, len(signal))
    
    # Plot the waveform
    plt.figure(figsize=(15, 5))
    plt.plot(time, signal_normalized)
    plt.title(f"Waveform - {Path(file_path).name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    
    # Print audio signal info
    print(f"\nAudio signal info:")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample rate: {metadata['sample_rate']} Hz")
    print(f"Signal range: {signal_normalized.min():.2f} to {signal_normalized.max():.2f}")
    
    # Create a temporary directory for WAV files
    temp_dir = tempfile.mkdtemp()
    print(f"\nCreated temporary directory: {temp_dir}")
    
    try:
        # Save and play the audio
        wav_path = save_and_play_audio(signal_normalized, metadata['sample_rate'], temp_dir)
        
        # Wait for user input before continuing
        input("\nPress Enter when done listening to continue...")
        
    except Exception as e:
        print(f"Audio playback error: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Test files - replace with your actual file paths
    pcm_file = "/Volumes/Work Drive/raw_audio_cache/raw_audio/D006615/2024/05/29/20240529_05_15_50_000000_rain_000.bin"
    alac_file = "/Volumes/Work Drive/raw_audio_cache/raw_audio/D006615/2025/01/20/20250120_14_48_00_000000_rain_032.bin"
    
    # Test PCM file
    print("\n=== Testing PCM File ===")
    visualize_and_play_audio(pcm_file, force_file_type='pcm')
    
    # Test ALAC file
    print("\n=== Testing ALAC File ===")
    visualize_and_play_audio(alac_file, force_file_type='alac')

if __name__ == "__main__":
    main() 
