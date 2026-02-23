import os
import sys
import argparse

from audio_processing_tools.parse import create_dict_by_kaitai

def test_audio_parser(file_path: str):
    """
    Test the audio parser by reading a file and printing its header information
    
    Parameters
    ----------
    file_path : str
        Path to the audio file to test
    """
    # Read the binary file
    with open(file_path, 'rb') as f:
        audio_data = f.read()
    
    # Parse the file
    try:
        parsed_data = create_dict_by_kaitai(audio_data)
        
        # Create a copy without the audio data for cleaner printing
        header_info = parsed_data.copy()
        header_info.pop('audio')  # Remove the audio data
        
        print("Successfully parsed audio file!")
        print("\nHeader Information:")
        print("-" * 50)
        for key, value in header_info.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error parsing file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse one or more audio files.")
    parser.add_argument("path", help="Path to a file or folder containing audio files")
    args = parser.parse_args()

    input_path = args.path

    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist.")
        sys.exit(1)

    # If it's a folder, parse all files in it
    if os.path.isdir(input_path):
        print(f"Parsing all files in folder: {input_path}")
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(".bin"):
                    file_path = os.path.join(root, file)
                    test_audio_parser(file_path)
    else:
        # If it's a single file, just parse it
        test_audio_parser(input_path)

# if __name__ == "__main__":
#    if len(sys.argv) < 2:
#        print("Usage: python script.py <file_path>")
#        sys.exit(1)
# 
#    file_path = sys.argv[1]
#    test_audio_parser(file_path)
