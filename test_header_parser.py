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
    # Replace with your audio file path
    # print("new file type:")
    # file_path = "/Volumes/Work Drive/raw_audio_cache/raw_audio/D006615/2025/01/20/20250120_14_48_00_000000_rain_032.bin"
    # test_audio_parser(file_path) 
    print("new file type:")
    file_path = "/Volumes/Work Drive/raw_audio_cache/raw_audio/D006534/2025/02/16/20250216_23_07_00_000000_rain_b26.bin"
    test_audio_parser(file_path)
    print("old file type:")
    file_path = "/Volumes/Work Drive/raw_audio_cache/raw_audio/D006615/2024/05/29/20240529_05_15_50_000000_rain_000.bin"
    test_audio_parser(file_path) 
