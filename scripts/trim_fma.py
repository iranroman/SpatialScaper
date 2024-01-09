import os
import librosa
import soundfile as sf

def trim_audio(input_path, output_path, start_time, end_time):
    # Load the audio file using librosa
    audio, sr = librosa.load(input_path, sr=None, offset=start_time, duration=end_time - start_time)
    
    # Save the trimmed audio using soundfile
    sf.write(output_path, audio, sr)

def main():
    root_dir = '/datasets/fma' #FIX THIS
    output_dir = '/datasets/fma_10sec'
    start_time = 10.0  # Start trimming from 10 seconds
    end_time = 20.0   # End trimming at 20 seconds (adjust as needed)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mp3'):
                input_path = os.path.join(root, file)
                output_subdir = os.path.relpath(root, start=root_dir)
                output_subdir_path = os.path.join(output_dir, output_subdir)
                os.makedirs(output_subdir_path, exist_ok=True)
                
                output_filename = file
                output_path = os.path.join(output_subdir_path, output_filename)
                
                trim_audio(input_path, output_path, start_time, end_time)
                print(f"Trimmed {input_path} and saved to {output_path}")

if __name__ == "__main__":
    main()
