import os
from pydub import AudioSegment
import wave

def convert_directory_to_wav(input_dir, output_dir):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through each file in the input directory
    for filename in os.listdir(input_dir):
        # if filename.endswith('.ogg') or filename.endswith('.mp4'):
            # Construct full input and output paths
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.wav')

        # Load the audio file
        audio = AudioSegment.from_file(input_path)

        # Set the desired sample rate to 44100 Hz
        audio = audio.set_frame_rate(44100)

        # Export the audio to WAV format
        audio.export(output_path, format='wav')

# Example usage:
input_directory = 'dataset/help_no_emotion'  # Replace with your input directory path
output_directory = 'dataset/help_no_emotion_wav'  # Replace with your desired output directory path
# convert_directory_to_wav(input_directory, output_directory)

# Directory containing WAV files
directory = 'dataset/help_no_emotion_wav'

# Iterate through WAV files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.wav'):
        file_path = os.path.join(directory, filename)
        with wave.open(file_path, 'rb') as wav_file:
            # Get duration of the WAV file
            duration = wav_file.getnframes() / float(wav_file.getframerate())
            if duration > 10:
                # Remove the file if duration exceeds 10 seconds
                os.remove(file_path)
                print(f"Removed {filename} because it exceeds 10 seconds.")