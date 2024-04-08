import os
import wave
import numpy as np
import openwakeword
from openwakeword.model import Model
from scipy.signal import resample
import time 

# Load pre-trained openwakeword model
openwakeword.utils.download_models()

# Initialize openWakeWord model
owwModel = Model(wakeword_models=['./models/Help_me.tflite', './models/Help_us.tflite', 
                                  './models/please_help_me.tflite', './models/Sombody_Help.tflite',
                                  './models/Someone_Help_me.tflite', 'models/help.tflite',
                                  './models/please_help.tflite', './models/Someone_Help.tflite'], inference_framework='tflite')
n_models = len(owwModel.models.keys())

# Directory containing WAV files
wav_directory = "./dataset/help_set"  # Change this to your directory path

# Counter for detecting scores above 0.05
score_above_threshold_counter = 0

def resample_audio(audio_data, new_rate=16000):
    original_rate = wav_file.getframerate()
    resampled_data = resample(audio_data, int(len(audio_data) * float(new_rate) / original_rate))
    return resampled_data.astype(np.int16)

# totaltime = 0

# Iterate over WAV files in the directory
for filename in os.listdir(wav_directory):
    if filename.endswith(".wav"):
        wav_file_path = os.path.join(wav_directory, filename)

        # Load WAV file
        wav_file = wave.open(wav_file_path, 'rb')

        # Generate output string header
        print("\n\n")
        print("#" * 100)
        print(f"Listening for wakewords in {filename}...")
        print("#" * 100)
        print("\n" * (n_models * 3))

        # Read and process audio data from the WAV file
        CHUNK = 1028
        audio_data = wav_file.readframes(CHUNK)

        while audio_data:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Resample audio if needed
            if wav_file.getframerate() != 16000:
                audio_array = resample_audio(audio_array)

            # Feed audio to openWakeWord model
            # start1 = time.time()
            prediction = owwModel.predict(audio_array)
            # end1 = time.time()

            # Check if any score is above 0.05
            if any(score >= 0.2 for score in prediction.values()):
                score_above_threshold_counter += 1
                break

            # Print results for each chunk
            # print(f"Prediction for chunk: {prediction}")

            # Read next chunk of audio data
            audio_data = wav_file.readframes(CHUNK)
            # totaltime += (end1-start1)
            # print(f"Inference Time {end1-start1}")

        # Close the WAV file
        wav_file.close()

# Print total count of scores above 0.05
print(f"Total count of scores above 0.5: {score_above_threshold_counter}")
# print(f"Final inference time {totaltime}")
