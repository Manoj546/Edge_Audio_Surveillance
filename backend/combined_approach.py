import os
import pandas as pd
import torch
from PIL import Image
import torchaudio
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import wave
import numpy as np
import openwakeword
from openwakeword.model import Model
from scipy.signal import resample
# from pydub import AudioSegment

# Load pre-trained openwakeword model
openwakeword.utils.download_models()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((64, 862)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3, :, :])
])


#help_emotion.wav
#my_help_me.wav
#scream_help.wav

def pad_waveform(waveform, target_length):
    num_channels, current_length = waveform.shape

    if current_length < target_length:
        # Calculate the amount of padding needed
        padding = target_length - current_length
        # Pad the waveform with zeros on the right side
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    return waveform

# Define a function to transform audio data into images
def transform_data_to_image(audio, sample_rate):
    # Pad waveform to a consistent length of 44100 samples
    audio = pad_waveform(audio, 441000)

    spectrogram_tensor = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64, n_fft=1024)(audio)[0] + 1e-10

    # Save the spectrogram as an image
    image_path = f'my2_new_testing.png'

    plt.imsave(image_path, spectrogram_tensor.log2().numpy(), cmap='viridis')
    return image_path



def main():

    file_path = './whatsapp_help.wav'

    scream_model = torch.load("./models/Resnet34_Model.pt", map_location=torch.device('cpu'))

    keyword_model = Model(wakeword_models=['./models/Help_me.tflite', './models/Help_us.tflite', 
                                  './models/please_help_me.tflite', './models/Sombody_Help.tflite',
                                  './models/Someone_Help_me.tflite', 'models/help.tflite',
                                  './models/please_help.tflite', './models/Someone_Help.tflite'], inference_framework='tflite')
    
    n_models = len(keyword_model.models.keys())
    # load_end = time.time()
    scream_model.eval()
    # print(f"Total loading time {load_start-load_end}")

    audio, sample_rate = torchaudio.load(file_path)

    # Transform audio to an image and save it
    image_path = transform_data_to_image(audio, sample_rate)

    # Load the saved image and apply transformations
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make predictions using the model
    scream_model.eval()
    with torch.no_grad():
        # start1 = time.time()
        outputs = scream_model(image.to(device))
        # end1 = time.time()

    scream_predict = outputs.argmax(dim=1).cpu().detach().numpy().ravel()[0]
    # print(outputs.cpu().numpy())

    if scream_predict == 0:
        print("No Scream is detected")
    else:
        print("Scream is detected")

    scream_val = outputs[0][1].cpu().numpy()



    ##Keyword Prediction##

    def resample_audio(audio_data, new_rate=16000):
        original_rate = wav_file.getframerate()
        resampled_data = resample(audio_data, int(len(audio_data) * float(new_rate) / original_rate))
        return resampled_data.astype(np.int16)

    count = 0
    # print("\n\nKeyword Detection\n\n")
    keyflag = False

    wav_file = wave.open(file_path, 'rb')

    CHUNK = 1028
    audio_data = wav_file.readframes(CHUNK)

    while audio_data:
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Resample audio if needed
        if wav_file.getframerate() != 16000:
            audio_array = resample_audio(audio_array)

        prediction = keyword_model.predict(audio_array)

        if any(score >= 0.2 for score in prediction.values()):
            # print("Hot Word Detected")
            keyflag = True
            count += 1
            # score_above_threshold_counter += 1
            # break

        audio_data = wav_file.readframes(CHUNK)

    wav_file.close()

    if keyflag == True:
        print("Hotword Detected")
    else:
        print("Hotword Not Detected")

    if scream_predict == 1 and keyflag == True:
        print("Critical Situation")
    elif scream_val > 2 and keyflag == False:
        print("Critical Situation")
    elif scream_predict == 0 and keyflag == True: ##Emotion
        print("Checking for emotion")
        ##if that emotion is true: print("Critical Situation")
        ##else: print("No Crictical Situation")
    elif count > 1:
        print("Critical Situation")
    else:
        print("Not a Critical Situation")


if __name__ == "__main__":
    main()
