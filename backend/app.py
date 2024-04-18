from flask import Flask, request, jsonify
import socketio

import pandas as pd
import torch
from PIL import Image
import torchaudio
from torchvision import transforms
import matplotlib.pyplot as plt
import time

import os
import wave
import numpy as np
import openwakeword
from openwakeword.model import Model
from scipy.signal import resample
import time 

# Initialize Flask app
app = Flask(__name__)

# Initialize Socket.IO
sio = socketio.Server(cors_allowed_origins='*')
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)


#Scream detaction functions
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((64, 862)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3, :, :])
])

def pad_waveform(waveform, target_length):
    num_channels, current_length = waveform.shape

    if current_length < target_length:
        padding = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    return waveform

def transform_data_to_image(audio, sample_rate):
    audio = pad_waveform(audio, 441000)

    spectrogram_tensor = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64, n_fft=1024)(audio)[0] + 1e-10

    image_path = f'my2_new_testing.png'

    plt.imsave(image_path, spectrogram_tensor.log2().numpy(), cmap='viridis')
    return image_path
    
model = torch.load("./models/Resnet34_Model.pt", map_location=torch.device('cpu'))

def scream_detection_ml(final_path):
    print("Inside scream")
    pred = "Scream Not Detected"
    model.eval()
    audio, sample_rate = torchaudio.load(final_path)
    image_path = transform_data_to_image(audio, sample_rate)
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  
    model.eval()
    with torch.no_grad():
        start1 = time.time()
        outputs = model(image.to(device))
        end1 = time.time()

    predict = outputs.argmax(dim=1).cpu().detach().numpy().ravel()[0]
    print(outputs.cpu().numpy())
    
    scream_predict = outputs[0][1].cpu().numpy()

    if predict == 0:
        print("No Scream is detected")
    else:
        pred = "Scream Detected"
        print("Scream is detected")
        
    return predict, scream_predict

    # print(f"This is total time {end1-start1}")
    
#End of scream detection code


#Start of openwake word

# Load pre-trained openwakeword model
openwakeword.utils.download_models()

# Initialize openWakeWord model
owwModel = Model(wakeword_models=['./models/Help_me.tflite', './models/Help_us.tflite', 
                                  './models/please_help_me.tflite', './models/Sombody_Help.tflite',
                                  './models/Someone_Help_me.tflite', 'models/help.tflite',
                                  './models/please_help.tflite', './models/Someone_Help.tflite'], inference_framework='tflite')
n_models = len(owwModel.models.keys())



def key_word_function(final_path):
    
    count = 0
    keyflag = False

    wav_file = wave.open(final_path, 'rb')

    
    def resample_audio(audio_data, new_rate=16000):
        original_rate = wav_file.getframerate()
        resampled_data = resample(audio_data, int(len(audio_data) * float(new_rate) / original_rate))
        return resampled_data.astype(np.int16)

    print(f"Listening for wakewords in {final_path}...")

    # Read and process audio data from the WAV file
    CHUNK = 1024
    audio_data = wav_file.readframes(CHUNK)
    pred = "Help Not Detected"

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
        if any(score >= 0.1 for score in prediction.values()):
            keyflag = True
            count += 1
            # score_above_threshold_counter += 1

        # Print results for each chunk
        # print(f"Prediction for chunk: {prediction}")

        # Read next chunk of audio data
        audio_data = wav_file.readframes(CHUNK)
        # totaltime += (end1-start1)
        # print(f"Inference Time {end1-start1}")

    # Close the WAV file
    wav_file.close()
    return keyflag, count

#End of openwakeword


# Define a route to handle POST requests
@app.route('/process_data', methods=['POST'])
@app.route('/data', methods=['POST'])
def process_data():
    print(request.files['messageFile'])

    audio_file = request.files['messageFile']
    
    # Save the audio file
    final_path = "./static/received_audio.wav"

    audio_file.save(final_path)
    print("Saved")
    
    keyflag = False
    scream_predict, scream_val = scream_detection_ml(final_path)
    keyflag, count = key_word_function(final_path)
    print(f'This is keyflag: {keyflag}')
    
    situation = ''

    if scream_predict == 1 and keyflag == True:
        situation = "Critical Situation"
    elif scream_val > 2 and keyflag == False:
        situation = 'Critical Situation'
    elif scream_predict == 0 and keyflag == True: ##Emotion
        situation = "Checking for emotion"
        ##if that emotion is true: print("Critical Situation")
        ##else: print("No Crictical Situation")
    elif count > 1:
        situation = "Critical Situation"
    else:
        situation = "Not a Critical Situation"
        
    scream_str = ['Scream Detected' if scream_predict == 1 else 'Scream Not Detected']
    key_str = ['Help Detected' if keyflag == True else 'Help Not Detected']

    result = {'text': audio_file.filename, 'scream': scream_str[0], 'key_word': key_str[0], 'situation': situation}
    sio.emit('result', result)
    print("Emitted result:", result)  # Add this line
    return jsonify({'result': result})


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0')
