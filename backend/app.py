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
    
model = torch.load("./models/Resnet34_Model.pt", map_location=torch.device('cuda'))

def scream_detection_ml(final_path):
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

    if predict == 0:
        print("No Scream is detected")
    else:
        pred = "Scream Detected"
        print("Scream is detected")
        
    return pred

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

def resample_audio(audio_data, new_rate=16000):
    original_rate = wav_file.getframerate()
    resampled_data = resample(audio_data, int(len(audio_data) * float(new_rate) / original_rate))
    return resampled_data.astype(np.int16)

def key_word_function(final_path):

    wav_file = wave.open(final_path, 'rb')

    # Generate output string header
    print("\n\n")
    print("#" * 100)
    print(f"Listening for wakewords in {final_path}...")
    print("#" * 100)
    print("\n" * (n_models * 3))

    # Read and process audio data from the WAV file
    CHUNK = 1028
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
        if any(score >= 0.3 for score in prediction.values()):
            print("Help Detected")
            pred = "Help Detected"
            # score_above_threshold_counter += 1
            break

        # Print results for each chunk
        # print(f"Prediction for chunk: {prediction}")

        # Read next chunk of audio data
        audio_data = wav_file.readframes(CHUNK)
        # totaltime += (end1-start1)
        # print(f"Inference Time {end1-start1}")

    # Close the WAV file
    wav_file.close()
    return pred

#End of openwakeword


# Define a route to handle POST requests
@app.route('/data', methods=['POST'])
@app.route('/process_data', methods=['POST'])
def process_data():
    if 'text' not in request.form or 'audio' not in request.files:
        return 'Missing text or audio data', 400

    text = request.form['text']
    audio_file = request.files['audio']
    final_path = 'static/' + audio_file.filename
    print("Received data:", text, audio_file.filename)  # Add this line

    audio_file.save(final_path)
    
    result = {'text': text, 'audio_file': audio_file.filename, 'scream': scream_detection_ml(final_path), 'key_word': key_word_function(final_path)}
    sio.emit('result', result)
    print("Emitted result:", result)  # Add this line
    return jsonify({'result': result})


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0')
