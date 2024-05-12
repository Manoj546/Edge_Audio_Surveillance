from flask import Flask, request, jsonify
import socketio

import torch
from PIL import Image
import torchaudio
from torchvision import transforms
import matplotlib.pyplot as plt
import time

import os
import wave
import numpy as np
# import openwakeword
# from openwakeword.model import Model
from scipy.signal import resample
import time 

import whisper
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Initialize Socket.IO
sio = socketio.Server(cors_allowed_origins='*')
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)


#Scream detaction functions
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
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

scream_load_start = time.time()
model = torch.load("./models/Resnet34_Model.pt", map_location=torch.device(device))
scream_load_end = time.time()

print(f" Scream load time : {scream_load_end - scream_load_start}")

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


key_load_start = time.time()
key_model = whisper.load_model("base")
key_load_end = time.time()
print(f"Keyword Load Time : {key_load_end - key_load_start}")
key_model = key_model.to(device)



def hindi_key_word_function(file_path):
    # Set up automatic speech recognition pipeline
    hindi_flag = False
    hindi_emotion = False

    transcribe_hindi = pipeline(
        task="automatic-speech-recognition",
        model="vasista22/whisper-hindi-small",
        chunk_length_s=30,
        device=device
    )

    # Configure the model for Hindi transcription
    transcribe_hindi.model.config.forced_decoder_ids = transcribe_hindi.tokenizer.get_decoder_prompt_ids(language="hi", task="transcribe")
    transcription_result = transcribe_hindi(file_path)

    transcribed_text = transcription_result["text"]
    print(transcribed_text)

    count_bachao = transcribed_text.count("बचाओ")
    count_bachav = transcribed_text.count("बचाव")

    if count_bachao > 2 or count_bachav > 2:
        hindi_flag = True
        hindi_emotion = True

    print(f"This is count of bachao {count_bachao}")
    print(f"This is count of bachav {count_bachav}")


    if "बचाओ" in transcribed_text or "बचाव" in transcribed_text:
        hindi_flag = True
    #     print("The words 'बचाओ' (bachao) or 'बचाव' (bachav) are present in the transcription.")
    # else:
    #     print("Neither 'बचाओ' (bachao) nor 'बचाव' (bachav) are present in the transcription.")

    
    # print(sentiment_output_value)
    return hindi_emotion, hindi_flag

def key_word_function(file_path):

    emotionflag = False
    keyflag = False

    # print("Whisper Model Loaded!")
    sentiment_analysis = pipeline("sentiment-analysis", framework="pt", model="SamLowe/roberta-base-go_emotions")

    def analyze_sentiment(text):
        results = sentiment_analysis(text)
        sentiment_results = {result['label']: result['score'] for result in results}
        return sentiment_results
    

    # Function to display sentiment results
    def display_sentiment_results(sentiment_results, option):
        sentiment_text = ""
        for sentiment, score in sentiment_results.items():
            if option == "Sentiment":
                sentiment_text += f"{sentiment}\n"
        # print(f"This is sentiment text {sentiment_text}")
        return sentiment_text

    # Function to perform inference
    def inference(ans, sentiment_option):
        sentiment_results = analyze_sentiment(ans)
        sentiment_output = display_sentiment_results(sentiment_results, sentiment_option)
        return sentiment_output

    start = time.time()
    result = key_model.transcribe(file_path)
    end = time.time()
    ans = result["text"]
    print(f"This is the text transcribed {ans}")

    if ans.lower().count("help") >= 1:
        keyflag = True
    if ans.lower().count("help") >= 2:
        print("Help count is > 2")
        emotionflag = True
        count = ans.lower().count("help")

    sentiment_option = "Sentiment"
    sentiment_output_value = inference(ans, sentiment_option)
    print(f"This is the sentiment {sentiment_output_value}")

    if sentiment_output_value in ['sadness', 'anger', 'fear']:
        emotionflag = True

    
    # print(sentiment_output_value)
    return emotionflag, keyflag




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
    scream_start = time.time()
    scream_predict, scream_val = scream_detection_ml(final_path)
    scream_end = time.time()
    key_start = time.time()
    emotionflag, keyflag = key_word_function(final_path)
    key_end = time.time()
    print(f'This is keyflag: {keyflag}')

    hindi_emotion, hindi_flag = hindi_key_word_function(final_path)
    # if hindi_flag:
    #     print("Hindi Wakeword Detected")


    
    situation = ''

    if scream_predict == 1 and ( keyflag == True or hindi_flag == True ):
        situation = "Critical Situation"
    elif scream_val >= 1.3:
        situation = "Critical Situation"
    elif keyflag == True and hindi_flag == True:
        situation = 'Critical Situations'
    elif scream_predict == 0 and (keyflag == True or hindi_flag == True): ##Emotion
        print("Checking for emotion")
        if emotionflag == True or hindi_emotion == True:
            situation = "Critical Situation"
        else:
            situation = "Not a critical situation"

    else:
        situation = "Not a critical situation"
        
    scream_str = ['Scream Detected' if scream_predict == 1 else 'Scream Not Detected']
    key_str = ['Help Detected' if keyflag == True else 'Help Not Detected']
    hindi_str = ['Bachao Detected' if hindi_flag == True else 'Bachao Not Detected']

    print(f"Scream time : {scream_end - scream_start}")
    print(f"Keyword Time : {key_end - key_start}")

    result = {'text': audio_file.filename, 'scream': scream_str[0], 'key_word': key_str[0], 'hindi_str': hindi_str, 'situation': situation}
    sio.emit('result', result)
    print("Emitted result:", result)
    return jsonify({'result': result})


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0')