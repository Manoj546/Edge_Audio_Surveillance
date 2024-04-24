from flask import Flask, request, jsonify
import socketio

import torch
from PIL import Image
import torchaudio
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import numpy as np
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


#Start of whisper model
key_model = whisper.load_model("base")
key_model = key_model.to(device)

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


    # Transcribe audio and perform sentiment analysis
    # print(f"Transcribing Audio: {filename}")
    start = time.time()
    result = key_model.transcribe(file_path)
    end = time.time()
    ans = result["text"]
    print(f"This is the text transcribed {ans}")
    # print(result['segments'])
    count = 0
    # print(ans)
    print(f"Keyword total time {end-start}")

    # Check if "help" is present in the text
    # if count_help_occurrences(ans) > 0:
    #     help_count += 1

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




# def key_word_function(final_path):
    
#     count = 0
#     keyflag = False

#     wav_file = wave.open(final_path, 'rb')

    
#     def resample_audio(audio_data, new_rate=16000):
#         original_rate = wav_file.getframerate()
#         resampled_data = resample(audio_data, int(len(audio_data) * float(new_rate) / original_rate))
#         return resampled_data.astype(np.int16)

#     print(f"Listening for wakewords in {final_path}...")

#     # Read and process audio data from the WAV file
#     CHUNK = 1024
#     audio_data = wav_file.readframes(CHUNK)
#     pred = "Help Not Detected"

#     while audio_data:
#         audio_array = np.frombuffer(audio_data, dtype=np.int16)

#         # Resample audio if needed
#         if wav_file.getframerate() != 16000:
#             audio_array = resample_audio(audio_array)

#         # Feed audio to openWakeWord model
#         # start1 = time.time()
#         prediction = owwModel.predict(audio_array)
#         # end1 = time.time()

#         # Check if any score is above 0.05
#         if any(score >= 0.1 for score in prediction.values()):
#             keyflag = True
#             count += 1
#             # score_above_threshold_counter += 1

#         # Print results for each chunk
#         # print(f"Prediction for chunk: {prediction}")

#         # Read next chunk of audio data
#         audio_data = wav_file.readframes(CHUNK)
#         # totaltime += (end1-start1)
#         # print(f"Inference Time {end1-start1}")

#     # Close the WAV file
#     wav_file.close()
#     return keyflag, count

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
    emotionflag, keyflag = key_word_function(final_path)
    print(f'This is keyflag: {keyflag}')
    
    situation = ''

    if scream_predict == 1 and keyflag == True:
        situation = "Critical Situation"
    elif scream_val > 2 and keyflag == False:
        situation = "Critical Situation"
    elif scream_predict == 0 and keyflag == True: ##Emotion
        print("Checking for emotion")
        if emotionflag == True:
            situation = "Critical Situation"
        else:
            situation = "Not a critical situation"
        ##if that emotion is true: print("Critical Situation")
        ##else: print("No Crictical Situation")
    else:
        situation = "Not a critical situation"
        
    scream_str = ['Scream Detected' if scream_predict == 1 else 'Scream Not Detected']
    key_str = ['Help Detected' if keyflag == True else 'Help Not Detected']

    result = {'text': audio_file.filename, 'scream': scream_str[0], 'key_word': key_str[0], 'situation': situation}
    sio.emit('result', result)
    print("Emitted result:", result)  # Add this line
    return jsonify({'result': result})


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0')