import requests
import time
import pyaudio
import wave

url = 'http://192.168.115.147:5000/data'


# Function to start recording
def start_recording(output_filename, duration_seconds):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = duration_seconds
    WAVE_OUTPUT_FILENAME = output_filename

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* Recording started")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* Recording finished")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# Example usage: Record for 10 seconds and save to "output.wav"
def listening():
    # while(1):
        start_recording("output.wav", 6)
        print("Done")
        with open("output.wav", 'rb') as file:
            files = {'messageFile': file}
            response = requests.post(url, files=files)

        print(response.status_code)
        print(response.text)

listening()