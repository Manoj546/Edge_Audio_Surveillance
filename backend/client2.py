import threading
import requests
import pyaudio
import wave
import time


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
def listening(url):
    # while(1):
    print('started')
    # start_recording("output.wav", 6)
    # print("Done")
    with open("./static/received_audio.wav", 'rb') as file:
        files = {'messageFile': file}
        response = requests.post(url, files=files)

    print(response.status_code)

    print(f"Response from {url}: {response.text}")


#End


# Define the URLs and data for the POST requests

url1 = "http://192.168.247.147:5000/data"
url2 = "http://192.168.247.147:5000/data1"
url3 = "http://192.168.247.147:5000/data2"
url4 = "http://192.168.247.34:5000/data1"
url5 = "http://192.168.247.147:5000/data2"



# Create threads for sending POST requests
# thread1 = threading.Thread(target=listening, args=(url2,))
# thread2 = threading.Thread(target=listening, args=(url3,))

start_time = time.time()
# Start the threads
# thread1.start()
# thread2.start()
listening(url1)

# Wait for both threads to finish
# thread1.join()
# thread2.join()

end_time = time.time()

print(f"Total time taken: {end_time - start_time} seconds")

print("Both POST requests sent successfully.")
