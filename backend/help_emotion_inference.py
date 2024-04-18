import os
import pandas as pd
import torch
from PIL import Image
import torchaudio
from torchvision import transforms
import matplotlib.pyplot as plt
import time
# from pydub import AudioSegment

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((64, 862)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3, :, :])
])

# import os
# os.environ['TORCHAUDIO_BACKEND'] = 'soundfile'

# model.to("cuda")

# def convert_mp3_to_wav(mp3_file, output_folder):
#     # Load the MP3 file
#     audio = AudioSegment.from_mp3(mp3_file)
    
#     # Define the output filename
#     output_filename = os.path.splitext(os.path.basename(mp3_file))[0] + '.wav'
    
#     # Define the output path
#     output_path = os.path.join(output_folder, output_filename)
    
#     # Export the audio to WAV format
#     audio.export(output_path, format="wav")
    
#     return output_path

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
    image_path = f'images/my2_new_testing.png'

    plt.imsave(image_path, spectrogram_tensor.log2().numpy(), cmap='viridis')
    return image_path


def main():
    # wav_file = convert_mp3_to_wav("./scream_help.mp3", "output")
    # print(wav_file)
    # Load the audio

    model = torch.load("./models/Resnet34_Model_Help_Emotion_2024-04-13-04-14-27.pt", map_location=torch.device('cpu'))
    model.eval()

    file_path = './4_emotion.wav'

    audio, sample_rate = torchaudio.load(file_path)

    # Transform audio to an image and save it
    image_path = transform_data_to_image(audio, sample_rate)

    # Load the saved image and apply transformations
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make predictions using the model
    model.eval()
    with torch.no_grad():
        # start1 = time.time()
        outputs = model(image.to(device))
        # end1 = time.time()

    predict = outputs.argmax(dim=1).cpu().detach().numpy().ravel()[0]
    # print(outputs[0][1].cpu().numpy())


    # if predict == 0:
    #     pass
    #     print("No Scream is detected")
    # else:
    #     print("Scream is detected")
        # avg_scream += outputs[0][1].cpu().numpy()
        # max_scream = max(max_scream, outputs[0][1].cpu().numpy())
        # count += 1

            # print(f"This is total time {end1-start1}")
    # print(count)
    # print(f"This is avg {avg_scream}")
    # print(f"This is max {max_scream}")

    print(predict)

if __name__ == "__main__":
    main()


