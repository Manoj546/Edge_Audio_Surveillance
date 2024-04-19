import whisper
from transformers import pipeline
import wave



temp_audio_file_path = "./backend/4_emotion.wav"

# Load models
model = whisper.load_model("base")
print("Whisper Model Loaded!")
sentiment_analysis = pipeline("sentiment-analysis", framework="pt", model="SamLowe/roberta-base-go_emotions")

# Function to analyze sentiment
def analyze_sentiment(text):
    results = sentiment_analysis(text)
    sentiment_results = {result['label']: result['score'] for result in results}
    return sentiment_results

# Function to get sentiment emoji
def get_sentiment_emoji(sentiment):
    emoji_mapping = {
        "disappointment": "😞",
        "sadness": "😢",
        "annoyance": "😠",
        "neutral": "😐",
        "disapproval": "👎",
        "realization": "😮",
        "nervousness": "😬",
        "approval": "👍",
        "joy": "😄",
        "anger": "😡",
        "embarrassment": "😳",
        "caring": "🤗",
        "remorse": "😔",
        "disgust": "🤢",
        "grief": "😥",
        "confusion": "😕",
        "relief": "😌",
        "desire": "😍",
        "admiration": "😌",
        "optimism": "😊",
        "fear": "😨",
        "love": "❤️",
        "excitement": "🎉",
        "curiosity": "🤔",
        "amusement": "😄",
        "surprise": "😲",
        "gratitude": "🙏",
        "pride": "🦁"
    }
    return emoji_mapping.get(sentiment, "")

# Function to display sentiment results
def display_sentiment_results(sentiment_results, option):
    sentiment_text = ""
    for sentiment, score in sentiment_results.items():
        emoji = get_sentiment_emoji(sentiment)
        if option == "Sentiment Only":
            sentiment_text += f"{sentiment} {emoji}\n"
        elif option == "Sentiment + Score":
            sentiment_text += f"{sentiment} {emoji}: {score}\n"
    return sentiment_text

# Function to perform inference
def inference(ans, sentiment_option):
    sentiment_results = analyze_sentiment(ans)
    sentiment_output = display_sentiment_results(sentiment_results, sentiment_option)
    return sentiment_output

def count_help_occurrences(text):
    return text.lower().count("help")


audio_path = 'backend/4_emotion.wav'
audio = wave.open(audio_path, 'rb')
result = model.transcribe(audio_path)
ans = result["text"]
print(result['segments'][0])
sentiment_option = "Sentiment + Score"
sentiment_output_value = inference(ans, sentiment_option)
print(sentiment_output_value)