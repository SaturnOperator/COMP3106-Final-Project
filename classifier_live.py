import os
import numpy as np
import sounddevice as sd
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from collections import deque
import time 

MODEL_PATH = 'audio_classification_model.keras' # Trained model path

# Audio Parameters
SAMPLE_RATE = 44100 # Sampling rate (Hz)
BUFFER_DURATION = 4 # Audio buffer (seconds)
BUFFER_SIZE = SAMPLE_RATE * BUFFER_DURATION  # Total samples in buffer

# Spectrogram Parameters
N_MELS = 128 # Number of Mel bands
FMAX = SAMPLE_RATE / 2 # Maximum freq (Hz)
N_FFT = 2048 # FFT window size
HOP_LENGTH = 512 # Hop length for STFT

# Image Parameters
IMG_HEIGHT = 224 
IMG_WIDTH = 224 
IMG_CHANNELS = 1 # Only 1 channel because images are greyscale

PREDICTION_THRESHOLD = 0.5 # Threshold for distress detection

# Open CV Parameters
WINDOW_NAME = 'Distress Detection'
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_COLOR_DISTRESS = (0, 0, 255) # Red to show distress
FONT_COLOR_NONE = (255, 255, 255) 
FONT_THICKNESS = 2


# Init a deque to store audio samples
audio_buffer = deque(maxlen=BUFFER_SIZE)

# Load model
print("Loading the trained model")
model = load_model(MODEL_PATH)
print("Model loaded successfully")


def audio_callback(indata, frames, time_info, status):
    # Captures mic data and puts it into the buffer
    if status:
        print(f"Audio Callback Status: {status}")
    # Flatten the incoming data and add to the buffer
    audio_buffer.extend(indata[:, 0])

def process_audio(buffer):
    # Converts audio buffer to a mel spectrogram image 

    y = np.array(buffer)

    # Generate mel spectrogram
    S = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        fmax=FMAX,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    S_dB = librosa.power_to_db(S, ref=np.max) # Get audio decibel to normalize
    S_dB_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min()) # Normalize the spectrogram to [0, 1]
    S_dB_resized = cv2.resize(S_dB_norm, (IMG_WIDTH, IMG_HEIGHT)) # Resize spectrogram to match model input (224x224)
    input_img = np.expand_dims(S_dB_resized, axis=(0, -1))

    return input_img, S_dB_resized


def predict_distress(input_img):
    # Uses trained model on the spectrogram image to predict distress, returns prediction
    prediction = 1 - model.predict(input_img)[0][0]

    # Show label based on prediction & threshold
    if prediction >= PREDICTION_THRESHOLD:
        classification = 1
    else:
        classification = 0

    return classification, prediction

def visualize_spectrogram(spectrogram_img, label, confidence, color):
    # Display spectrogram and overlay prediction using OpenCV

    # Convert the spectrogram to an image format suitable for OpenCV
    spectrogram_img = (spectrogram_img * 255).astype(np.uint8) # Rescale
    spectrogram_img = cv2.cvtColor(spectrogram_img, cv2.COLOR_GRAY2BGR) # Make greyscale

    # Create overlay label
    if label:
        text = f"{label} ({confidence:.2f})"
        cv2.putText( # Show distress and confidence score
            spectrogram_img,
            text,
            (10, 30),
            FONT,
            FONT_SCALE,
            color,
            FONT_THICKNESS,
            cv2.LINE_AA
        )
        cv2.rectangle( # Draw a red rectangle
            spectrogram_img,
            (0, 0),
            (spectrogram_img.shape[1] - 1, spectrogram_img.shape[0] - 1),
            color,
            thickness=2
        )

    cv2.imshow(WINDOW_NAME, spectrogram_img)

    if cv2.waitKey(1) & 0xFF == ord('q'): # exit the program
        print("Closing OpenCV Window...")
        cv2.destroyAllWindows()
        exit(0)

# ------------------------ Main Function ------------------------

def main():

    # Init audio stream using microphone
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE):
        print("Microphone is active, starting distress detection")
        print("[Press ctrl+C to stop]")

        label_counter = 0
        label_active = False

        try:
            while True:
                if len(audio_buffer) == BUFFER_SIZE: # Wait until buffer is full to show full image
                    input_img, spectrogram_img = process_audio(audio_buffer) # Convert the audio buffer into spectrogram
                    classification, confidence = predict_distress(input_img) # Predict distress 

                    if classification:
                        print(f"Distress detected with confidence {confidence:.2f}")
                        label_active = True
                        label_counter = 0

                    if label_active:
                        label = "Distress Detected"
                        color = FONT_COLOR_DISTRESS
                    else:
                        label = ""
                        color = FONT_COLOR_NONE

                    visualize_spectrogram(spectrogram_img, label, confidence, color) # Show the spectrogram with prediction

                else:
                    # Wait until buffer is full
                    pass
                time.sleep(0.1) # 100ms cooldown 
                label_counter += 1
                if(label_counter >= 10):
                    label_counter = 0
                    label_active = False

        except KeyboardInterrupt:
            print("\nDistress detection stopped")
        finally:
            cv2.destroyAllWindows() # Clsoe all OpenCV windows

if __name__ == "__main__":
    main()
