import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def generate_clean_mel_spectrogram(wav_file, output_file, fixed_duration=4.0, img_size=224):
    """
    This function generates a greyscale mel spectrogram from a .wav file then saves it as .png

    Args:
    - wav_file: Path to input .wav file
    - output_file: Path to output .png image
    - fixed_duration (float): Duration in seconds
    - img_size: Size for the output image
    """
    try:
        sr = 44100  # Sampling rate
        y, _ = librosa.load(wav_file, sr=sr)
        
        # Pad audio if it's not 4s long
        max_length = int(fixed_duration * sr)
        if len(y) < max_length:
            y = np.pad(y, (0, max_length - len(y)), 'constant')
        else:
            y = y[:max_length]
        
        # Generate mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=128, 
            fmax=sr/2, 
            n_fft=2048, 
            hop_length=512
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Normalize the spectrogram to [0, 1]
        S_dB_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
        
        # Save the spectrogram image
        plt.figure(figsize=(img_size/100, img_size/100), dpi=100)
        librosa.display.specshow(S_dB_norm, sr=sr, fmax=sr/2, cmap="gray")
        plt.axis('off')  # Remove axis labels and ticks
        plt.tight_layout(pad=0)  # Remove padding
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"Error generating spectrogram for {wav_file}: {e}")

def process_audio_files(input_dir, output_dir):
    """
    Takes dir of .wav files as input_dir and saves spectrogram images to the output dir.

    Args:
    - input_dir: Input dir of .wav files.
    - output_dir: Output dir of .png files
    """
    os.makedirs(output_dir, exist_ok=True)
    wav_files = glob.glob(os.path.join(input_dir, '*.wav')) # Find all .wave file paths
    
    for wav_file in wav_files: # Generate spectrogram and save to output dir as same name
        file_name = os.path.basename(wav_file).replace('.wav', '.png')
        output_file = os.path.join(output_dir, file_name)
        generate_clean_mel_spectrogram(wav_file, output_file)


# Input directories
distress_wav_dir = 'processed_audio/distress/' # Distress class
nondistress_wav_dir = 'processed_audio/nondistress/' # Non-distress class

# Output directories
distress_img_dir = 'spectrograms/distress/'
nondistress_img_dir = 'spectrograms/nondistress/'

# Step 1: Convert .wav files to grayscale spectrogram images

# Generate images from audio files 
process_audio_files(distress_wav_dir, distress_img_dir) # Positive dataset
process_audio_files(nondistress_wav_dir, nondistress_img_dir) # Negative dataset

# Step 2: Split data into Training, Validation, and Test sets
data_dir = 'spectrograms/' # Training Data Dir

img_height, img_width = 224, 224
batch_size = 32

# Split: 70% training, 30% (validation + test)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

# Training data generator
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='binary',
    subset='training',
    shuffle=True
)

# Split: 15% validation and 15% test
datagen_val_test = ImageDataGenerator(rescale=1./255, validation_split=0.5)

# Validation data generator
validation_generator = datagen_val_test.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='binary',
    subset='validation',
    shuffle=True,
    seed=42
)

# Test data generator
test_generator = datagen_val_test.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=1,  # Batch_size=1 gets individual predictions
    color_mode='grayscale',
    class_mode='binary',
    subset='training',
    shuffle=False,
    seed=42
)

# Step 3: Train CNN Model

# Define the CNN model architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile the model
model.compile(optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

# Train the model
epochs = 50
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs
)

# Step 4: Evaluate the Model 

# Plot Training and Validation Accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.show()

# Compute Confusion Matrix
# This tests the model for the True Positives & True Negatives / False Positives & False Negatives

# Predict on the test data
test_generator.reset()
predictions = model.predict(test_generator, steps=test_generator.samples)
predicted_classes = (predictions > 0.5).astype(int).flatten()

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
cm = confusion_matrix(true_classes, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(6,6))
disp.plot(ax=ax)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Save the trained model so it can be used with the classifier
model.save('audio_classification_model.keras')
