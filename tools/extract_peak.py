import os
from pydub import AudioSegment, effects
import argparse

def find_loudest_peak(audio, exclude_start_ms=250, exclude_end_ms=250):
    # Takes in audio object and finds the timestamp of the loudest peak in the audio

    total_length_ms = len(audio)
    start_time_ms = exclude_start_ms
    end_time_ms = total_length_ms - exclude_end_ms

    if end_time_ms <= start_time_ms:
        raise ValueError("Audio duration is too short")

    relevant_audio = audio[start_time_ms:end_time_ms] # Extract the relevant segment for peak detection
    normalized_audio = effects.normalize(relevant_audio) # Normalize audio
    
    peak_amplitude = normalized_audio.max # Find loudest peak
    peak_positions = [
        i for i, sample in enumerate(normalized_audio.get_array_of_samples())
        if sample == peak_amplitude or sample == -peak_amplitude
    ]

    # Default to start if no peak found
    if not peak_positions:
        return start_time_ms

    # Use first peak position
    peak_position = peak_positions[0]

    # Convert the peak position to time in milliseconds
    duration_per_sample_ms = 1000 / (normalized_audio.frame_rate * normalized_audio.channels)
    peak_time_ms = start_time_ms + int(peak_position * duration_per_sample_ms)

    return peak_time_ms

def extract_segment(audio, peak_time_ms, duration_ms=2000):
    # Takes audio segment and trims it around the peak_audio timestamp (2s before it and 2s after it)
    start_time = max(peak_time_ms - duration_ms, 0)
    end_time = min(peak_time_ms + duration_ms, len(audio))
    return audio[start_time:end_time]

def normalize_audio(audio, target_dBFS=-20.0):
    # Normalizes audio to the target dBFS
    return effects.normalize(audio, headroom=target_dBFS)

def process_audio(input_path, output_path, target_dBFS=-20.0):
    """
    Processes audio file to extract the segment around the loudest peak.

    Args:
        input_path: Input .wav file
        output_path: Output .wav file
        target_dBFS: dbFS to normalize to
    """
    try:
        audio = AudioSegment.from_file(input_path) # Load audio file
    except Exception as e:
        print(f"Error: Unable to load {input_path}: {e}")
        return

    if len(audio) < 500:  # Less than 0.5 seconds
        print(f"Error: Audio file '{input_path}' is too short")
        return

    try:
        # Find loudest peak, exclude first 250ms and last 250ms
        peak_time_ms = find_loudest_peak(audio, exclude_start_ms=250, exclude_end_ms=250)
        print(f"Loudest peak in '{input_path}' found at {peak_time_ms} ms")
    except ValueError as ve:
        print(f"Skipping file '{input_path}': {ve}")
        return

    # Extract peak
    segment = extract_segment(audio, peak_time_ms, duration_ms=2000)

    # Normalize audio
    normalized_segment = normalize_audio(segment, target_dBFS=target_dBFS)

    # Export extracted audio
    try:
        output_format = os.path.splitext(output_path)[1][1:]  # Retain same format as input format
        normalized_segment.export(output_path, format=output_format)
        print(f"Extracted segment saved to '{output_path}'\n")
    except Exception as e:
        print(f"Error saving '{output_path}': {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Extracts 4-second segment around the loudest peak in an audio file"
    )
    parser.add_argument("input", help="Path to the input audio file")
    parser.add_argument("output", help="Path to the output audio file")
    parser.add_argument(
        "--target_dBFS",
        type=float,
        default=-20.0,
        help="Target dBFS for normalization (default: -20.0)"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Input file '{args.input}' does not exist")
        return

    process_audio(args.input, args.output, target_dBFS=args.target_dBFS)

if __name__ == "__main__":
    main()
