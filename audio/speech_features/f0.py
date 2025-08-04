"""
File: Amit Tzadok
Author: Amit Tzadok <amit.tzadok@icloud.com>
Created: 2025-07-22 11:05:31
Description: This script processes audio files to estimate fundamental frequency (F0) and compress audio segments using DCT.
"""

import wave
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks
from scipy.io import wavfile
from scipy.fftpack import dct, idct

# Load a wave file and return audio data, sample rate, and number of channels
def load_wave_file(wave_fname):
    sample_width_to_dtype = {1: np.int8, 2: np.int16, 4: np.int32}
    with wave.open(wave_fname, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        n_bytes_per_sample = wav_file.getsampwidth()
        if n_bytes_per_sample not in sample_width_to_dtype:
            raise ValueError(f"Unsupported sample width {n_bytes_per_sample}")
        raw_data = wav_file.readframes(wav_file.getnframes())
        audio_array = np.frombuffer(raw_data, dtype=sample_width_to_dtype[n_bytes_per_sample])
        if num_channels == 2:
            audio_array = audio_array.reshape(-1, 2)
            audio_array = audio_array[:, 0]  # Convert to mono
    # print(f"Loaded {wave_fname}: {audio_array.shape[0]} samples, {sample_rate} Hz, {num_channels} channels")
    # print(f"audio_array: {audio_array[:100]}...")  # Print first 10 samples for debugging
    return audio_array, sample_rate, num_channels

# Crop the waveform to a specified range in seconds
def crop_waveform(audio_array, sample_rate, seconds_range):
    if len(seconds_range) == 2:
        start_sample = int(seconds_range[0] * sample_rate)
        end_sample = int(seconds_range[1] * sample_rate)
        audio_subarray = audio_array[start_sample:min(end_sample, audio_array.shape[0])]
    else:
        raise ValueError(f"Unsupported seconds_range len {len(seconds_range)}")
    return audio_subarray

# Plot the waveform of the audio data
def plot_waveform(audio_array, sample_rate, num_channels, seconds_range=None):
    if audio_array is None:
        print("Cannot plot: Audio data not loaded.")
        return
    if seconds_range is None:
        audio_subarray = audio_array
    else:
        audio_subarray = crop_waveform(audio_array, sample_rate, seconds_range)
    num_samples = audio_subarray.shape[0]
    duration = num_samples / sample_rate
    time_axis = np.linspace(0, duration, num_samples)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_axis, audio_subarray, label='Mono', color='blue')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Audio Waveform')
    ax.legend()
    ax.grid(True)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    time_ticks = ax.get_xticks()
    time_ticks = time_ticks[(time_ticks >= 0) & (time_ticks <= duration)]
    sample_ticks = (time_ticks * sample_rate).astype(int)
    ax2.set_xticks(time_ticks)
    ax2.set_xticklabels([f'{s}' for s in sample_ticks])
    ax2.set_xlabel('Sample Number')
    plt.tight_layout()
    plt.show()

# Compute similarity using a sliding window approach
def compute_similarity(audio_array_cropped):
    corr_patch = audio_array_cropped[0:audio_array_cropped.shape[0] // 5]
    sliding_window_view_audio = sliding_window_view(audio_array_cropped, window_shape=corr_patch.shape[0])
    corr_values = -1 * np.sum(np.abs(np.subtract(sliding_window_view_audio, corr_patch)), axis=1)
    corr_values = corr_values - np.min(corr_values)
    return corr_values

# Plot the similarity values
def plot_similarity(corr_values):
    plt.figure(figsize=(10, 4))
    plt.plot(corr_values)
    plt.ylabel('Amplitude')
    plt.title('Similarity')
    plt.grid(True)
    plt.show()

def estimate_fundamental_frequency(corr_values, sample_rate):
    max_value = np.max(corr_values[100:])  # ignore the first 100 samples due to strong corr
    peaks, _ = find_peaks(corr_values, height=max_value * 0.7, rel_height=0.9)
    print(f"number of peaks is {len(peaks)}")
    print(f"peaks = {peaks}")
    peaks_values = corr_values[peaks]
    print(f"peaks values = {peaks_values}")
    diff_peaks = peaks[1:] - peaks[:-1]
    print(f"diff_peaks = {diff_peaks}")
    if len(diff_peaks) > 0:
        median_period_samples = np.median(diff_peaks)
        fundamental_freq = sample_rate / median_period_samples
        print(f"Estimated fundamental frequency (F0): {fundamental_freq:.2f} Hz")
        return fundamental_freq
    else:
        print("Could not estimate fundamental frequency (no peaks found).")
        return None

def estimate_f0_over_time(audio_array, sample_rate, frame_size=512, hop_size=256):
    f0_list = []
    frame_starts = range(0, len(audio_array) - frame_size, hop_size)
    for start in frame_starts:
        frame = audio_array[start:start+frame_size]
        corr_values = compute_similarity(frame)
        f0 = estimate_fundamental_frequency(corr_values, sample_rate)
        f0_list.append(f0 if f0 is not None else 0)
    return np.array(f0_list), np.array(frame_starts) / sample_rate

def detect_pitch_changes(f0_array, threshold=30):
    changes = np.where(np.abs(np.diff(f0_array)) > threshold)[0]
    return changes

def compress_audio_segments_dct(audio_array, sample_rate, times, pitch_changes, n_coeffs=20):
    # Segment boundaries
    segment_starts = np.concatenate(([times[0]], times[pitch_changes]))
    segment_ends = np.concatenate((times[pitch_changes], [times[-1]]))
    compressed_segments = []
    segment_lengths = []
    for start, end in zip(segment_starts, segment_ends):
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment = audio_array[start_sample:end_sample]
        if len(segment) == 0:
            continue
        # Compute DCT and keep only the first n_coeffs
        segment_dct = dct(segment, norm='ortho')
        compressed = segment_dct[:n_coeffs]
        compressed_segments.append(compressed)
        segment_lengths.append(len(segment))
    return compressed_segments, segment_lengths

def reconstruct_audio_from_dct(compressed_segments, segment_lengths, n_coeffs=20):
    # Reconstruct each segment using inverse DCT
    reconstructed = []
    for coeffs, seg_len in zip(compressed_segments, segment_lengths):
        # Pad with zeros to original segment length
        padded_coeffs = np.zeros(seg_len)
        padded_coeffs[:n_coeffs] = coeffs
        segment_recon = idct(padded_coeffs, norm='ortho')
        reconstructed.append(segment_recon)
    return np.concatenate(reconstructed).astype(np.int16)

def main():
    from pathlib import Path
    script_dir = Path(__file__).parent
    recordings_folder = script_dir.parent / "tests" / "recordings"
    file_path = recordings_folder / "recording-aba.wav"
    if file_path.exists():
        print(f"Found file: {file_path}")
    else:
        print(f"File {file_path} does not exist.")
        return
    wave_fname = str(file_path)  # Convert Path to string
    seconds_range = [3.0, 4.0]
    audio_array, sample_rate, num_channels = load_wave_file(wave_fname)
    plot_waveform(audio_array, sample_rate, num_channels)
    plot_waveform(audio_array, sample_rate, num_channels, seconds_range)
    audio_array_cropped = crop_waveform(audio_array, sample_rate, seconds_range)
    corr_values = compute_similarity(audio_array_cropped)
    plot_similarity(corr_values)
    estimate_fundamental_frequency(corr_values, sample_rate)
    # Analyze pitch over time
    f0_array, times = estimate_f0_over_time(audio_array_cropped, sample_rate)
    pitch_changes = detect_pitch_changes(f0_array)
    print("Pitch change detected at times (seconds):", times[pitch_changes])

    # Print all time segments between pitch changes
    segment_starts = np.concatenate(([times[0]], times[pitch_changes]))
    segment_ends = np.concatenate((times[pitch_changes], [times[-1]]))
    print("\nTime segments between pitch changes:")
    for start, end in zip(segment_starts, segment_ends):
        print(f"Segment: {start:.3f} s to {end:.3f} s")

    # Optionally plot
    plt.figure(figsize=(10, 4))
    plt.plot(times, f0_array, label="F0 (Hz)")
    plt.scatter(times[pitch_changes], f0_array[pitch_changes], color='red', label="Change")
    plt.xlabel("Time (s)")
    plt.ylabel("Estimated F0 (Hz)")
    plt.legend()
    plt.title("Fundamental Frequency Over Time")
    plt.show()

    # Plot waveform with segmentation lines
    plt.figure(figsize=(14, 5))
    duration = len(audio_array) / sample_rate
    time_axis = np.linspace(0, duration, len(audio_array))
    plt.plot(time_axis, audio_array, label="Waveform", color='blue', alpha=0.6)
    for t in times[pitch_changes]:
        plt.axvline(x=t, color='red', linestyle='--', alpha=0.8, label='Pitch Change' if t == times[pitch_changes][0] else "")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio Segmentation by Pitch Change")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Advanced compression using DCT
    n_coeffs = 20  # Number of DCT coefficients to keep per segment
    # Use the full audio array for compression
    f0_array, times = estimate_f0_over_time(audio_array, sample_rate)
    pitch_changes = detect_pitch_changes(f0_array)
    compressed_segments, segment_lengths = compress_audio_segments_dct(
        audio_array, sample_rate, times, pitch_changes, n_coeffs=n_coeffs
    )
    reconstructed = reconstruct_audio_from_dct(compressed_segments, segment_lengths, n_coeffs=n_coeffs)
    wavfile.write("compressed_dct.wav", sample_rate, reconstructed)

if __name__ == "__main__":
    main()