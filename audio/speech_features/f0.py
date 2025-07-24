"""
File: Amit Tzadok
Author: Amit Tzadok <amit.tzadok@icloud.com>
Created: 2025-07-22 11:05:31
Description: 
"""




import wave
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks

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
    return audio_array, sample_rate, num_channels

def crop_waveform(audio_array, sample_rate, seconds_range):
    if len(seconds_range) == 2:
        start_sample = int(seconds_range[0] * sample_rate)
        end_sample = int(seconds_range[1] * sample_rate)
        audio_subarray = audio_array[start_sample:min(end_sample, audio_array.shape[0])]
    else:
        raise ValueError(f"Unsupported seconds_range len {len(seconds_range)}")
    return audio_subarray

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

def compute_similarity(audio_array_cropped):
    corr_patch = audio_array_cropped[0:audio_array_cropped.shape[0] // 5]
    sliding_window_view_audio = sliding_window_view(audio_array_cropped, window_shape=corr_patch.shape[0])
    corr_values = -1 * np.sum(np.abs(np.subtract(sliding_window_view_audio, corr_patch)), axis=1)
    corr_values = corr_values - np.min(corr_values)
    return corr_values

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

def main():
    from pathlib import Path
    script_dir = Path(__file__).parent
    recordings_folder = script_dir.parent / "tests" / "recordings"
    file_path = recordings_folder / "recording-aba.wav"
    if file_path.exists():
        print(f"Found file: {file_path}")
        # Add your processing logic here (e.g., open with a library)
    else:
        print(f"File {file_path} does not exist.")
    wave_fname = file_path
    seconds_range = [3.65, 3.75]
    audio_array, sample_rate, num_channels = load_wave_file(wave_fname)
    plot_waveform(audio_array, sample_rate, num_channels)
    plot_waveform(audio_array, sample_rate, num_channels, seconds_range)
    audio_array_cropped = crop_waveform(audio_array, sample_rate, seconds_range)
    corr_values = compute_similarity(audio_array_cropped)
    plot_similarity(corr_values)
    estimate_fundamental_frequency(corr_values, sample_rate)

if __name__ == "__main__":
    main()