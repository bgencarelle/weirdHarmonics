import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.playback import play
import io

def generate_waveform(wave_type, frequency, n_harmonics, duration_ms, phase_invert_alternate=True, sample_rate=44100):
    t = np.linspace(0, duration_ms / 1000, num=int(sample_rate * duration_ms / 1000), endpoint=False)

    # Waveform functions with the additional phase_invert parameter
    def triangle_wave(frequency, t, phase_invert):
        return (8 / (np.pi**2)) * np.sum([(np.sin((2 * n + 1) * 2 * np.pi * frequency * t + np.pi * phase_invert) / (2 * n + 1)**2) for n in range(n_harmonics)], axis=0)

    def square_wave(frequency, t, phase_invert):
        return (4 / np.pi) * np.sum([(np.sin((2 * n + 1) * 2 * np.pi * frequency * t + np.pi * phase_invert) / (2 * n + 1)) for n in range(n_harmonics)], axis=0)

    def sawtooth_wave(frequency, t, phase_invert):
        return (2 / np.pi) * np.sum([(np.sin(n * 2 * np.pi * frequency * t + np.pi * phase_invert) / n) for n in range(1, n_harmonics + 1)], axis=0)

    def inverse_sawtooth_wave(frequency, t, phase_invert):
        return (-2 / np.pi) * np.sum([(np.sin(n * 2 * np.pi * frequency * t + np.pi * phase_invert) / n) for n in range(1, n_harmonics + 1)], axis=0)

    if wave_type == 'triangle':
        wave_func = triangle_wave
    elif wave_type == 'square':
        wave_func = square_wave
    elif wave_type == 'sawtooth':
        wave_func = sawtooth_wave
    elif wave_type == 'inverse_sawtooth':
        wave_func = inverse_sawtooth_wave
    else:
        raise ValueError("Invalid wave type")

    nyquist_limit = sample_rate // 2
    max_harmonics = nyquist_limit // frequency
    n_harmonics = min(n_harmonics, max_harmonics)

    waveform = np.zeros_like(t)
    for n in range(1, n_harmonics * 2, 2):
        harmonic_freq = n * frequency
        amplitude_factor = 1 / n
        phase_invert = phase_invert_alternate and n % 4 == 3 and n > 1
        waveform += amplitude_factor * wave_func(harmonic_freq, t, phase_invert)

    amplitude = 2**15 - 1
    waveform = (amplitude * waveform / np.max(np.abs(waveform))).astype(np.int16)

    return waveform

def play_audio(waveform, sample_rate=44100):
    audio_segment = AudioSegment(waveform.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)
    play(audio_segment)

def plot_waveform(waveform, sample_rate=44100):
    t = np.linspace(0, len(waveform) / sample_rate, num=len(waveform), endpoint=False)
    plt.plot(t, waveform)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Oscilloscope')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    wave_type = 'sawtooth'  # Choose from: 'triangle', 'square', 'sawtooth', 'inverse_sawtooth'
    frequency = 440  # Frequency in Hz
    n_harmonics = 100  # Number of harmonics to include
    duration_ms = 10  # Duration of the audio in milliseconds
    phase_invert_alternate = False  # Invert the phase of alternate harmonics

    waveform = generate_waveform(wave_type, frequency, n_harmonics, duration_ms, phase_invert_alternate)
    play_audio(waveform)
    plot_waveform(waveform)
