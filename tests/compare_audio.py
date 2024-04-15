import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.fft


def main(path_a, path_b):
    audio_a, sr_a = librosa.load(path_a, sr=None, mono=False)
    audio_b, sr_b = librosa.load(path_b, sr=None, mono=False)
    assert sr_a == sr_b, "sampling rates dont match"
    
    print(f"sampling rate: {sr_a}")

    # Check the shape of the loaded audio arrays
    print("Audio A shape:", audio_a.shape)
    print("Audio B shape:", audio_b.shape)
    print(np.allclose(audio_a, audio_b), np.abs(audio_a-audio_b).max())
    n_ch = audio_a.shape[0]
    
    # Calculate residuals for each channel and visualize again
    fig, axs = plt.subplots(n_ch, 3, figsize=(14, 10))

    for i in range(n_ch):
        Sa = librosa.amplitude_to_db(np.abs(librosa.stft(audio_a[i])), ref=np.max)
        Sb = librosa.amplitude_to_db(np.abs(librosa.stft(audio_b[i])), ref=np.max)
        # Compute residual
        residual = Sa - Sb
        print(Sa.shape, Sb.shape, np.allclose(Sa, Sb), residual.min(), residual.max())
        librosa.display.specshow(Sa, y_axis='log', sr=sr_a, ax=axs[i, 0])
        librosa.display.specshow(Sa, y_axis='log', sr=sr_a, ax=axs[i, 1])
        librosa.display.specshow(residual, y_axis='log', sr=sr_a, ax=axs[i, 2], cmap='coolwarm')
    plt.savefig('compare_spec.png')


if __name__ == '__main__':
    import fire
    fire.Fire(main)