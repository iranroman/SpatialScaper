#%%

import numpy as np
import scipy.fft

def stft_ham_new(y, fft_size=512, win_size=256, hop_size=128, move_to_start=True):
    # Generate the window
    window = np.sin(np.pi / win_size * np.arange(win_size))**2
    
    # Compute padding and pad the input signal
    # n_frames = 2 * int(np.ceil(y.shape[-1] / (2.0 * hop_size)))
    # n_frames = (y.shape[-1] + win_size) // hop_size  # consistent with previous implementation
    # extra_pad = 0#(n_frames - 1) * hop_size + win_size - len(y)
    n_frames = int(2 * np.ceil(len(y) / (2.0 * hop_size)) + 1)
    # frontpad = win_size - hop_size
    # backpad = n_frames * hop_size - len(y)
    pad_width = [(0, 0)] * (y.ndim - 1) + [(win_size - hop_size, n_frames * hop_size - y.shape[-1])]
    y_padded = np.pad(y, pad_width, mode='constant')

    # Use stride tricks to efficiently extract windows
    shape = y_padded.shape[:-1] + (win_size, n_frames)
    strides = y_padded.strides[:-1] + (y_padded.strides[-1], y_padded.strides[-1] * hop_size)
    windows = np.lib.stride_tricks.as_strided(y_padded, shape=shape, strides=strides)
    
    # Apply window function and compute FFT
    # spec = np.fft.rfft(windows * window[:, None], fft_size, norm="backward", axis=-2)
    spec = scipy.fft.rfft(windows * window[:, None], fft_size, norm="backward", axis=-2)
    if move_to_start:
        spec = np.moveaxis(np.moveaxis(spec, -1, 0), -1, 0)
    spec = np.ascontiguousarray(spec)
    return spec

def stft_ham_old(insig, winsize=256, fftsize=512, hopsize=128):
    nb_dim = len(np.shape(insig))
    lSig = int(np.shape(insig)[0])
    nCHin = int(np.shape(insig)[1]) if nb_dim > 1 else 1
    x = np.arange(0, winsize)
    nBins = int(fftsize / 2 + 1)
    nWindows = int(np.ceil(lSig / (2.0 * hopsize)))
    nFrames = int(2 * nWindows + 1)

    winvec = np.zeros((len(x), nCHin))
    for i in range(nCHin):
        winvec[:, i] = np.sin(x * (np.pi / winsize)) ** 2

    frontpad = winsize - hopsize
    backpad = nFrames * hopsize - lSig

    if nb_dim > 1:
        insig_pad = np.pad(insig, ((frontpad, backpad), (0, 0)), "constant")
        spectrum = np.zeros((nBins, nFrames, nCHin), dtype="complex")
    else:
        insig_pad = np.pad(insig, ((frontpad, backpad)), "constant")
        spectrum = np.zeros((nBins, nFrames), dtype="complex")

    idx = 0
    nf = 0
    if nb_dim > 1:
        while nf <= nFrames - 1:
            insig_win = np.multiply(winvec, insig_pad[idx + np.arange(0, winsize), :])
            print(winvec.shape, insig_pad.shape, insig_win.shape, flush=True)
            inspec = scipy.fft.fft(insig_win, n=fftsize, norm="backward", axis=0)
            inspec = inspec[:nBins, :]
            spectrum[:, nf, :] = inspec
            idx += hopsize
            nf += 1
    else:
        while nf <= nFrames - 1:
            insig_win = np.multiply(
                winvec[:, 0], insig_pad[idx + np.arange(0, winsize)]
            )
            inspec = scipy.fft.fft(insig_win, n=fftsize, norm="backward", axis=0)
            inspec = inspec[:nBins]
            spectrum[:, nf] = inspec
            idx += hopsize
            nf += 1

    return spectrum

for i in range(30):
    # Generate a test signal: a simple sinusoid
    fs = 48000  # sampling frequency
    t = np.linspace(0, 1, int(fs * i**1.5), endpoint=False)
    freq = 440  # A4 note
    test_signal = 0.5 * np.sin(2 * np.pi * freq * t)
    test_signal = test_signal + np.random.randn()

    # Apply both STFT functions
    spec_new = stft_ham_new(test_signal)
    spec_old = stft_ham_old(test_signal)

    # Compare the results
    comparison = np.allclose(spec_new, spec_old, atol=1e-5)
    print(f"Are the outputs approximately equal? {comparison}")

    # If they are not equal, print the differences
    if not comparison:
        diff = np.abs(spec_new - spec_old)
        print(f"Maximum difference: {np.max(diff)}")

# # %%

# import numpy as np

# # Parameters for STFT
# fft_size = 512
# win_size = 256
# hop_size = 128

# # Generate a random test signal
# np.random.seed(0)
# test_signal = np.random.randn(1024)

# # Old method calculation from the original implementation
# lSig_old = len(test_signal)
# nWindows_old = int(np.ceil(lSig_old / (2.0 * hop_size)))
# nFrames_old = int(2 * nWindows_old + 1)

# # New method calculation with proposed change
# n_frames_new = (test_signal.shape[0] + win_size) // hop_size

# # Output the calculated frame numbers
# print("nFrames_old:", nFrames_old)
# print("n_frames_new:", n_frames_new)

# # Check if the frame numbers are identical
# print("Are frame calculations identical?", nFrames_old == n_frames_new)
