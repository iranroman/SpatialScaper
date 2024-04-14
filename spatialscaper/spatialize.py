import numpy as np
import scipy.fft
# import tqdm


def stft_ham(y, fft_size=512, win_size=256, hop_size=128, move_to_start=True):
    # Generate the window
    window = np.sin(np.pi / win_size * np.arange(win_size))**2
    
    # Compute padding and pad the input signal
    # Compute padding and pad the input signal
    n_frames = 2 * int(np.ceil(y.shape[-1] / (2.0 * hop_size))) + 1
    # n_frames = (y.shape[-1] + win_size) // hop_size  # consistent with previous implementation
    # extra_pad = 0#(n_frames - 1) * hop_size + win_size - len(y)
    # n_frames = int(2 * np.ceil(len(y) / (2.0 * hop_size)) + 1)
    # frontpad = win_size - hop_size
    # backpad = n_frames * hop_size - len(y)
    # print(n_frames, win_size, hop_size, y.shape, win_size - hop_size, n_frames * hop_size - y.shape[-1])
    pad_width = [(0, 0)] * (y.ndim - 1) + [(win_size - hop_size, n_frames * hop_size - y.shape[-1])]
    y_padded = np.pad(y, pad_width, mode='constant')

    # Use stride tricks to efficiently extract windows
    # n_frames = (y.shape[-1] + win_size) // hop_size  # consistent with previous implementation
    shape = y_padded.shape[:-1] + (win_size, n_frames)
    strides = y_padded.strides[:-1] + (y_padded.strides[-1], y_padded.strides[-1] * hop_size)
    windows = np.lib.stride_tricks.as_strided(y_padded, shape=shape, strides=strides)
    
    # Apply window function and compute FFT
    spec = scipy.fft.rfft(windows * window[:, None], fft_size, norm="backward", axis=-2)
    if move_to_start:
        spec = np.moveaxis(np.moveaxis(spec, -1, 0), -1, 0)
    spec = np.ascontiguousarray(spec)
    return spec


def generate_interpolation_matrix(ir_times, sr, hop_size):
    # frames: n_irs
    frames = np.ceil((ir_times * sr + hop_size) / hop_size)
    # G_interp: n_frames, n_irs
    G_interp = np.zeros((int(frames[-1]+1), len(frames))) # FIXME: +1 is a hack
    for ni in range(len(frames) - 1):
        tpts = np.arange(frames[ni], frames[ni + 1] + 1, dtype=int) - 1
        ntpts_ratio = np.linspace(0, 1, len(tpts))
        G_interp[tpts, ni] = 1 - ntpts_ratio
        G_interp[tpts, ni + 1] = ntpts_ratio
    return G_interp


# @profile
def perform_time_variant_convolution(
        sigspec, irspec, ir_interp, 
        win_size, hop_size
    ):
    # get shapes
    n_freq, n_frames_ir, n_ch, n_irs = irspec.shape  # NOTE: (channels, n_irs) is muchh faster than the other way around
    n_frames = sigspec.shape[1]
    fft_size = 2 * win_size

    # Temporary buffers: shifted spectrum, interpolation weights
    W = np.zeros((n_frames_ir, n_irs), dtype=complex)
    S = np.zeros((n_freq, n_frames_ir), dtype=complex)
    ctf_ltv = np.zeros((n_freq, n_frames_ir, n_ch), dtype=complex)

    # Output:
    # spatialized audio signal: (n_samples, n_ch)
    spatial_signal = np.zeros(((n_frames + 1) * win_size // 2 + win_size, n_ch))

    # tqdm.tqdm(range(n_frames), desc='calculating 🥵...', leave=False)
    for i in range(n_frames):
        # Shift interpolation buffer
        W[1:] = W[:-1]  # Shift up
        W[0] = ir_interp[i]  # Update with current interpolation weights

        # Shift spectrogram buffer
        S[:, 1:] = S[:, :-1]  # Shift up
        S[:, 0] = sigspec[:, i]  # Update with the new signal spectrum

        # compute the weighted IR spec: ijkl,jl->ijk  # XXX: Takes ~89% of the time
        #   irspec:  (freq, frame, channel, n_ir) = (513, 27, 4, 36)
        # x W:       (    , frame,          n_ir) = (     27,    36)
        # = ctf_ltv: (freq, frame, channel      ) = (513, 27, 4    )
        ctf_ltv = np.einsum('ijkl,jl->ijk', irspec, W)
        # , order='C', casting='no', optimize=['einsum_path', (0, 1)]

        # Multiply the signal spectrum with the CTF:  # XXX: Takes about ~7% of the time
        #   S:       (freq, frame,        ) = (513, 27, 4)
        # x ctf_ltv: (freq, frame, channel) = (513, 27, 4)
        # = spec_i:  (freq,      , channel) = (513,   , 4)
        spec_i = (S[:, :, None] * ctf_ltv).sum(1)

        # Inverse FFT to convert the convolution result back to time domain
        sig_part = scipy.fft.irfft(spec_i, fft_size, norm="forward", axis=0)

        # overlap-add synthesis
        spatial_signal[i * hop_size : i * hop_size + fft_size] += sig_part

    # clip output - consistent with previous implementation
    spatial_signal = spatial_signal[win_size:(n_frames * win_size) // 2, :]
    return spatial_signal


def apply_snr(x, snr):
    x /= max(np.abs(x).max(), 1e-15)
    return x * snr


# @profile
def spatialize(audio, irs, ir_times, sr, win_size=512, snr=1.0):  # TODO: s->snr
    '''Performs time-variant convolution of a signal with multiple impulse responses.

    This function convolves an input signal with a series of impulse responses that vary over time.
    The convolution is performed in the frequency domain using Short-Time Fourier Transform (STFT).
    It handles both single and multi-channel impulse responses and signals.

    '''
    # irs = irs.transpose(1, 2, 0)  # flip dimensions
    
    n_ch, n_irs, n_ir_samples = irs.shape
    if n_irs == 1:  # trivial - single ir
        audio = scipy.signal.fftconvolve(audio[:, None], irs[:, 0].T, mode="full", axes=0)[:len(audio), :]
        return audio
    if n_irs == 0:  # unspatialized
        return np.repeat(audio[:, None], n_ch, 1)

    # Get parameters and shapes
    win_size = int(win_size)
    hop_size = win_size // 2

    # check ir shape
    assert n_irs == ir_times.shape[0], f"ir_times must have the same number of IRs as irs. got {n_irs} and {ir_times.shape}"

    # get the ir interpolation matrix: (n_frames, n_irs)
    G_interp = generate_interpolation_matrix(ir_times, sr, hop_size)

    # irspec: (n_irs, n_ch_irs, n_freq, n_frames)
    # sigspec: (n_ch, n_freq, n_frames)
    irspec = stft_ham(irs, fft_size=2 * win_size, win_size=win_size, hop_size=hop_size, move_to_start=True)
    sigspec = stft_ham(audio, fft_size=2 * win_size, win_size=win_size, hop_size=hop_size, move_to_start=True)
    spatial_signal = perform_time_variant_convolution(sigspec, irspec, G_interp, win_size, hop_size)
    spatial_signal = apply_snr(spatial_signal, snr)
    return spatial_signal
