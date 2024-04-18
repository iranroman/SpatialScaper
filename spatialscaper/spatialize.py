import numpy as np
import scipy.fft
# import tqdm

__all__ = ['spatialize']


def stft(y, fft_size=512, win_size=256, hop_size=128, stft_dims_first=True):
    # Generate the window
    window = np.sin(np.pi / win_size * np.arange(win_size))**2
    
    # Compute padding and pad the input signal
    n_frames = 2 * int(np.ceil(y.shape[-1] / (2.0 * hop_size))) + 1  # consistent with previous implementation
    pad_width = [(0, 0)] * (y.ndim - 1) + [(win_size - hop_size, n_frames * hop_size - y.shape[-1])]
    y_padded = np.pad(y, pad_width, mode='constant')

    # Use stride tricks to efficiently extract windows
    shape = y_padded.shape[:-1] + (win_size, n_frames)
    strides = y_padded.strides[:-1] + (y_padded.strides[-1], y_padded.strides[-1] * hop_size)
    windows = np.lib.stride_tricks.as_strided(y_padded, shape=shape, strides=strides)

    # Apply window function and compute FFT
    spec = scipy.fft.rfft(windows * window[:, None], fft_size, norm="backward", axis=-2)

    # move stft dims to the front (it's what the tv conv expects)
    if stft_dims_first:
        spec = np.moveaxis(np.moveaxis(spec, -1, 0), -1, 0)
    spec = np.ascontiguousarray(spec)
    return spec


def generate_interpolation_matrix(ir_times, sr, hop_size, n_frames=None):
    # frames: n_irs
    frames = np.round((ir_times * sr + hop_size) / hop_size)
    n_frames = n_frames if n_frames is not None else int(frames[-1])
    # G_interp: n_frames, n_irs
    G_interp = np.zeros((n_frames, len(frames)))
    for ni in range(len(frames) - 1):
        tpts = np.arange(frames[ni], frames[ni + 1] + 1, dtype=int) - 1
        ntpts_ratio = np.linspace(0, 1, len(tpts))
        G_interp[tpts, ni] = 1 - ntpts_ratio
        G_interp[tpts, ni + 1] = ntpts_ratio
    return G_interp


# @profile
def perform_time_variant_convolution(S_audio, S_ir, W_ir, win_size, hop_size, _ir_slice_min=0, _ir_relevant_ratio_max=1):
    # get shapes
    n_freq, n_frames_ir, n_ch, n_irs = S_ir.shape  # NOTE: (channels, n_irs) is muchh faster than the other way around
    n_frames = min(S_audio.shape[1], W_ir.shape[0])  # TODO: constant pad ir_interp to sigspec length
    fft_size = 2 * win_size

    # Invert time for convolution
    S = np.ascontiguousarray(S_audio[:, ::-1])
    W_ir = np.ascontiguousarray(W_ir[::-1]).astype(complex)

    # Output: spatialized audio signal: (n_samples, n_ch)
    spatial_audio = np.zeros(((n_frames + 1) * win_size // 2 + win_size, n_ch))

    # tqdm.tqdm(range(n_frames), desc='calculating 🥵...', leave=False)
    for i in range(n_frames):
        # reverse indices for IR frames
        i_ir = -i-1
        j_ir = min(-i-1+n_frames_ir, 0) or None

        # compute the weighted IR spectrogram: ijkl,jl->ijk  # XXX: Takes ~72% of the time
        sir = S_ir[:, :i+1]
        wir = W_ir[i_ir:j_ir]
        # slice active irs to reduce computation
        if _ir_slice_min is not None and n_irs >= _ir_slice_min:
            relevant = np.any(wir != 0, axis=0)
            if relevant.mean() < _ir_relevant_ratio_max:  # could optimize this
                sir = sir[:, :, :, relevant]  # this is a copy because of the boolean array :/
                wir = wir[:, relevant]

        # (freq,  , ch, nir) x (   , frame,   , nir) = (freq, frame, ch,   )
        ctf_ltv = np.einsum('ijkl,jl->ijk', sir, wir)

        # Multiply the signal spectrogram with the CTF:  # XXX: Takes about ~6% of the time
        # (freq,  , ch) x (freq, frame,   ) = (freq,   ch)
        Si = np.einsum('ijk,ij->ik', ctf_ltv, S[:, i_ir:j_ir])

        # Inverse FFT to convert freq to time domain  # XXX: Takes about ~14% of the time
        audio_frame = np.real(scipy.fft.irfft(Si, fft_size, norm="forward", axis=0))

        # overlap-add synthesis
        spatial_audio[i * hop_size : i * hop_size + fft_size] += audio_frame

    # clip output - consistent with previous implementation
    spatial_audio = spatial_audio[win_size:(n_frames * win_size) // 2, :]
    return spatial_audio


def apply_snr(x, snr):
    x *= snr / np.abs(x).max(initial=1e-15)
    return x


# @profile
def spatialize(audio, irs, ir_times, sr, win_size=512, snr=1.0):
    '''Performs time-variant convolution of a signal with multiple impulse responses.

    This function convolves an input signal with a series of impulse responses that vary over time.
    The convolution is performed in the frequency domain using Short-Time Fourier Transform (STFT).
    It handles both single and multi-channel impulse responses and signals.

    Arguments:
        audio (np.ndarray): 1-D Input audio signal with shape [audio samples]. 
        irs (np.ndarray): Impulse response audio signals of shape (channels, IR index, audio samples).
        ir_times (np.ndarray): Start times for each impulse response corresponding to [IR index]. 
            Currently, they are linearly cross-faded.
        sr (float): The audio sample rate for both the signal and the impulse responses.
        win_size (int): The window size of the FFT
        snr (float): The signal-to-noise ratio of the audio file. By default, the audio peak is normalized to 1.
            This is equivalent to multiplying the output signal by this number.

    Returns:
        np.ndarray: The spatialized audio signal with shape [audio samples, channels].
    '''
    n_ch, n_irs, n_ir_samples = irs.shape
    if n_irs == 1:  # trivial - single ir
        spatial_audio = scipy.signal.fftconvolve(audio[:, None], irs[:, 0].T, mode="full", axes=0)[:len(audio), :]
        assert spatial_audio.shape == (audio.shape[0], n_ch)
        return spatial_audio
    if n_irs == 0:  # unspatialized
        return np.repeat(audio[:, None], n_ch, 1)

    # Get parameters and shapes
    win_size = int(win_size)
    hop_size = win_size // 2

    # check ir shape
    assert n_irs == 1 or n_irs == ir_times.shape[0], f"ir_times must have the same number of IRs as irs. got {n_irs} and {ir_times.shape}"

    # compute spectrograms
    # ir_spec:    (n_freq, n_frames, n_ch, n_irs)
    # audio_spec: (n_freq, n_frames,            )
    ir_spec = stft(irs, fft_size=2 * win_size, win_size=win_size, hop_size=hop_size)
    audio_spec = stft(audio, fft_size=2 * win_size, win_size=win_size, hop_size=hop_size)  # NOTE: as audio grows, this takes majority of time
    _assert_shape_match(ir_spec.shape, (win_size+1, None, n_ch, n_irs))
    _assert_shape_match(audio_spec.shape, (win_size+1, None))

    # get the ir interpolation weight matrix: (n_frames, n_irs)  # TODO: pass audio_spec.shape[1]
    W_ir = generate_interpolation_matrix(ir_times, sr, hop_size)#, audio_spec.shape[1]
    _assert_shape_match(W_ir.shape, (None, n_irs))
    assert audio_spec.shape[1] -2 <= W_ir.shape[0] <= audio_spec.shape[1], f'{W_ir.shape}: {W_ir.shape[0]}!={audio_spec.shape[1]} - {ir_times}'

    # convolve signal with irs
    spatial_audio = perform_time_variant_convolution(audio_spec, ir_spec, W_ir, win_size, hop_size)
    spatial_audio = apply_snr(spatial_audio, snr)
    _assert_shape_match(spatial_audio.shape, (None, n_ch)) #audio.shape[0]
    return spatial_audio

def _assert_shape_match(shape_a, shape_b, msg=None):
    assert all(a==b or b is None for a, b in zip(shape_a, shape_b)), msg or f'{shape_a} != {shape_b}'
