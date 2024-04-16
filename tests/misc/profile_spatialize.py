import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from spatialscaper.spatialize import spatialize
from spatialscaper.utils import spatialize as spatialize_old

PLT_DIR = os.path.dirname(__file__)

def generate_data(spatialize, audio_lengths, num_irs, ir_samples, sr, runs_per_test):
    results = np.zeros((len(num_irs), len(audio_lengths)))
    for i, n_ir in enumerate(tqdm.tqdm(num_irs, leave=False)):
        for j, length in enumerate(tqdm.tqdm(audio_lengths, leave=False)):
            run_times = []
            for _ in range(runs_per_test):
                audio = np.random.randn(int(length * sr))
                irs = np.random.randn(2, n_ir, ir_samples)
                ir_times = np.linspace(0, length / sr, n_ir)
                start_time = time.time()
                result = spatialize(audio, irs, ir_times, sr)
                run_times.append(time.time() - start_time)
            results[i, j] = np.median(run_times)
    return results

def plot_lines(results, n_irs, audio_lengths):
    # Plotting results
    # plt.figure(figsize=(10, 6))
    for n_ir, times in zip(n_irs, results):
        plt.plot(audio_lengths, times, label=f'Num IRs: {n_ir}')
    plt.xlabel('Audio Length (seconds)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time of Spatialize Function')
    plt.legend()
    plt.grid(True)

def main():
    audio_lengths = [1, 15, 30, 60, 120, 240, 580]
    num_irs = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    ir_samples = 512
    sr = 24000
    runs_per_test = 5

    spatialize(np.random.randn(1000), np.random.randn(2, 1, ir_samples), np.linspace(0, 1000 / sr, 2), sr=sr)

    results_new = generate_data(spatialize, audio_lengths, num_irs, ir_samples, sr, runs_per_test)
    results_old = generate_data(spatialize_old, audio_lengths, num_irs, ir_samples, sr, runs_per_test)
    difference = results_old / results_new

    plt.figure(figsize=(15, 8))
    plt.subplot(3, 1, 1)
    plot_lines(results_new, num_irs, audio_lengths)
    plt.subplot(3, 1, 2)
    plot_lines(results_old, num_irs, audio_lengths)
    plt.subplot(3, 1, 3)
    plot_lines(difference * 100, num_irs, audio_lengths)
    plt.ylabel('Relative Execution Time (%)')
    plt.tight_layout()
    plt.savefig(f"{PLT_DIR}/profile_spatialize.png")

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(difference, annot=True, fmt=".0%", cmap='RdBu', xticklabels=audio_lengths, yticklabels=num_irs, vmin=-difference.max(), vmax=difference.max())
    # ax=plt.gca()
    # plt.imshow(difference, cmap='coolwarm')
    ax.set_title('Relative Execution Time (Old / New %)')
    ax.set_xlabel('Audio Length (seconds)')
    ax.set_ylabel('Number of Impulse Responses')
    plt.savefig(f"{PLT_DIR}/profile_spatialize_diff.png")

if __name__ == '__main__':
    import fire
    fire.Fire(main)
