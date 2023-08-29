import os
import argparse

METU_PATH = "/home/iran/datasets/spargair/em32/"

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('microphone', type=str, help='Name of the microphone type to be processed: em32, mic')
    args = parser.parse_args()
    microphone = args.microphone
    mic_nch = 32
    if microphone == "em32":
        process_channels = [i for i in range(1, mic_nch+1)]
    elif microphone == "mic":
        process_channels = [6, 10, 26, 22] # correspond to mic channels in em32
    else:
        parser.error("You must provide a valid microphone name: em32, mic")

    top_height = 5
    for meas in os.listdir(METU_PATH):
        # Impulse Reponse to be processed
        rir_name = meas[0] + meas[1] + meas[2]
        # Load the 32 eigenmike IR wavefiles and merge into a multi-channel file
        cmd_mix_all_ch = ""
        for ch_idx in process_channels:
            cmd_mix_all_ch += f' {os.path.join(METU_PATH, rir_name)}/IR{ch_idx:05}.wav '
        # SoX -M to merge 32 into multi-channel signal
        ir_path = os.path.join(METU_PATH, rir_name, f"IR_{microphone}.wav")
        print("Preparing", ir_path)
        os.system(f'sudo sox -M {cmd_mix_all_ch} {ir_path}')

if __name__ == '__main__':
    main()         
