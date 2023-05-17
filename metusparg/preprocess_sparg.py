import os
import argparse

METU_PATH = "/mnt/ssdt7/RIR-datasets/spargair/em32"

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('microphone', type=str, help='Name of the microphone type to be processed: em32, mic')
    args = parser.parse_args()
    microphone = args.microphone
    mic_nch = 32
    if microphone == "em32":
        process_channels = [i for i in range(1, mic_nch+1)]
    elif microphone == "mic":
        process_channels = [6, 10, 26, 22]
    else:
        parser.error("You must provide a valid microphone name: em32, mic")

    outter_trayectory_bottom = ["034", "024", "014", "004", "104", "204",
                                "304", "404", "504", "604", "614", "624",
                                "634", "644", "654", "664", "564", "464",
                                "364", "264", "164", "064", "054", "044"]
    top_height = 5
    for height in range(0, top_height): # loop through heights
        for num in outter_trayectory_bottom:
            # Impulse Reponse to be processed
            rir_name = num[0] + num[1] + str(int(num[2])-height)
            # Load the 32 eigenmike IR wavefiles and merge into a multi-channel file
            cmd_mix_all_ch = ""
            for ch_idx in process_channels:
                cmd_mix_all_ch += f' {os.path.join(METU_PATH, rir_name)}/IR{ch_idx:05}.wav '
            # SoX -M to merge 32 into multi-channel signal
            ir_path = os.path.join(METU_PATH, rir_name, f"IR_{microphone}.wav")
            print("generating", ir_path)
            os.system(f'sudo sox -M {cmd_mix_all_ch} {ir_path}')

if __name__ == '__main__':
    main()            
