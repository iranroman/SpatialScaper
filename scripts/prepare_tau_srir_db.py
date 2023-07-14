from utils import parse_args
import os

def main():
    args = parse_args()

    tau_rir_files = os.listdir(args.path)
    tau_rir_pickles = [f for f in tau_rir_files if f.endswith('pkl')]


if __name__ == "__main__":
    main()
