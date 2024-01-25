import numpy as np
import scipy.io
import csv
import librosa
import os
import pickle


class DBConfig(object):
    def __init__(self, params):
        self._rirpath = params["rirpath"]
        self._mixturepath = params["mixturepath"]
        self._rirdata = self._load_rirdata()
        self._nb_folds = params["nb_folds"]
        self._rooms2fold = params["rooms2fold"]
        self._db_path = params["db_path"]
        self._db_name = params["db_name"]
        if self._db_name == "fsd50k":
            self._fs = 44100
            self._classes = [
                "femaleSpeech",
                "maleSpeech",
                "clapping",
                "telephone",
                "laughter",
                "domesticSounds",
                "footsteps",
                "doorCupboard",
                "music",
                "musicInstrument",
                "waterTap",
                "bell",
                "knock",
            ]
            self._nb_classes = len(self._classes)
            self._class_mobility = [2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0]
            self._apply_class_gains = True
            #             self._class_gains = [[0,    0.2004,    0.8008,    6.8766,  357.8846], # femaleSpeech
            #                                  [0.0060,    0.4901,    2.5097,   14.3011,  372.2183], # maleSpeech
            #                                  [0.3607,    1.1029,    2.6719,    3.9629,   26.6442],  #clapping
            #                                  [0.0072,    0.8222,    2.3849,   34.1233,  168.5152],  #telephone
            #                                  [0.0273,    0.8911,    1.9856,    5.6164,   79.1070],  #laughter
            #                                  [0.0268,    0.1009,    1.8363,   13.9294,   83.2484],  #domesticSounds
            #                                  [0.0099,    0.3764,    1.2759,    5.4426,  318.8329],  #footsteps
            #                                  [0.0697,    0.4919,    2.7159,   28.0537,  313.8807],  #doorCupboard
            #                                  [0.0219,    0.3189,    0.7787,    2.3823,  355.9656],  #music
            #                                  [0.0160,    0.9563,    2.3413,    5.6720,  168.6679],  #musicInstrument
            #                                  [0.0972,    0.1828,    0.6304,    0.9522,  125.1975],  #waterTap
            #                                  [0.0160,    0.9563,    2.3413,    5.6720,  168.6679],  #bell
            #                                  [0.0697,    0.4919,    2.7159,   28.0537,  313.8807]] #knock
            self._class_gains = [
                [0.0791, 0.5330, 1.3132, 2.2365, 541.3376],  # femaleSpeech
                [0.0116, 0.6913, 1.2199, 3.0048, 235.0189],  # maleSpeech
                [0.5083, 2.2579, 3.0934, 7.6387, 100.1174],  # clapping
                [0.0126, 0.3373, 0.7526, 2.1165, 18.5226],  # telephone
                [0.1909, 1.4950, 3.2206, 8.2153, 221.2892],  # laughter
                [0.0004, 1.8347, 3.4778, 5.9276, 555.4895],  # domesticSounds
                [0.0099, 0.3969, 0.8870, 2.0800, 15.7529],  # footsteps
                [0.0146, 0.9141, 7.8186, 109.0767, 3979.700],  # doorCupboard
                [0.1153, 0.4313, 1.2903, 3.3541, 52.6977],  # music
                [0.0596, 1.4146, 5.3529, 20.6286, 362.0704],  # musicInstrument
                [0.0117, 0.5505, 1.4926, 2.1936, 44.9466],  # waterTap
                [0.0596, 1.4146, 5.3529, 20.6286, 362.0704],  # bell
                [2.4502, 2.4502, 41.3609, 80.2716, 80.2716],
            ]  # knock
            self._samplelist = self._load_db_fileinfo_fsd()

        elif self._db_name == "nigens":
            self._fs = 44100
            self._class_dict = {
                "alarm": 0,
                "baby": 1,
                "crash": 2,
                "dog": 3,
                "engine": 4,
                "femaleScream": 5,
                "femaleSpeech": 6,
                "fire": 7,
                "footsteps": 8,
                "knock": 9,
                "maleScream": 10,
                "maleSpeech": 11,
                "phone": 12,
                "piano": 13,
                "general": 14,
            }
            self._class_mobility = [0, 2, 0, 2, 2, 0, 2, 0, 1, 0, 0, 2, 2, 0, 0]
            self._classes = list(self._class_dict.keys())
            self._nb_classes = len(self._classes)
            self._samplelist = self._load_db_fileinfo_nigens()
            self._apply_class_gains = False
            self._class_gaines = []

    def _load_rirdata(self):
        with open(os.path.join(self._rirpath, "rirdata_dict.pkl"), "rb") as file:
            rirdata = pickle.load(file)
        return rirdata

    def _load_db_fileinfo_fsd(self):
        samplelist_per_fold = []
        folds = self._make_selected_filelist()

        for nfold in range(self._nb_folds):
            print("Preparing sample list for fold {}".format(str(nfold + 1)))
            counter = 1
            samplelist = {
                "class": np.array([]),
                "audiofile": np.array([]),
                "duration": np.array([]),
                "onoffset": [],
                "nSamples": [],
                "nSamplesPerClass": np.array([]),
                "meanStdDurationPerClass": np.array([]),
                "minMaxDurationPerClass": np.array([]),
            }
            for ncl in range(self._nb_classes):
                nb_samples_per_class = len(folds[ncl][nfold])

                for ns in range(nb_samples_per_class):
                    samplelist["class"] = np.append(samplelist["class"], ncl)
                    samplelist["audiofile"] = np.append(
                        samplelist["audiofile"], folds[ncl][nfold][ns]
                    )
                    audiopath = self._db_path + "/" + folds[ncl][nfold][ns]
                    audio, sr = librosa.load(audiopath)
                    duration = len(audio) / float(sr)
                    samplelist["duration"] = np.append(samplelist["duration"], duration)
                    samplelist["onoffset"].append(
                        np.array(
                            [
                                [0.0, duration],
                            ]
                        )
                    )
                    samplelist["nSamples"].append(counter)
                    counter += 1
            samplelist["onoffset"] = np.squeeze(
                np.array(samplelist["onoffset"], dtype=object)
            )
            for n_class in range(self._nb_classes):
                class_idx = samplelist["class"] == n_class
                samplelist["nSamplesPerClass"] = np.append(
                    samplelist["nSamplesPerClass"], np.sum(class_idx)
                )
                if n_class == 0:
                    samplelist["meanStdDurationPerClass"] = np.array(
                        [
                            [
                                np.mean(samplelist["duration"][class_idx]),
                                np.std(samplelist["duration"][class_idx]),
                            ]
                        ]
                    )
                    samplelist["minMaxDurationPerClass"] = np.array(
                        [
                            [
                                np.min(samplelist["duration"][class_idx]),
                                np.max(samplelist["duration"][class_idx]),
                            ]
                        ]
                    )
                else:
                    samplelist["meanStdDurationPerClass"] = np.vstack(
                        (
                            samplelist["meanStdDurationPerClass"],
                            np.array(
                                [
                                    np.mean(samplelist["duration"][class_idx]),
                                    np.std(samplelist["duration"][class_idx]),
                                ]
                            ),
                        )
                    )
                    samplelist["minMaxDurationPerClass"] = np.vstack(
                        (
                            samplelist["minMaxDurationPerClass"],
                            np.array(
                                [
                                    np.min(samplelist["duration"][class_idx]),
                                    np.max(samplelist["duration"][class_idx]),
                                ]
                            ),
                        )
                    )
            samplelist_per_fold.append(samplelist)

        return samplelist_per_fold

    def _load_db_fileinfo_nigens(self):
        samplelist_per_fold = []

        for nfold in range(self._nb_folds):
            print("Preparing sample list for fold {}".format(str(nfold + 1)))
            foldlist_file = (
                self._db_path
                + "/NIGENS_8-foldSplit_fold"
                + str(nfold + 1)
                + "_wo_timit.flist"
            )
            filelist = []
            with open(foldlist_file, newline="") as flist:
                flist_reader = csv.reader(flist, delimiter="\t")
                for fline in flist_reader:
                    filelist.append(fline)
            flist_len = len(filelist)

            samplelist = {
                "class": np.array([]),
                "audiofile": np.array([]),
                "duration": np.array([]),
                "onoffset": [],
                "nSamples": flist_len,
                "nSamplesPerClass": np.array([]),
                "meanStdDurationPerClass": np.array([]),
                "minMaxDurationPerClass": np.array([]),
            }
            for file in range(flist_len):
                clsfilename = filelist[file][0].split("/")
                clsname = clsfilename[0]
                filename = clsfilename[1]

                samplelist["class"] = np.append(
                    samplelist["class"], int(self._class_dict[clsname])
                )
                samplelist["audiofile"] = np.append(
                    samplelist["audiofile"], clsname + "/" + filename
                )
                audiopath = self._db_path + "/" + clsname + "/" + filename
                # print(audiopath)
                # with contextlib.closing(wave.open(audiopath,'r')) as f:
                audio, sr = librosa.load(audiopath)
                samplelist["duration"] = np.append(
                    samplelist["duration"], len(audio) / float(sr)
                )

                if clsname == "general":
                    onoffsets = []
                    onoffsets.append([0.0, samplelist["duration"][file]])
                    samplelist["onoffset"].append(np.array(onoffsets))
                else:
                    meta_file = self._db_path + "/" + clsname + "/" + filename + ".txt"
                    onoffsets = []
                    with open(meta_file, newline="") as meta:
                        meta_reader = csv.reader(meta, delimiter="\t")
                        for onoff in meta_reader:
                            onoffsets.append([float(onoff[0]), float(onoff[1])])

                    samplelist["onoffset"].append(np.array(onoffsets))
            samplelist["onoffset"] = np.squeeze(
                np.array(samplelist["onoffset"], dtype=object)
            )

            for n_class in range(self._nb_classes):
                class_idx = samplelist["class"] == n_class
                samplelist["nSamplesPerClass"] = np.append(
                    samplelist["nSamplesPerClass"], np.sum(class_idx)
                )
                if n_class == 0:
                    samplelist["meanStdDurationPerClass"] = np.array(
                        [
                            [
                                np.mean(samplelist["duration"][class_idx]),
                                np.std(samplelist["duration"][class_idx]),
                            ]
                        ]
                    )
                    samplelist["minMaxDurationPerClass"] = np.array(
                        [
                            [
                                np.min(samplelist["duration"][class_idx]),
                                np.max(samplelist["duration"][class_idx]),
                            ]
                        ]
                    )
                else:
                    samplelist["meanStdDurationPerClass"] = np.vstack(
                        (
                            samplelist["meanStdDurationPerClass"],
                            np.array(
                                [
                                    np.mean(samplelist["duration"][class_idx]),
                                    np.std(samplelist["duration"][class_idx]),
                                ]
                            ),
                        )
                    )
                    samplelist["minMaxDurationPerClass"] = np.vstack(
                        (
                            samplelist["minMaxDurationPerClass"],
                            np.array(
                                [
                                    np.min(samplelist["duration"][class_idx]),
                                    np.max(samplelist["duration"][class_idx]),
                                ]
                            ),
                        )
                    )
            samplelist_per_fold.append(samplelist)

        return samplelist_per_fold

    def _make_selected_filelist(self):
        folds = []
        folds_names = ["train", "test"]  # TODO: make it more generic
        nb_folds = len(folds_names)
        class_list = self._classes  # list(self._classes.keys())

        for ntc in range(self._nb_classes):
            classpath = self._db_path + "/" + class_list[ntc]

            per_fold = []
            for nf in range(nb_folds):
                foldpath = classpath + "/" + folds_names[nf]
                foldcont = os.listdir(foldpath)
                nb_subdirs = len(foldcont)
                filelist = []
                for ns in range(nb_subdirs):
                    subfoldcont = os.listdir(foldpath + "/" + foldcont[ns])
                    for nfl in range(len(subfoldcont)):
                        if subfoldcont[nfl][0] != "." and subfoldcont[nfl].endswith(
                            ".wav"
                        ):
                            filelist.append(
                                class_list[ntc]
                                + "/"
                                + folds_names[nf]
                                + "/"
                                + foldcont[ns]
                                + "/"
                                + subfoldcont[nfl]
                            )
                per_fold.append(filelist)
            folds.append(per_fold)

        return folds
