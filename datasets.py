import os
import numpy
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torch.utils.data as data
import scipy.io.wavfile as wav
import speechpy



def get_train_loader(args):
    dataset = WAVTrainSet(args)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=padding_collate)


def get_test_loader(args):
    dataset = WAVTestSet(args)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=padding_collate_test)

def padding_collate(batch):
    wavs = [i[0] for i in batch]
    targets = torch.LongTensor([i[1] for i in batch])
    max_size = max([wav.shape[1] for wav in wavs])
    wavs_padded = []
    for wav in wavs:
        while wav.shape[1]<max_size:
            shape = wav.shape
            pad = wav[:,:min((max_size-shape[1],shape[1]))]
            wav = numpy.concatenate((wav, pad), axis=1)
        wavs_padded.append(wav)
    wavs_tensor = torch.stack([torch.from_numpy(b) for b in wavs_padded], 0)
    return wavs_tensor, targets

def padding_collate_test(batch):
    wavs = [i[0] for i in batch]
    targets = torch.LongTensor([i[1] for i in batch])
    files = [i[2] for i in batch]
    max_size = max([wav.shape[1] for wav in wavs])
    wavs_padded = []
    for wav in wavs:
        while wav.shape[1]<max_size:
            shape = wav.shape
            pad = wav[:,:min((max_size-shape[1],shape[1]))]
            wav = numpy.concatenate((wav, pad), axis=1)
        wavs_padded.append(wav)
    wavs_tensor = torch.stack([torch.from_numpy(b) for b in wavs_padded], 0)
    return wavs_tensor, targets, files


class WAVTrainSet(data.Dataset):
    def __init__(self, args):
        self.waves = list()
        self.targets = list()
        self.args = args

        # for path, _, image_set in os.walk(os.path.join(args.data_dir, 'train')):
        #     if os.path.isdir(path):
        #         for image in image_set:
        lines = open(args.train_list).readlines()
        for line in lines:
            path, label = line.strip().split(' ')
            self.waves.append(os.path.join(args.data_dir, path))
            self.targets.append(int(label))
        self.presave = [None]*len(self.waves)

    def __getitem__(self, index):
        if not self.presave[index] is None:
            return self.presave[index], self.waves[index]
        fs, signal = wav.read(self.waves[index])
        #signal = signal[:,0]
        signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.98)
        mfcc = speechpy.feature.mfcc(signal_preemphasized, sampling_frequency=fs, frame_length=0.025, num_cepstral=23, frame_stride=0.025,num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
        mfcc_wcmvn = speechpy.processing.cmvnw(mfcc, win_size=121)
        mfcc_wcmvn = numpy.swapaxes(mfcc_wcmvn, 0, 1).astype(numpy.float32)
        self.presave[index] = mfcc_wcmvn
        return mfcc_wcmvn, self.targets[index]
    def __len__(self):
        return len(self.targets)


class WAVTestSet(data.Dataset):
    def __init__(self, args):
        self.waves = list()
        self.targets = list()
        self.args = args

        lines = open(args.test_list).readlines()
        for line in lines:
            path, label = line.strip().split(' ')
            self.waves.append(os.path.join(args.data_dir, path))
            self.targets.append(int(label))
        self.presave = [None]*len(self.waves)

    def __getitem__(self, index):
        if not self.presave[index] is None:
            return self.presave[index], self.waves[index]
        fs, signal = wav.read(self.waves[index])
        #signal = signal[:,0]
        signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.98)
        mfcc = speechpy.feature.mfcc(signal_preemphasized, sampling_frequency=fs, frame_length=0.025, num_cepstral=23, frame_stride=0.025,num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
        mfcc_wcmvn = speechpy.processing.cmvnw(mfcc, win_size=121)
        mfcc_wcmvn = numpy.swapaxes(mfcc_wcmvn, 0, 1)
        self.presave[index] = mfcc_wcmvn
        return mfcc_wcmvn, self.targets[index], self.waves[index]

    def __len__(self):
        return len(self.targets)
