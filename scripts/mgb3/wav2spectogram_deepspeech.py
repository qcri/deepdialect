import librosa
import numpy as np
import scipy.signal
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import argparse
import errno
import json
import os
import time
import sys
from torch.autograd import Variable
import pickle
import scipy as sp
import glob
import argparse


audio_conf = dict(sample_rate='16000', window_size='.02', window_stride='.01', window='hamming')



def wav_to_deepspeech_img(_wavfile):
    y, _ = librosa.core.load(_wavfile,'16000')
    n_fft = int(16000*0.02)
    win_length = n_fft
    hop_length = int(16000*0.01)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hamming')
    spect, phase = librosa.magphase(D)
    spect = np.log1p(spect)
    spect = torch.FloatTensor(spect)
    mean = spect.mean()
    std = spect.std()
    spect.add_(-mean)
    spect.div_(std)
    specto_np=spect.numpy()
    return specto_np
	

def process_folder(in_folder, out_folder):
    if not os.path.exists(out_folder):  os.makedirs(out_folder)
    files = glob.glob(in_folder+'/*.wav')
    for infile in files:
        outfile = out_folder + '/' + os.path.basename(infile).replace(".wav", ".deepSpeech.jpg")
        img = wav_to_deepspeech_img(infile)
        print 'Processing: ', infile, outfile, img.shape
        sp.misc.imsave(outfile, img)
		
		


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='wav to spectogram using deep speech implementation')
    parser.add_argument("-i", "--inwavfolder", help='input 16K hz wave file', type=str, required=True)
    parser.add_argument("-o", "--outimagefolder", help='output spectogram', type=str, required=True)
    
    args = parser.parse_args()
	
    process_folder (args.inwavfolder,args.outimagefolder)
