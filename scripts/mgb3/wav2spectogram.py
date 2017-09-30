#basic code from here: https://github.com/pietz/language-recognition/blob/master/Language%20Classifier.ipynb

import scipy as sp
import os
import librosa as lr
import glob
import argparse


def wav_to_img(path, height=192, width=192):
    signal, sr = lr.load(path, res_type='kaiser_fast')
    hl = signal.shape[0]//(width*1.1) #this will cut away 5% from start and end
    spec = lr.feature.melspectrogram(signal, n_mels=height, hop_length=int(hl))
    img = lr.logamplitude(spec)**2
    start = (img.shape[1] - width) // 2
    return img[:, start:start+width]
	

def process_folder(in_folder, out_folder):
    if not os.path.exists(out_folder):  os.makedirs(out_folder)
    dell = in_folder + '*.wav'
    files = glob.glob(in_folder+'/*.wav')
    for infile in files:
        outfile = out_folder + '/' + os.path.basename(infile).replace(".wav", ".jpg")
        print 'Processing: ', infile, outfile
        img = wav_to_img(infile)
        sp.misc.imsave(outfile, img)
		

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='wav to spectogram.')
    parser.add_argument("-i", "--inwavfolder", help='wave file', type=str, required=True)
    parser.add_argument("-o", "--outimagefolder", help='output spectogram', type=str, required=True)
    
    args = parser.parse_args()
	
    process_folder (args.inwavfolder,args.outimagefolder)