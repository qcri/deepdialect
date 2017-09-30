#!/usr/bin/python

import os, re, wave, glob, sys, shutil
import argparse
import subprocess

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputFolder', help='the folder that contains dialects')
    parser.add_argument('filename', help='filename of the input audio')
    parser.add_argument('outputFolder', help='the output folder')
    parser.add_argument('interval', type=int, help='length of segmentations')
    return parser.parse_args()

if __name__=="__main__":

    args = get_args()

    #audio filename = folder_no.wav
    inputFolder = args.inputFolder
    audio_name = args.filename
    audio_name_path = os.path.join(inputFolder, audio_name)
    audio_reader = wave.open(audio_name_path, 'rb')

    params = audio_reader.getparams()
    frames_len = audio_reader.getnframes()
    sample_rate = audio_reader.getframerate()
    interval = args.interval
    start = 0
    current_pos = interval * sample_rate

    segment_no = 1

    outputFolder = args.outputFolder
    while(current_pos <= frames_len):

        segment_audio = "seg" + str(interval) + "_" + str(segment_no) + "_" + audio_name
        segment_audio_path = os.path.join(outputFolder, segment_audio)
        audio_writer = wave.open(segment_audio_path, 'wb')
        audio_writer.setparams(params)

        audio_reader.setpos(start)
        segment = audio_reader.readframes(current_pos - start)
        audio_writer.writeframes(segment)
        audio_writer.close()
        print segment_audio_path + " has been written."
        start = current_pos
        current_pos += interval * sample_rate
        segment_no += 1

    audio_reader.close()
