from __future__ import print_function
import os
import pickle
import numpy as np
import cv2
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
import pandas as pd
import argparse

def load_csv(file):
    _data = pd.read_csv(file)
    _labels = encoder.fit_transform(_data.dialect.values)
    _values = []
    for _image in _data.image.values:
        img = cv2.imread(_image,0)
        #print (_image, img.shape)
        #img2 = cv2.resize(img, (img_rows, img_cols));
        #print img2.shape
        _values.append(img)
    _values = np.array(_values, dtype=np.uint8)
    return _values, _labels

	
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_csv', help='csv file formatted dialect,image')
    parser.add_argument('out_pickle', help='filename of the output pickle file')
    return parser.parse_args()

if __name__=="__main__":

    args = get_args()
    print ('Loading CSV file ...')
    x_test, y_test = load_csv (args.in_csv)
    
    data=args.out_pickle+'.data'
    print ('Storing features in', data)
    np.save(data,  x_test)
	
    labels=args.out_pickle+'.labels'
    print ('Storing labelss in', labels)
    np.save(labels, y_test)
    