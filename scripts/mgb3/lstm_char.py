'''
started from https://github.com/bhaveshoswal/CNN-text-classification-keras
TD: add LSTM layer on top of the CNN
'''
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from data_helpers import load_data
from keras.layers import Input
from keras.layers import merge
from keras.layers import  Convolution2D
from keras.layers import  MaxPooling2D
from keras.layers.core import Reshape
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import LSTM
from keras.models import Sequential

import numpy as np
import sys
from results import report

sequence_length=100
drop = 0.5
nb_epoch = 50
batch_size = 32
embedding_dim = 256
filter_sizes = [3,4,5]
num_filters = 512



print 'Loading data'
train_vec, train_labels, dev_vec, dev_labels, test_vec, test_labels, vocabulary_size = load_data(type="words",maxlen=sequence_length)

model = Sequential()
model.add(Embedding(sequence_length,256))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.add(Dense(5, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Convert labels to categorical one-hot encoding


# Train the model, iterating on the data in batches of 32 samples
model.fit(train_vec, train_labels, epochs=3, batch_size=32)


#evaluate the model
scores = model.evaluate(test_vec, test_labels)

#print scores
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#calculate predictions
pred = model.predict(test_vec,batch_size=32, verbose=10)
np.savetxt('pred.out', pred, delimiter=' ') 

report (pred,test_labels)
