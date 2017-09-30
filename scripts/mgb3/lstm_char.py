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
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers.core import Reshape
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

import numpy as np
import sys
from results import report

sequence_length=1000
drop = 0.5
nb_epoch = 100
batch_size = 32
best_model="./weights_lstm_char.hdf5"
ext='lst_char'



print 'Loading data'
train_vec, train_labels, dev_vec, dev_labels, test_vec, test_labels, vocabulary_size = load_data(type="words",maxlen=sequence_length)



checkpointer = ModelCheckpoint(filepath=best_model, monitor='val_acc', 
                               verbose=1, save_best_only=True, mode='auto')

earlystopper = EarlyStopping(monitor='val_loss', min_delta=0,
                             patience=1, verbose=1, mode='auto')

inputs = Input(shape=(sequence_length,), dtype='int32')
embed  = Embedding(sequence_length,256) (inputs)
lstm   = LSTM(64, dropout=0.2, recurrent_dropout=0.2) (embed)
fc1    = Dense(32, activation='sigmoid') (lstm)
outputs = Dense(output_dim=5, activation='softmax')(fc1)
model = Model(inputs=inputs, outputs=outputs)

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy'])


model.summary()
              
model.fit(train_vec, train_labels, batch_size=batch_size, 
          epochs=nb_epoch, verbose=10,
          validation_data=(dev_vec, dev_labels),
          callbacks=[checkpointer, earlystopper])


#evaluate the model
scores = model.evaluate(test_vec, test_labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


#detailed report
report (test_vec,test_labels,best_model,ext)


          
