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
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
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
best_model="./weights_cnn_char.hdf5"
ext='cnn_char'


print 'Loading data'
train_vec, train_labels, dev_vec, dev_labels, test_vec, test_labels, vocabulary_size = load_data(type="words",maxlen=sequence_length)


# this returns a tensor
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(output_dim=embedding_dim, input_dim=int(vocabulary_size), input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

conv_0 = Convolution2D(num_filters, filter_sizes[0], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)
conv_1 = Convolution2D(num_filters, filter_sizes[1], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)
conv_2 = Convolution2D(num_filters, filter_sizes[2], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)

maxpool_0 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_0)
maxpool_1 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_1)
maxpool_2 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_2)

merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2], mode='concat', concat_axis=1)
flatten = Flatten()(merged_tensor)
dropout = Dropout(drop)(flatten)


output = Dense(output_dim=5, activation='softmax')(dropout)

# this creates a model that includes
model = Model(input=inputs, output=output)


earlystopper = EarlyStopping(monitor='val_loss', min_delta=0,
                             patience=1, verbose=1, mode='auto')

checkpoint = ModelCheckpoint(filepath=best_model, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.summary()

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_vec, train_labels, batch_size=batch_size, epochs=nb_epoch, verbose=1, callbacks=[checkpoint,earlystopper], validation_data=(dev_vec, dev_labels))  # starts training



#evaluate the model
scores = model.evaluate(test_vec, test_labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


#detailed report
report (test_vec,test_labels,best_model,ext)
