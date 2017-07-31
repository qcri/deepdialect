from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from results import report
from data_helpers import load_data
import numpy as np

batch_size=32
nb_epoch=50

print 'Loading data'
train_vec, train_labels, dev_vec, dev_labels, test_vec, test_labels, vocabulary_size = load_data(type="ivec")
    
#model and train
inputs   = Input(shape=(400,))
fc1      = Dense(64, activation='relu')(inputs)
dropout1 = Dropout(0.2) (fc1)
fc2      = Dense(32, activation='relu')(dropout1)
#dropout2 = Dropout(0.2) (fc2)
outputs  = Dense(5, activation='softmax') (fc2)

model = Model(input=inputs, output=outputs)


earlystopper = EarlyStopping(monitor='val_loss', min_delta=0,
                             patience=1, verbose=1, mode='auto')

checkpoint = ModelCheckpoint(filepath="./weights_dnn_ivec.hdf5", monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.summary()

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_vec, train_labels, batch_size=batch_size, epochs=nb_epoch, verbose=1, 
          callbacks=[checkpoint,earlystopper], validation_data=(dev_vec, dev_labels))  

#evaluate the model
scores = model.evaluate(test_vec, test_labels)

#print scores
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#calculate predictions
pred = model.predict(test_vec,batch_size=32, verbose=10)
np.savetxt('pred.out', pred, delimiter=' ') 

report (pred,test_labels)
