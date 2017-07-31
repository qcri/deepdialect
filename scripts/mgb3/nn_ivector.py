from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
from keras.utils import to_categorical
from results import report
from data_helpers import load_data


print 'Loading data'
train_vec, train_labels, dev_vec, dev_labels, test_vec, test_labels, vocabulary_size = load_data(type="ivec")
    
#model and train
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=400))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(5, activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model, iterating on the data in batches of 32 samples
model.fit(train_vec, train_labels, epochs=20, batch_size=32)



#evaluate the model
scores = model.evaluate(test_vec, test_labels)

#print scores
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#calculate predictions
pred = model.predict(test_vec,batch_size=32, verbose=10)
np.savetxt('pred.out', pred, delimiter=' ') 

report (pred,test_labels)
