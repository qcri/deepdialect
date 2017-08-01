from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.models import load_model
import h5py
import math
import numpy as np
import sys

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

    


def report (test_vec,test_labels,best_model,ext):
    
    prob_out='prob.out.'+ext
    pred_out='pred.out.'+ext
    class_out='class.out.'+ext
    
    #load models
    model = load_model(best_model)
    
    #save predictions
    pred = model.predict(test_vec)
    np.savetxt(pred_out, pred, delimiter=' ')    
    
    prob = np.copy(pred)
    i=0
    for row in prob: 
      raw = [sigmoid(i) for i in row]
      norm = [float(i)/sum(raw) for i in raw]
      prob[int(i)]=norm
      i=i+1
    
    #save prob
    np.savetxt(prob_out, prob, delimiter=' ') 
    
    
    test_classes = np.argmax(prob, axis=1)
    #save classes
    np.savetxt(class_out, test_classes, delimiter=' ')
    

    test_labels = np.argmax(test_labels, axis=1) # to remove the Keras to_categorical
    
    

    #print confusion matrix 
    confusion_matrix(test_labels, test_classes)

    #print classification report
    print(classification_report(test_labels, test_classes))


    #print prec, recall and f1
    print "Precision:\t{:0.3f}".format(precision_score(test_labels, test_classes,average='weighted'))
    print "Recall:  \t{:0.3f}".format(recall_score(test_labels, test_classes,average='weighted'))
    print "F1 Score:\t{:0.3f}".format(f1_score(test_labels, test_classes,average='weighted'))


