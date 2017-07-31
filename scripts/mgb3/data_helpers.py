import numpy as np
import re, sys, glob
import itertools
from collections import Counter
from keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


dialects = ['EGY','GLF','LAV','MSA','NOR']

def clean_str(string):
    '''
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    '''
    string=string.strip() # strip the new line
    string=' '.join(string.split()[1:]) #remove the first column
    string=string.replace("<UNK>","U")
    string = re.sub(r"^", "S ", string)
    string = re.sub(r"$", " E", string)
    return string.strip()


def read_file (word_file): 
    '''
    Read each word file and return clean words list.
    '''
    content = list(open(word_file, "r").readlines())
    content = [clean_str(s) for s in content]
    return content
    
def create_vectorizer(train_files):
    '''
    Loads training data from all files.
    Returns vectorizers for test and dev
    '''
    train_files+="/*.words"
    files = glob.glob(train_files)
    alltext = []
    # iterate over the training data to vectorize it
    for fle in files:
        content = read_file(fle)
        alltext+=content
    
    vectorizer = text.Tokenizer(num_words=None,             
                                lower=False,
                                split=" ",
                                char_level=True)
    vectorizer.fit_on_texts(alltext)
    
    return vectorizer, alltext
    
def make_vectors (vectorizer,words_files,maxlen):
    
    vectors = np.empty((0,maxlen))
    labels = []
    
    # loop over the five dialects 
    for i,lang in enumerate(dialects):
        fle = words_files+'/'+lang+'.words'
                
        content = read_file(fle) 
        charvector= vectorizer.texts_to_sequences(content)
        charvector_pad = pad_sequences(charvector, maxlen=maxlen)
        a = np.empty(len(charvector_pad));a.fill(i)
    
        vectors = np.append(vectors, charvector_pad,axis=0)
        labels = np.append(labels, a, axis=0)
        
    one_hot_labels = to_categorical(labels, num_classes=5)
    return vectors, one_hot_labels
        

def load_ivectors (ivec_files):
    
    
    #load data
    vectors = np.empty((0,400))
    labels = []
    
    for i,lang in enumerate(dialects):
        fle = ivec_files+'/'+lang+'.ivec'
        ivector = np.loadtxt(fle,usecols=range(1,401),dtype='float32')
        a = np.empty(len(ivector));a.fill(i)
        
        vectors = np.append(vectors, ivector,axis=0)
        labels = np.append(labels, a, axis=0)
    one_hot_labels = to_categorical(labels, num_classes=5)
    return vectors, one_hot_labels
    

def build_vocab(sentences):
    '''
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    '''
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]



def load_data(type="words",maxlen=100):
    '''
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    '''
    # Load and preprocess data
    max_features=0 
    if type == "words":
        vectorizer, alltext_clean = create_vectorizer ('../../data/mgb3/train')
    
        print "Train Data..."
        trn_vec, trn_labels = make_vectors (vectorizer, '../../data/mgb3/train',maxlen)
    
        print "Dev Data..."  
        dev_vec, dev_labels = make_vectors (vectorizer, '../../data/mgb3/dev',maxlen)
    
        print "Test Data..."
        tst_vec, tst_labels = make_vectors (vectorizer, '../../data/mgb3/test',maxlen)

        max_features = max([max(x) for x in trn_vec] + 
                           [max(x) for x in dev_vec] +
                           [max(x) for x in tst_vec]) + 1
                   
    if type == "ivec":
        print "Train Data..."
        trn_vec, trn_labels = load_ivectors ('../../data/mgb3/train')
    
        print "Dev Data..."  
        dev_vec, dev_labels = load_ivectors ('../../data/mgb3/dev')
    
        print "Test Data..."
        tst_vec, tst_labels = load_ivectors ('../../data/mgb3/test')
                        
    return trn_vec, trn_labels, dev_vec, dev_labels, tst_vec, tst_labels, max_features
