<img src="header.png" align="left"/>

# Exercise Sentiment Classification (10 points)

The goal of this example is to classify movie reviews as positive or negative sentiments. This can be used to classify for example social media postings.

Parts of the example are taken from [1]. The code used the Glove model [2].

- [1] [https://stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras/](https://stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras/)
- [2] [https://nlp.stanford.edu/pubs/glove.pdf](https://nlp.stanford.edu/pubs/glove.pdf)


Citation GloVe [4] and dataset [5]:
```
[4] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation.

[5] Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher, Learning Word Vectors for Sentiment Analysis, Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, June 2011, Portland, Oregon, USA, Association for Computational Linguistics, http://www.aclweb.org/anthology/P11-1015

```

**NOTE**

Document your results by simply adding a markdown cell or a python cell (as comment) and writing your statements into this cell. For some tasks the result cell is already available.



```python
BasicNNEpochen = 50
ImprovedModel = 25
improved300epochen = 25
```


```python
# Zeitmessung für die gesammte Notebook-Ausführung
from datetime import datetime # für den TimeStamp
tstart = datetime.now()
```

'''
Versuch TF GPU auf meinem HP 8570w laufen zu lassen. Derzeit ist eine "NVIDIA QUADRO K2000M" verbaut
**Installation Notes GPU**
[Link](https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781)

1) [Install  Visual Studio 2019 Community](https://visualstudio.microsoft.com/de/thank-you-downloading-visual-studio/?sku=Community&rel=16)<br>
Bei der Installation benötigt es keine Workloads, sprich 750 mb reichen

2) [Install Cuda-Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork)<br>
ich habe version cuda 11.0 heruntergeladen

3) [Download & install CUDNN==11.0](https://developer.nvidia.com/rdp/cudnn-download)<br>
die Credentials sind entpsrechend hinterlegt


'''''

# Import of Modules


```python
#!pip install nltk
#
# Import of modules
#
import os
import re
import string
from urllib.request import urlretrieve
import tarfile
import zipfile
from glob import glob

import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense, SpatialDropout1D
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

from datetime import datetime # für den TimeStamp

import random

import re
```


```python
#
# Turn off error messages
#
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=Warning)
```


```python
#
# GPU support
#
import tensorflow as tf
print ( tf.__version__ ) 

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR )
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
```

    2.4.0-dev20200815
    

# Constants


```python
#
# Path and URL constants
#
urlDataSource = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
localExtractionFolder = 'data/moviereviews'
localDataArchive = localExtractionFolder + '/aclImdb_v1.tar.gz'
textData = localExtractionFolder + '/aclImdb/'
```

# Support functions


```python
#
# Load data from URL
#
def download_dataset(url,dataset_file_path,extraction_directory):
    if (not os.path.exists(extraction_directory)):
        os.makedirs(extraction_directory)
    if os.path.exists(dataset_file_path):
        print("archive already downloaded.")
    else:
        print("started loading archive from url {}".format(url))
        filename, headers = urlretrieve(url, dataset_file_path)
        print("finished loading archive from url {} to {}".format(url,filename))

def extract_dataset(dataset_file_path, extraction_directory):
    if (not os.path.exists(extraction_directory)):
        os.makedirs(extraction_directory)
    if (dataset_file_path.endswith("tar.gz") or dataset_file_path.endswith(".tgz")):
        tar = tarfile.open(dataset_file_path, "r:gz")
        tar.extractall(path=extraction_directory)
        tar.close()
    elif (dataset_file_path.endswith("tar")):
        tar = tarfile.open(dataset_file_path, "r:")
        tar.extractall(path=extraction_directory)
        tar.close()
    print("extraction of dataset from {} to {} done.".format(dataset_file_path,extraction_directory) )
```

# Load the data


```python
#
# Download if not already loaded
#
download_dataset(urlDataSource,localDataArchive,localExtractionFolder)
```

    archive already downloaded.
    
#
# Extract from archive
#
extract_dataset(localDataArchive,localExtractionFolder)
# How are the files organized on the file system?

Take a quick look how the files are organized on the file system.

Hier ein visueller Einblick:

![Folder-Strucutre](./imgs/MovieReviewsDataFolderStructure.png)



```python
#
# Collect data from the files
#
def load_texts_labels_from_folders(path, folders):
    print('scanning path {}'.format(path))
    texts,labels = [],[]
    for idx,label in enumerate(folders):
        print('scanning {}'.format(idx))
        for fname in glob(os.path.join(path, label, '*.*')):
            texts.append(open(fname, 'rb').read())
            labels.append(idx)
    return texts, np.array(labels).astype(np.int8)
```


```python
# # Loading of positive and negative examples
#

tstart_task = datetime.now()
classes = ['neg','pos']
x_train,y_train = load_texts_labels_from_folders( textData + 'train/', classes)
x_test,y_test = load_texts_labels_from_folders( textData + 'test/', classes)
tend_task = datetime.now()
print("Der Lauf dieses Tasks dauert: " + str(tend_task-tstart_task))
```

    scanning path data/moviereviews/aclImdb/train/
    scanning 0
    scanning 1
    scanning path data/moviereviews/aclImdb/test/
    scanning 0
    scanning 1
    Der Lauf dieses Tasks dauert: 0:03:40.302222
    

# First checks on the data


```python
#
# Check shapes of data
#
len(x_train),len(y_train),len(x_test),len(y_test)
```




    (25000, 25000, 25000, 25000)




```python
#
# Check data types
#
(type(x_train),type(y_train))
```




    (list, numpy.ndarray)




```python
#
# Check classes
#
np.unique(y_train)
```




    array([0, 1], dtype=int8)




```python
#
# Print some negative examples
#
for index in range (0,1):
    print(x_train[index])
    print("label {}".format(y_train[index]))
    print()
```

    b"Story of a man who has unnatural feelings for a pig. Starts out with a opening scene that is a terrific example of absurd comedy. A formal orchestra audience is turned into an insane, violent mob by the crazy chantings of it's singers. Unfortunately it stays absurd the WHOLE time with no general narrative eventually making it just too off putting. Even those from the era should be turned off. The cryptic dialogue would make Shakespeare seem easy to a third grader. On a technical level it's better than you might think with some good cinematography by future great Vilmos Zsigmond. Future stars Sally Kirkland and Frederic Forrest can be seen briefly."
    label 0
    
    


```python
#
# Print some positive examples
#
for index in range (13001,13002):
    print(x_train[index])
    print("label {}".format(y_train[index]))
    print()

```

    b'"Before Sunrise" is a wonderful love story and has to be among my Top 5 favorite movies ever. Dialog and acting are great. I love the characters and their ideas and thoughts. Of course, the romantic Vienna, introduced in the movie does not exist (you won\'t find a poet sitting by the river in the middle of the night) and it isn\'t possible to get to all the places in only one night, either (especially if you\'re a stranger and it\'s your first night in Vienna). But that\'s not the point. The relationship of the two characters is much more important and this part of the story is not at all unrealistic. Although, nothing ever really happens, the movie never gets boring. The ending is genuinely sad without being "Titanic" or something. Even if you don\'t like love stories you should watch this film! I\'m a little skeptic about the sequel that is going to be released in summer. The first part is perfect as it is, in my opinion.'
    label 1
    
    

# Task: Clean text (1 points)

Write a function called preprocess_text(text) which takes a text piece and **cleans out** the following artifacts:

1. html tags, but leave text between tags intact => why the Tag-Content a? href...?...???
1. punctuations and numbers => Clear
1. single characters => a ,i ... => clear
1. multiple white spaces => clear


```python
def preprocess_text(sen):
    # Transfer into string
    sentence = sen.decode("utf-8") # transform byte to string
    sentence = re.sub(r"\b[a-zA-Z]\b", " ", sentence) #single characters,e.g "a", "I"
    sentence = re.sub(r"\d+", " ", sentence)
    
    sentence = re.sub(r'[^\w\s]',' ',sentence) #punctuations and numbers && html tags, but leave text between tags intact
    #sentence = re.sub(r'[^\d]','',sentence)
    sentence = (re.sub('\s+',' ',sentence)) #multiple white spaces
    return(sentence)
```


```python
#
# Clean all texts
#

x_train_clean = []
print("X-Train-Cleaning")
j = 0
for review in x_train:
    if(j%5000 == 0):
        tnow = datetime.now()
        print(str(j) + "/" +str(len(x_train)))
        print(tnow)
        print()
    x_train_clean.append(preprocess_text(review))
    j = j+1

print("X-Test-Cleaning")
j = 0
x_test_clean = []
for review in x_test:
    if(j%5000 == 0):
        tnow = datetime.now()
        print(str(j) + "/" +str(len(x_test)))
        print(tnow)
        print()
    x_test_clean.append(preprocess_text(review))
    j = j+1
    
x_test = x_test_clean
x_train = x_train_clean
```

    X-Train-Cleaning
    0/25000
    2020-08-22 22:04:05.512986
    
    5000/25000
    2020-08-22 22:04:06.669983
    
    10000/25000
    2020-08-22 22:04:07.807990
    
    15000/25000
    2020-08-22 22:04:08.928984
    
    20000/25000
    2020-08-22 22:04:10.003984
    
    X-Test-Cleaning
    0/25000
    2020-08-22 22:04:11.062985
    
    5000/25000
    2020-08-22 22:04:12.422984
    
    10000/25000
    2020-08-22 22:04:13.469985
    
    15000/25000
    2020-08-22 22:04:14.536984
    
    20000/25000
    2020-08-22 22:04:15.527984
    
    


```python
# CHeck x_train, y_train || x_test, y_tets
```

# Find mean text length


```python
#
# Count length of text strings
#
textLength = []
for index in range (0,len(x_train)):
    textLength.append(len(x_train[index]))

#
# Plot histogram
#
plt.hist(textLength)
lengthArray = np.array(textLength)
print('text character length mean {}'.format(np.mean(lengthArray)))
```

    text character length mean 1247.70244
    


![png](output_30_1.png)


# Convert words into tokens


```python
#
# Split text up into tokens
#
tokenizer = Tokenizer(num_words=10000, lower=True, oov_token='unknwn')
#
# Train tokenizer
#
tokenizer.fit_on_texts(x_train)
```


```python
#
# Convert words into integer sequences
#
x_train_v = tokenizer.texts_to_sequences(x_train)
x_test_v = tokenizer.texts_to_sequences(x_test)
```


```python
# check original sentence
print(x_train[0], len(x_train[0]) )
```

    Story of man who has unnatural feelings for pig Starts out with opening scene that is terrific example of absurd comedy formal orchestra audience is turned into an insane violent mob by the crazy chantings of it singers Unfortunately it stays absurd the WHOLE time with no general narrative eventually making it just too off putting Even those from the era should be turned off The cryptic dialogue would make Shakespeare seem easy to third grader On technical level it better than you might think with some good cinematography by future great Vilmos Zsigmond Future stars Sally Kirkland and Frederic Forrest can be seen briefly  629
    


```python
# check token sequence
print(x_train_v[0], len(x_train_v[0]))
```

    [60, 4, 122, 33, 44, 7445, 1383, 14, 4161, 501, 42, 15, 617, 131, 11, 6, 1271, 453, 4, 1705, 203, 1, 7348, 295, 6, 662, 80, 32, 2096, 1079, 2969, 31, 2, 893, 1, 4, 8, 5075, 460, 8, 2637, 1705, 2, 217, 54, 15, 55, 786, 1287, 826, 223, 8, 40, 95, 120, 1453, 56, 143, 35, 2, 956, 139, 26, 662, 120, 2, 1, 406, 58, 92, 1757, 301, 749, 5, 813, 1, 20, 1707, 630, 8, 125, 70, 19, 228, 99, 15, 46, 47, 613, 31, 677, 82, 1, 1, 677, 369, 3322, 1, 3, 1, 7918, 48, 26, 105, 3304] 105
    


```python
# reverse tokens to text for check
text = tokenizer.sequences_to_texts([x_train_v[0]])
print(text)
```

    ['story of man who has unnatural feelings for pig starts out with opening scene that is terrific example of absurd comedy unknwn orchestra audience is turned into an insane violent mob by the crazy unknwn of it singers unfortunately it stays absurd the whole time with no general narrative eventually making it just too off putting even those from the era should be turned off the unknwn dialogue would make shakespeare seem easy to third unknwn on technical level it better than you might think with some good cinematography by future great unknwn unknwn future stars sally unknwn and unknwn forrest can be seen briefly']
    


```python
#
# Count length of integer sequences (aka word sequences)
#
textLength = []
for index in range (0,len(x_train_v)):
    textLength.append(len(x_train_v[index]))

#
# Plot histogram
#
plt.hist(textLength)
lengthArray = np.array(textLength)
print('vectorized length mean {}'.format(np.mean(lengthArray)))
```

    vectorized length mean 226.28068
    


![png](output_37_1.png)



```python
#
# Get size of vocabulary of tokenizer
#
vocab_size = len(tokenizer.word_index) + 1
print('count of words {}'.format(vocab_size))
```

    count of words 73530
    

# Task: select a proper maximum length of text (1 point)

Set maxlen to a suitable value for the text length. Longer text sequences are cut off, shorter sequences are padded.


```python
# Ich nehme die durchschnittliche Textlänge als Maximum
```


```python
def average(lst): 
    return sum(lst) / len(lst) 
maxlen = int(average(textLength))
```


```python
#
# Pad sequences
#
x_train_v = pad_sequences(x_train_v, padding='post', maxlen=maxlen)
x_test_v = pad_sequences(x_test_v, padding='post', maxlen=maxlen)
```

# Download Glove models
[GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.](https://nlp.stanford.edu/projects/glove/)






```python
gloveUrl = 'http://nlp.stanford.edu/glove.6B.zip'
gloveExtractionFolder = 'data/glove'
gloveDataArchive = gloveExtractionFolder + '/glove.6B.zip'

#
# Select 100 dims for embedding space
#
gloveData = gloveExtractionFolder + '/' + 'glove.6B.100d.txt'
gloveDims = 100
```


```python
def unzip_dataset(dataset_file_path, extraction_directory):  
    if (not os.path.exists(extraction_directory)):
        os.makedirs(extraction_directory)        
    zip = zipfile.ZipFile(dataset_file_path)
    zip.extractall(path=extraction_directory)        
    print("extraction of dataset from {} to {} done.".format(dataset_file_path,extraction_directory) )
```


```python
#
# Execute download
#
if ( not os.path.exists(gloveData)):
    download_dataset(gloveUrl,gloveDataArchive,gloveExtractionFolder)
```


```python
#
# Unzip glove
#
if ( not os.path.exists(gloveData)):
    unzip_dataset(gloveDataArchive,gloveExtractionFolder)
```

# Load glove embeddings into memory


```python
#
# Create dict of glove vectors for each word in glove model
#
embeddings_dictionary = dict()
glove_file = open(gloveData, encoding="utf8")
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()
```


```python
print(len(embeddings_dictionary['the']))
```

    100
    


```python
for word, index in tokenizer.word_index.items():
    if(index%7501==0):
        print(word)
```

    satanic
    rumored
    bleach
    idolized
    beet
    weihenmayer
    allusive
    borrough
    buttermilk
    


```python
embedding_matrix = np.zeros((vocab_size, gloveDims))
print(embedding_matrix.shape)
# 73530 = Vocab dass ich aus den Comments erzeugt habe
# gloveDims = Aus dem pretrained Vector
print(embedding_matrix[2])
print(embedding_matrix[73529])
```

    (73530, 100)
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0.]
    


```python
#
# Copy glove vectors for each word in the tokenizer model
#
embedding_matrix = np.zeros((vocab_size, gloveDims))
for word, index in tokenizer.word_index.items():
    #print(word)
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
```


```python
# check shape
print(embedding_matrix.shape)
```

    (73530, 100)
    

# Task: Print some examples of glove vectors for words (1 point)

Select some random word from the tokenizer and print the glove vectors for those words.


```python
print(len(list(tokenizer.word_index.keys())))
print(len(embedding_matrix))

```

    73529
    73530
    


```python
#
# Result:
#

someNumber = 3

def returnSomeWordsFromTokenizer(number):
    returnList = []
    interimList = list(tokenizer.word_index.keys())
    randomlist = []
    for i in range(0,number):
        n = random.randint(1,len(interimList))
        randomlist.append(interimList[n])
    return(randomlist)
#listRandomWords = returnSomeWordsFromTokenizer(someNumber)

def printVector(someNumber):
    wordingList = returnSomeWordsFromTokenizer(someNumber)
    for word in wordingList:
        print(word)
        print(embeddings_dictionary[word])
        
printVector(someNumber)
```

    counterculture
    [ 0.15349    0.50122   -0.1809    -0.48029    0.02361    0.71774
     -0.21085   -0.75048   -0.0019896  0.82295   -0.21566   -0.38023
     -0.66459   -0.45833    0.21449    0.11166    0.22101   -0.13607
      0.42996   -0.16288    0.36205    0.23642    0.052333  -0.30428
     -0.034045   0.35305    0.22006   -0.97723   -0.062583  -0.16177
      0.18757    0.19923   -0.21413    0.7281    -0.22562   -0.36698
      0.17307   -0.027887   0.32421   -0.93756   -0.41259    0.2157
      0.18947    0.35217    0.12168   -0.3786     1.1456     0.51183
      0.15035    0.26981   -0.11518    0.27657    0.09155   -0.0091759
     -0.17851    0.31174   -0.68437    0.64491    0.46658    0.39113
      0.12987    0.80004    0.10756   -0.57317    0.51623   -0.63218
      0.5533    -0.98339    0.65706    0.033639   0.74332   -0.23175
     -0.91126    0.048412  -0.91224   -0.52944    0.59903   -0.5017
      0.15238   -0.61384   -0.96819    0.37858    0.91837    0.20121
     -0.092778   0.47721   -0.29522   -0.072558  -0.74166   -0.060152
      0.40256    0.0016197 -0.33769   -0.022924  -0.23195   -1.1311
      0.10341   -0.29679   -0.26389    0.61275  ]
    fishburne
    [ 1.6746e-01  4.2512e-01  1.2576e-01 -3.2513e-01 -5.9721e-01 -3.5022e-01
     -7.4792e-01 -3.7055e-02  1.3464e-01  3.9914e-01 -2.1727e-01 -3.6291e-01
     -7.0057e-04  5.1823e-01  3.2649e-01  6.9381e-01  3.6471e-01  1.3151e-01
      1.7224e-01  1.4207e+00 -4.6675e-01  3.9247e-01  8.1819e-01 -2.6004e-01
      1.2936e+00  2.7977e-02  1.1169e-02  1.6041e-02 -2.1958e-01 -2.8038e-01
     -3.7368e-01  7.2446e-02  9.5984e-02 -4.4566e-01  3.2927e-02  1.5410e-01
      7.8031e-01  4.8408e-01  2.2365e-01  1.0928e-01 -6.0704e-01  7.4291e-01
     -3.5327e-01  8.3644e-01  8.2367e-02 -1.6220e-01 -2.3536e-01  1.8365e-01
      3.5595e-01 -4.4021e-01 -4.2086e-01 -5.5423e-01  4.1450e-01  2.8637e-01
     -3.2314e-01  1.2945e-01  7.6242e-01  1.6417e-02 -1.3109e+00 -3.6905e-01
     -4.0728e-01  7.8611e-01  5.6758e-01 -6.4084e-03 -3.6670e-01 -1.5455e-01
      4.4495e-01  4.9802e-01  1.1619e-01  7.2704e-01  2.3589e-01 -2.3045e-01
      2.0196e-01 -3.5146e-01 -1.5242e-01 -1.1204e+00 -1.1823e-01  4.7304e-01
      4.8565e-01  1.4935e-01  1.4280e-01 -7.4218e-01 -2.0743e-01 -3.3888e-01
      1.8099e-01 -1.8395e-01  7.1015e-01 -3.6816e-01  3.1555e-01 -3.6043e-01
      9.5336e-03 -4.8352e-02 -3.0013e-01  4.3276e-01  4.2599e-01 -4.8071e-01
      4.6746e-01  8.8031e-04 -5.1836e-01  5.8260e-02]
    outs
    [-7.4896e-02  1.0483e-01  1.7256e-01 -4.1512e-01  3.1838e-01 -2.0841e-01
     -3.6337e-01 -1.0026e+00 -6.7975e-01  4.6692e-04  9.3301e-01  2.1689e-01
     -1.3602e-01  1.2933e+00  9.4972e-01  6.1987e-02 -3.9168e-01  4.4931e-01
     -2.6990e-01  1.3683e-01  1.0234e+00  8.9271e-02  3.2650e-01  3.7260e-01
      2.8517e-01 -1.7178e-01 -6.2529e-01  2.2280e-01  3.2040e-02 -1.8118e-01
     -5.2776e-01 -3.7023e-01  4.5659e-01  1.8961e-01  7.1485e-01  5.9928e-02
     -1.0007e+00 -7.6334e-03  1.3356e-01  4.9473e-01  5.9656e-01  4.7068e-01
     -1.3467e-01  5.8492e-03 -3.4438e-01 -7.7244e-01  9.0516e-02 -6.8598e-01
      6.2075e-02  2.4427e-01 -9.1825e-01  4.7881e-02 -9.2264e-01  6.7043e-01
     -6.0809e-01 -6.0379e-01 -2.2503e-02  2.6674e-01  1.5148e+00  1.4436e+00
     -3.9155e-01  1.9565e-01  3.2192e-03  2.2683e-01  2.3341e-01 -5.0424e-01
      9.5390e-01 -3.1524e-01 -6.7688e-01 -2.8922e-01 -3.7031e-01  1.9980e-01
     -9.0216e-01 -1.2743e+00  9.4554e-01  7.5753e-01 -9.8235e-01 -1.0237e+00
      8.8260e-02  7.8409e-01 -6.8056e-02  3.4013e-01 -7.8037e-01 -2.4908e-01
     -8.6791e-01  1.8378e-01  3.9646e-01  2.1496e-01 -3.4205e-01  7.5143e-01
     -4.7492e-01 -6.6329e-01 -1.7326e-01 -1.1422e-01 -4.2812e-01 -9.6182e-02
      4.8082e-01  1.3476e+00 -3.5282e-01 -1.5792e-01]
    

# Create a simple model

**Note** how the embedding_matrix is used in the first layer to embed the token integers into vectors.


```python
def createNNModel():
    model = Sequential()
    embedding_layer = Embedding(vocab_size, gloveDims, weights=[embedding_matrix], input_length=maxlen , trainable=False)
    model.add(embedding_layer)
    model.add(Flatten())
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model
```


```python
model = createNNModel()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 226, 100)          7353000   
    _________________________________________________________________
    flatten (Flatten)            (None, 22600)             0         
    _________________________________________________________________
    dense (Dense)                (None, 100)               2260100   
    _________________________________________________________________
    dropout (Dropout)            (None, 100)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 101       
    =================================================================
    Total params: 9,613,201
    Trainable params: 2,260,201
    Non-trainable params: 7,353,000
    _________________________________________________________________
    None
    


```python
history = model.fit(x_train_v, y_train, batch_size=128, epochs=BasicNNEpochen, verbose=1, validation_split=0.2)
```

    Epoch 1/50
    157/157 [==============================] - 10s 66ms/step - loss: 0.8873 - acc: 0.6153 - val_loss: 0.9173 - val_acc: 0.2952
    Epoch 2/50
    157/157 [==============================] - 9s 55ms/step - loss: 0.5446 - acc: 0.7440 - val_loss: 0.8202 - val_acc: 0.5082
    Epoch 3/50
    157/157 [==============================] - 9s 59ms/step - loss: 0.4295 - acc: 0.8240 - val_loss: 0.8694 - val_acc: 0.5340
    Epoch 4/50
    157/157 [==============================] - 9s 57ms/step - loss: 0.3425 - acc: 0.8682 - val_loss: 0.6982 - val_acc: 0.6520
    Epoch 5/50
    157/157 [==============================] - 9s 59ms/step - loss: 0.2607 - acc: 0.9113 - val_loss: 1.0312 - val_acc: 0.5294
    Epoch 6/50
    157/157 [==============================] - 9s 57ms/step - loss: 0.2025 - acc: 0.9392 - val_loss: 1.2538 - val_acc: 0.4522
    Epoch 7/50
    157/157 [==============================] - 9s 59ms/step - loss: 0.1667 - acc: 0.9518 - val_loss: 0.9975 - val_acc: 0.5994
    Epoch 8/50
    157/157 [==============================] - 9s 58ms/step - loss: 0.1218 - acc: 0.9672 - val_loss: 1.1587 - val_acc: 0.5590
    Epoch 9/50
    157/157 [==============================] - 9s 56ms/step - loss: 0.1113 - acc: 0.9698 - val_loss: 1.2280 - val_acc: 0.5696
    Epoch 10/50
    157/157 [==============================] - 9s 59ms/step - loss: 0.0799 - acc: 0.9807 - val_loss: 1.1049 - val_acc: 0.6196
    Epoch 11/50
    157/157 [==============================] - 9s 57ms/step - loss: 0.0640 - acc: 0.9859 - val_loss: 1.3621 - val_acc: 0.5648
    Epoch 12/50
    157/157 [==============================] - 9s 59ms/step - loss: 0.0522 - acc: 0.9891 - val_loss: 1.1008 - val_acc: 0.6446
    Epoch 13/50
    157/157 [==============================] - 9s 57ms/step - loss: 0.0447 - acc: 0.9895 - val_loss: 1.5124 - val_acc: 0.5626
    Epoch 14/50
    157/157 [==============================] - 9s 58ms/step - loss: 0.0406 - acc: 0.9909 - val_loss: 1.4069 - val_acc: 0.5908
    Epoch 15/50
    157/157 [==============================] - 9s 58ms/step - loss: 0.0370 - acc: 0.9924 - val_loss: 1.5579 - val_acc: 0.5648
    Epoch 16/50
    157/157 [==============================] - 9s 57ms/step - loss: 0.0330 - acc: 0.9926 - val_loss: 1.8822 - val_acc: 0.5038
    Epoch 17/50
    157/157 [==============================] - 9s 59ms/step - loss: 0.0296 - acc: 0.9932 - val_loss: 1.3745 - val_acc: 0.6268
    Epoch 18/50
    157/157 [==============================] - 9s 56ms/step - loss: 0.0287 - acc: 0.9936 - val_loss: 1.6702 - val_acc: 0.5716
    Epoch 19/50
    157/157 [==============================] - 9s 59ms/step - loss: 0.0245 - acc: 0.9943 - val_loss: 1.9060 - val_acc: 0.5308
    Epoch 20/50
    157/157 [==============================] - 9s 57ms/step - loss: 0.0256 - acc: 0.9934 - val_loss: 1.9798 - val_acc: 0.5236
    Epoch 21/50
    157/157 [==============================] - 9s 59ms/step - loss: 0.0262 - acc: 0.9940 - val_loss: 1.5581 - val_acc: 0.5958
    Epoch 22/50
    157/157 [==============================] - 9s 58ms/step - loss: 0.0234 - acc: 0.9944 - val_loss: 1.6816 - val_acc: 0.5862
    Epoch 23/50
    157/157 [==============================] - 9s 57ms/step - loss: 0.0205 - acc: 0.9957 - val_loss: 1.5873 - val_acc: 0.6102
    Epoch 24/50
    157/157 [==============================] - 9s 60ms/step - loss: 0.0204 - acc: 0.9954 - val_loss: 1.6525 - val_acc: 0.6016: 0.
    Epoch 25/50
    157/157 [==============================] - 9s 56ms/step - loss: 0.0201 - acc: 0.9951 - val_loss: 1.8625 - val_acc: 0.5768
    Epoch 26/50
    157/157 [==============================] - 9s 60ms/step - loss: 0.0194 - acc: 0.9951 - val_loss: 2.0855 - val_acc: 0.5534
    Epoch 27/50
    157/157 [==============================] - 9s 56ms/step - loss: 0.0196 - acc: 0.9959 - val_loss: 1.8633 - val_acc: 0.5966
    Epoch 28/50
    157/157 [==============================] - 9s 59ms/step - loss: 0.0190 - acc: 0.9951 - val_loss: 2.1258 - val_acc: 0.5324
    Epoch 29/50
    157/157 [==============================] - 9s 59ms/step - loss: 0.0195 - acc: 0.9950 - val_loss: 2.2139 - val_acc: 0.4764
    Epoch 30/50
    157/157 [==============================] - 9s 60ms/step - loss: 0.0694 - acc: 0.9764 - val_loss: 1.4065 - val_acc: 0.5840
    Epoch 31/50
    157/157 [==============================] - 9s 60ms/step - loss: 0.0636 - acc: 0.9820 - val_loss: 1.2498 - val_acc: 0.6648
    Epoch 32/50
    157/157 [==============================] - 9s 56ms/step - loss: 0.0396 - acc: 0.9900 - val_loss: 1.5598 - val_acc: 0.5852
    Epoch 33/50
    157/157 [==============================] - 9s 59ms/step - loss: 0.0319 - acc: 0.9919 - val_loss: 1.6529 - val_acc: 0.5956
    Epoch 34/50
    157/157 [==============================] - 9s 58ms/step - loss: 0.0263 - acc: 0.9936 - val_loss: 1.8480 - val_acc: 0.5812
    Epoch 35/50
    157/157 [==============================] - 9s 59ms/step - loss: 0.0196 - acc: 0.9954 - val_loss: 1.8503 - val_acc: 0.5918
    Epoch 36/50
    157/157 [==============================] - 9s 58ms/step - loss: 0.0191 - acc: 0.9955 - val_loss: 1.8458 - val_acc: 0.5894
    Epoch 37/50
    157/157 [==============================] - 9s 57ms/step - loss: 0.0162 - acc: 0.9957 - val_loss: 1.7054 - val_acc: 0.6192
    Epoch 38/50
    157/157 [==============================] - 9s 58ms/step - loss: 0.0169 - acc: 0.9963 - val_loss: 1.7072 - val_acc: 0.6308
    Epoch 39/50
    157/157 [==============================] - 9s 56ms/step - loss: 0.0135 - acc: 0.9962 - val_loss: 2.1031 - val_acc: 0.5622
    Epoch 40/50
    157/157 [==============================] - 9s 59ms/step - loss: 0.0148 - acc: 0.9967 - val_loss: 1.6787 - val_acc: 0.6368
    Epoch 41/50
    157/157 [==============================] - 9s 56ms/step - loss: 0.0145 - acc: 0.9964 - val_loss: 1.9524 - val_acc: 0.6094
    Epoch 42/50
    157/157 [==============================] - 9s 59ms/step - loss: 0.0143 - acc: 0.9969 - val_loss: 2.0095 - val_acc: 0.5838
    Epoch 43/50
    157/157 [==============================] - 9s 58ms/step - loss: 0.0132 - acc: 0.9967 - val_loss: 1.8293 - val_acc: 0.6302
    Epoch 44/50
    157/157 [==============================] - 9s 58ms/step - loss: 0.0128 - acc: 0.9972 - val_loss: 1.9449 - val_acc: 0.6014
    Epoch 45/50
    157/157 [==============================] - 9s 59ms/step - loss: 0.0303 - acc: 0.9906 - val_loss: 1.5319 - val_acc: 0.6070
    Epoch 46/50
    157/157 [==============================] - 9s 56ms/step - loss: 0.0409 - acc: 0.9867 - val_loss: 1.3891 - val_acc: 0.6544
    Epoch 47/50
    157/157 [==============================] - 9s 59ms/step - loss: 0.0424 - acc: 0.9877 - val_loss: 1.7119 - val_acc: 0.5986
    Epoch 48/50
    157/157 [==============================] - 9s 57ms/step - loss: 0.0329 - acc: 0.9902 - val_loss: 1.4091 - val_acc: 0.6620
    Epoch 49/50
    157/157 [==============================] - 9s 59ms/step - loss: 0.0307 - acc: 0.9915 - val_loss: 1.9541 - val_acc: 0.5620
    Epoch 50/50
    157/157 [==============================] - 9s 58ms/step - loss: 0.0258 - acc: 0.9930 - val_loss: 2.0107 - val_acc: 0.5618
    


```python
score = model.evaluate(x_test_v, y_test, verbose=1)
```

    782/782 [==============================] - 7s 8ms/step - loss: 1.3561 - acc: 0.7007
    


```python
print("test loss:", score[0])
print("test accuracy:", score[1])
```

    test loss: 1.3561043739318848
    test accuracy: 0.7006800174713135
    


```python
def plotResults(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()
```


```python
plotResults(history)
```


![png](output_65_0.png)



![png](output_65_1.png)


# Save model


```python
#
# Save a model for later use
#
from keras.models import model_from_json

prefix = 'results/02_'
modelName = prefix + "model.json"
weightName = prefix + "model.h5"


def handle_model(model,save_model):
    # set to True if the model should be saved
    if save_model:
        model_json = model.to_json()
        with open( modelName , "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights( weightName )
        print("saved model to disk as {} {}".format(modelName,weightName))
        return model
    

    # load model (has to be saved before, is not part of git)    
    if not save_model:
        json_file = open(modelName, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(weightName)
        print("loaded model from disk")        
        return loaded_model
```


```python
model = handle_model(model,True)
```

    saved model to disk as results/02_model.json results/02_model.h5
    

# Task: Improved model based on LSTMs (2 points)

The previous model reaches around 70% of test accuracy. This is not sufficient for your customer. So we need a better model. Research the internet for sentiment analysis models using LSTMs and implement a better version of the model based on this information.

1. Implement an LSTM based model version for sentiment analysis (you can also use a different model if you find publications for it)
1. Document the sources you have found
1. Test the model in comparison to the older model version


```python
#
# Result: new model
#
def createLSTMModel():
    model = Sequential()
    embedding_layer = Embedding(vocab_size, gloveDims, weights=[embedding_matrix], input_length=maxlen , trainable=False)
    model.add(embedding_layer)
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model
```


```python
model2 = createLSTMModel()
```


```python
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model2.summary())
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 226, 100)          7353000   
    _________________________________________________________________
    lstm (LSTM)                  (None, 128)               117248    
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 129       
    =================================================================
    Total params: 7,470,377
    Trainable params: 117,377
    Non-trainable params: 7,353,000
    _________________________________________________________________
    None
    


```python
history = model2.fit(x_train_v, y_train, batch_size=128, epochs=ImprovedModel, verbose=1, validation_split=0.2)
score = model2.evaluate(x_test_v, y_test, verbose=1)
```

    Epoch 1/10
    157/157 [==============================] - 294s 2s/step - loss: 0.6603 - acc: 0.6269 - val_loss: 0.9709 - val_acc: 0.1012
    Epoch 2/10
    157/157 [==============================] - 308s 2s/step - loss: 0.6565 - acc: 0.6541 - val_loss: 1.0503 - val_acc: 2.0000e-04
    Epoch 3/10
    157/157 [==============================] - 320s 2s/step - loss: 0.6586 - acc: 0.6255 - val_loss: 0.9644 - val_acc: 0.0670
    Epoch 4/10
    157/157 [==============================] - 324s 2s/step - loss: 0.6488 - acc: 0.6384 - val_loss: 1.1421 - val_acc: 0.0692
    Epoch 5/10
    157/157 [==============================] - 328s 2s/step - loss: 0.5698 - acc: 0.7049 - val_loss: 0.6065 - val_acc: 0.7262
    Epoch 6/10
    157/157 [==============================] - 327s 2s/step - loss: 0.4047 - acc: 0.8214 - val_loss: 0.7339 - val_acc: 0.6022
    Epoch 7/10
    157/157 [==============================] - 328s 2s/step - loss: 0.3971 - acc: 0.8290 - val_loss: 0.5302 - val_acc: 0.7532
    Epoch 8/10
    157/157 [==============================] - 330s 2s/step - loss: 0.3631 - acc: 0.8440 - val_loss: 0.4832 - val_acc: 0.7762
    Epoch 9/10
    157/157 [==============================] - 338s 2s/step - loss: 0.3461 - acc: 0.8531 - val_loss: 0.4191 - val_acc: 0.8134
    Epoch 10/10
    157/157 [==============================] - 329s 2s/step - loss: 0.3385 - acc: 0.8543 - val_loss: 0.7734 - val_acc: 0.6332
    782/782 [==============================] - 60s 76ms/step - loss: 0.4358 - acc: 0.8030
    


```python
print("test loss:", score[0])
print("test accuracy:", score[1])
```

    test loss: 0.4358437955379486
    test accuracy: 0.8030400276184082
    


```python
plotResults(history)
```


![png](output_75_0.png)



![png](output_75_1.png)



```python
model2 = handle_model(model2,True)
```

    saved model to disk as results/02_model.json results/02_model.h5
    

# Task: Replace 100 d model with 300 d model for embedding (2 points)

Try better embedding model with 300 dimensions instead of the 100 dimension model. Load the different Glove weights, update the vector matrix for the embedding layer and the model structure for the better Glove model.



```python
gloveData = gloveExtractionFolder + '/' + 'glove.6B.300d.txt'
gloveDims = 300
# Load glove embeddings into memory¶
embeddings_dictionary = dict()
glove_file = open(gloveData, encoding="utf8")
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()
```


```python
print(len(embeddings_dictionary['the']))
```

    300
    


```python
embedding_matrix = np.zeros((vocab_size, gloveDims))
print(embedding_matrix.shape)
```

    (73530, 300)
    


```python
embedding_matrix = np.zeros((vocab_size, gloveDims))
for word, index in tokenizer.word_index.items():
    #print(word)
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
print(embedding_matrix.shape)
```

    (73530, 300)
    


```python
# CheckVecot
someNumber = 3
printVector(someNumber)
```

    afloat
    [ 0.34422   -0.26506   -0.079252   0.031283   0.59874   -0.065493
      0.20436   -0.49943   -0.13972   -0.55956    0.4486     0.018748
      0.15387   -0.15662   -0.10055    0.050015  -0.041241  -0.16123
      0.0036883 -0.5018     0.45497   -0.11354    0.49982    0.20653
      0.45676   -0.25825   -0.6354     0.65071   -0.4791     0.98903
     -0.16508   -0.089842   0.24866   -0.10836   -0.061778  -0.11268
     -0.0082934  0.13527    0.33242    0.16828   -0.0065926 -0.38829
      0.15563    0.46685   -0.13671    0.20964    0.67408   -0.020772
     -0.29837    0.086152   0.44294    0.06595    0.62687   -1.032
     -0.11953   -0.47521    0.24561    0.59252   -0.40056    0.18437
      0.16494   -0.46523    0.03746    0.25837    0.44265    0.16254
     -0.24062   -0.25099   -0.17595    0.079686  -0.29148    0.58751
     -0.27016   -0.027613   0.35962   -0.39758    0.15746   -0.097824
      0.34502   -0.014896  -0.20374   -0.076557  -0.63644    0.38698
      0.17411    0.5922    -0.5527    -0.36183   -1.031     -0.023998
      0.45831    0.48744    0.074968  -0.069973   0.66704    0.09502
      0.18715    0.36252   -0.36785    0.12801    0.53296    0.49721
      0.17359   -0.22706    0.22333    0.2548    -0.44647   -0.6041
      0.45916    0.011768  -0.27015    0.27668   -0.2139    -0.18266
     -0.61889   -0.45898   -0.054648  -0.2446     1.0646     0.070255
      0.083094  -0.50917    0.2814     0.37397    0.58886    0.063208
     -0.17362   -0.51374    0.14329    0.042841   0.67236    0.28095
      0.71859   -0.23921   -0.19221    0.52252   -0.22479   -0.0016872
      0.05161    0.24867   -0.096506   0.56128   -0.18748    0.71887
     -0.18911   -0.0069589 -0.19486   -0.43027    0.19377    0.63162
     -0.63657   -0.2581     0.097394   0.028668   0.55637   -0.15201
      0.02874   -0.25599   -0.414      0.77674   -0.12067    0.031001
     -0.42891    0.11528    0.026947  -0.37032   -0.081341  -0.11082
     -0.43534    0.60962    0.19892    0.23497   -0.88816   -0.091824
      0.29709    0.3577    -0.52152   -0.04646    0.2579     0.33193
      0.36402   -0.23909   -0.066567   0.009134   0.37671    0.16916
      0.20277   -0.10296    0.10961   -0.81522   -0.1507     0.47049
     -0.42967   -0.12618    0.0443     0.21577   -0.47502    0.27539
      0.33166   -0.38668    0.25669    0.56311    0.42538    0.38062
     -0.012128  -0.62213    0.55438   -0.3779    -0.18432   -0.36548
     -0.37358   -0.50768   -0.44737   -0.2283     0.52681    0.28638
     -0.2976     0.64648   -0.50483    0.42458    0.77851   -0.024888
     -0.017366   0.21819   -0.19916   -0.028884  -0.070708  -0.43585
      0.034882  -0.31072    0.33546   -0.51941    0.45187    0.10312
      0.13428   -0.49485   -0.065059   0.11176    0.12031    0.11585
      0.31576   -0.15925    0.087157   0.35219   -0.24303   -0.87
      0.88274    0.57477   -0.3535    -0.12302   -0.30824    0.49152
      0.95349    0.58604    0.18025   -0.38005   -0.33418   -0.0052073
      0.051762  -0.13285    0.35131   -0.19148   -0.38588   -0.71162
     -0.03253    0.59714    0.12595   -0.26442   -0.21623   -0.089494
     -0.22367    0.38993   -0.57813   -0.1286     0.45703    0.18785
     -0.095127  -0.14495   -0.096313  -0.24466    0.18571    0.2744
     -0.39264    0.018119   0.033666  -0.4735    -0.022747   0.56863
      0.45451   -0.19363   -0.11478    0.25406   -0.011606  -0.74207
      0.062862  -0.0029674 -0.17173   -0.16862   -0.18928   -0.0079698]
    entails
    [-1.5191e-02  2.3452e-01  8.4705e-01  2.5198e-01 -2.0954e-01 -7.5988e-02
     -6.1987e-02  4.5554e-02 -2.0533e-01 -8.0502e-01 -7.9412e-02  3.6318e-01
     -4.4609e-01 -1.0876e-01 -7.8596e-02 -3.5487e-01 -3.2433e-01  1.8042e-01
      3.3273e-01  4.9945e-01 -1.2461e-01 -1.1183e-01  2.8759e-02  1.3778e-01
      2.7221e-01 -1.8353e-01  9.1130e-02  1.5508e-01 -5.3085e-02  1.8329e-01
      3.7153e-02 -2.1452e-01 -2.2690e-01 -2.2138e-01  5.0748e-01  7.1101e-01
     -1.9713e-01 -3.3172e-01 -1.6835e-01 -1.9568e-01  1.3224e-01  2.1106e-01
      4.3322e-03 -5.3782e-01 -5.8199e-01 -4.1128e-02  3.9912e-01  3.2156e-01
      8.5406e-02 -1.3595e-01  3.5109e-02 -3.1765e-02  3.0471e-01  1.7234e-01
     -2.9088e-01  3.6639e-02  1.5966e-02 -5.4845e-02 -1.6677e-01  3.4363e-01
      1.3326e-01 -8.0590e-02  6.0427e-02 -2.9173e-02 -3.4203e-01  2.3591e-01
      8.5017e-02 -9.3341e-03  4.8843e-02  2.6382e-01 -1.8250e-01 -3.0190e-01
      1.7995e-01  1.0173e-01 -1.1086e-01 -9.8553e-02 -1.8239e-01  2.9476e-01
      1.0187e-01  1.8300e-01 -2.5516e-01 -2.5730e-01  2.4935e-01  3.7113e-02
      1.6069e-02 -1.2790e-02  1.4716e-01  1.0613e-01  1.5199e-02  2.1317e-01
     -1.6429e-01  5.8643e-02 -2.9288e-02  1.4958e-01  2.3517e-01 -7.2482e-01
     -5.6123e-01 -1.7522e-01  1.5367e-02 -2.5993e-01 -2.9494e-01  2.6721e-01
      7.2516e-02 -3.2792e-01  2.0835e-01  1.4637e-01 -2.0518e-02  1.4753e-02
     -7.7045e-01 -3.7078e-01  9.3130e-02  2.6774e-01 -3.7639e-01  2.8466e-02
      2.7820e-01  2.3652e-01  5.3252e-01 -1.1153e-01 -3.0168e-01 -2.4818e-01
      1.8983e-01  1.1895e-01 -1.1011e-01  4.5672e-01  4.0689e-01 -5.7584e-01
      3.4971e-02 -3.5197e-01 -3.1920e-02 -2.1895e-01  6.1983e-01 -7.6562e-02
     -1.2545e-01 -2.0920e-01 -4.1715e-02 -4.2001e-02 -1.9133e-01 -1.4217e-01
     -1.0757e-01 -2.5125e-01  6.0034e-02  8.8137e-02  1.0826e-01 -3.4313e-01
      1.2777e-01 -5.7657e-01  2.1999e-02  2.6155e-01  1.7495e-01  1.4967e-01
     -6.8004e-01 -2.5192e-01  1.6515e-01 -1.0183e-01 -2.6285e-01 -1.8821e-02
     -1.3951e-02  2.6934e-01 -4.8609e-01  2.7080e-01  3.0773e-02  1.4403e-02
     -6.5864e-02  2.3250e-01 -1.8082e-01  9.6556e-03  2.7536e-01  1.5534e-01
     -2.9619e-01  3.4803e-02 -6.7715e-02 -5.1765e-01  1.1225e-01 -4.0156e-01
     -9.1280e-02 -1.5559e-01 -3.0179e-01  4.8044e-02  2.3997e-03 -2.6991e-01
      1.1026e-01 -5.0276e-01 -1.2910e-01 -2.8824e-01 -3.6615e-01  3.3250e-01
      4.1315e-01 -3.6735e-01 -4.2539e-01 -4.9860e-01  7.9448e-02  5.5954e-02
      1.8682e-01 -2.3334e-01 -2.4155e-01  3.1862e-01 -2.3488e-01  1.0961e-02
      2.2385e-01  2.9734e-01 -3.5380e-01 -4.7784e-02 -4.6358e-01  2.5434e-01
      4.5686e-01  5.2244e-01  1.2275e-01 -1.6328e-01  9.5347e-01  1.2479e-01
     -2.3629e-01  7.8439e-02 -2.3833e-01  2.3147e-01  2.1966e-01  4.2991e-01
     -1.6007e-01  1.6299e-02 -2.8615e-01  1.3985e-01 -4.4456e-01  3.4903e-01
     -2.8593e-01  4.8365e-01 -5.8789e-02 -2.9967e-01  3.0205e-02  1.1730e-01
      3.3422e-01  3.7658e-01  7.5374e-02  7.6720e-02  2.5156e-01  8.3570e-02
      3.0454e-02 -3.2020e-01 -2.7266e-01 -2.6569e-01  2.9320e-01  2.1835e-01
      2.4429e-01  3.7565e-01  3.7413e-01 -1.0782e-01 -6.3093e-01 -6.5868e-02
     -9.6351e-02  3.9246e-01  7.6291e-02 -1.8305e-01 -2.9943e-01  3.8202e-01
     -7.9825e-02 -3.7119e-01 -7.6016e-01  2.1384e-01 -1.5437e-01  3.1191e-01
      1.6565e-01  1.1952e-02  1.6615e-01  3.2592e-01  2.1315e-01  4.3658e-01
      3.7486e-01  5.5816e-01 -1.3380e-01  2.2057e-01 -6.8038e-02  7.4322e-02
     -7.4061e-03  4.0802e-01  8.3528e-02  6.2027e-01 -1.4965e-01 -3.3041e-02
     -2.7097e-01 -3.7674e-01  2.4464e-01  2.2432e-01  4.6364e-01 -3.4604e-01
     -4.0671e-01  1.2945e-01  5.8567e-01  1.6738e-01 -1.3344e-04  6.7167e-02
     -1.6051e-01 -1.4018e-01 -5.3929e-01 -4.6060e-02 -1.7579e-03  2.1011e-02
      2.8010e-01 -1.9258e-01 -2.9008e-01  4.4152e-01  1.9391e-01 -3.3442e-01]
    kabei
    


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-46-e2c659274417> in <module>
          1 someNumber = 3
    ----> 2 printVector(someNumber)
    

    <ipython-input-41-a984904bbc8f> in printVector(someNumber)
         19     for word in wordingList:
         20         print(word)
    ---> 21         print(embeddings_dictionary[word])
         22 
         23 printVector(someNumber)
    

    KeyError: 'kabei'



```python
# direkt ins LSTM, das BASIS-NN überspringe ich
model3 = createLSTMModel()
model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model3.summary())
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, 226, 300)          22059000  
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 128)               219648    
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 129       
    =================================================================
    Total params: 22,278,777
    Trainable params: 219,777
    Non-trainable params: 22,059,000
    _________________________________________________________________
    None
    


```python
history_300d = model3.fit(x_train_v, y_train, batch_size=128, epochs=improved300epochen, verbose=1, validation_split=0.2)
score_300d = model3.evaluate(x_test_v, y_test, verbose=1)
```

    Epoch 1/10
    157/157 [==============================] - 608s 4s/step - loss: 0.6571 - acc: 0.6337 - val_loss: 0.6630 - val_acc: 0.3586
    Epoch 2/10
    157/157 [==============================] - 679s 4s/step - loss: 0.6432 - acc: 0.6314 - val_loss: 1.2434 - val_acc: 0.1086
    Epoch 3/10
    157/157 [==============================] - 730s 5s/step - loss: 0.6510 - acc: 0.6254 - val_loss: 1.0861 - val_acc: 0.1712
    Epoch 4/10
    157/157 [==============================] - 753s 5s/step - loss: 0.6332 - acc: 0.6541 - val_loss: 0.7837 - val_acc: 0.3488
    Epoch 5/10
    157/157 [==============================] - 759s 5s/step - loss: 0.6197 - acc: 0.6598 - val_loss: 0.8889 - val_acc: 0.5928
    Epoch 6/10
    157/157 [==============================] - 761s 5s/step - loss: 0.6456 - acc: 0.6555 - val_loss: 1.0388 - val_acc: 0.0102
    Epoch 7/10
    157/157 [==============================] - 765s 5s/step - loss: 0.6401 - acc: 0.6425 - val_loss: 0.9732 - val_acc: 0.0742
    Epoch 8/10
    157/157 [==============================] - 761s 5s/step - loss: 0.6103 - acc: 0.6725 - val_loss: 0.7265 - val_acc: 0.4446
    Epoch 9/10
    157/157 [==============================] - 762s 5s/step - loss: 0.5810 - acc: 0.7096 - val_loss: 0.9680 - val_acc: 0.1250
    Epoch 10/10
    157/157 [==============================] - 761s 5s/step - loss: 0.5121 - acc: 0.7414 - val_loss: 0.6222 - val_acc: 0.7172
    782/782 [==============================] - 159s 204ms/step - loss: 0.4023 - acc: 0.8312
    


```python
print("test loss:", score_300d[0])
print("test accuracy:", score_300d[1])
```

    test loss: 0.40231165289878845
    test accuracy: 0.8312000036239624
    


```python
plotResults(history_300d)
```


![png](output_86_0.png)



![png](output_86_1.png)


# Task: Replace Glove model with BERT model vectors (2 points)

Try to replace Glove with a BERT model. This is no easy task. Research the internet for tutorials about this goal and write down all changes you would need to implement for this change (concept only, implementation optional).


[LINK Basic TUtorial](https://towardsdatascience.com/bert-text-classification-in-3-lines-of-code-using-keras-264db7e7a358)




```python
import matplotlib
```


```python
#!pip install ktrain
import ktrain
from ktrain import text
```


```python
trn, val, preproc = text.texts_from_folder(textData, 
                                          maxlen=500, 
                                          preprocess_mode='bert',
                                          train_test_names=['train', 
                                                            'test'],
                                          classes=['pos', 'neg'])
```

    detected encoding: utf-8
    preprocessing train...
    language: en
    


done.


    Is Multi-Label? False
    preprocessing test...
    language: en
    


done.



```python
model = text.text_classifier('bert', trn, preproc=preproc)
learner = ktrain.get_learner(model,train_data=trn, val_data=val, batch_size=6)
```

    Is Multi-Label? False
    maxlen is 500
    done.
    


```python
# Train & Fine-Tune Model
historyLearner = learner.fit_onecycle(2e-3, 1)
```

    
    
    begin training using onecycle policy with max lr of 0.002...
      49/4167 [..............................] - ETA: 119:58:13 - loss: 0.8397 - accuracy: 0.5215
prefix = 'results/03_BERT_'
modelName = prefix + "model.json"
weightName = prefix + "model.h5"
model = handle_model(learner,True)
Ich habe den oberen Junk ausgeblendet, da dieser auf meiner Hardware unverhältnimäßig lange rechnet. Siehe folgendes Bild:

------------
------------
------------
![Folder-Strucutre](./imgs/ProblemZeit.png)

------------
------------
------------

Ich habe es so gelöst, dass ich den Teil des BERT-Models auf dem leistungsstarken Kubernete Cluster der FH laufen gelassen habe. Das fertige Modell habe ich von dem Cluster auf einen privaten FTP-Server geladen und das Model anschließend auf meine Hardware  geladen. Konkret habe ich folgendes gitalb.ci.yml - File erstellt:

```
stages:
    - MillingerRun
ExecuteBERT:
    stage: MillingerRun
    image: jhc1990/mlunnerk8sfh
    script:
        - mkdir results
        - python code-to-run.py
```

Für den Run habe ich diesen [Docker-Container](https://hub.docker.com/r/jhc1990/ml_exercise_ditmil_imdb) in Eigenarbeit gebaut.

Der tatsächliche Code im File "code-to-run.py" File ist der Code des gegenwärtig betrachteten Notebooks, angereichert um die Implementierung des FTP-Servers, auf welchen das fertig trainierte Modell deployt wurde.

```
import ftplib
session = ftplib.FTP('188.174.171.206','ftpuser','0c5acc0ae8793d0ed4f262c8d8c12adafd3b5104e0e3519b971b35f14596772f')
session.cwd('MillingerML')

file = open('results/03_BERT_model.h5','rb')                  # file to send
session.storbinary('STOR 03_BERT_model.h5', file)     # send the file
file.close()                                    # close file and FTP

file = open('results/03_BERT_model.json','rb')                  # file to send
session.storbinary('STOR 03_BERT_model.json', file)     # send the file
file.close()                                    # close file and FTP

session.quit()
```

Ich lasse den FTP-Server noch online für den Fall, dass dieses fertige Modell eingesehen bzw heruntergeladen werden soll. mit ein bisschen IT-Know-How findet man sicher die nötigen Credentials im oberen Junk heraus :-). 

Hier ein Screenshot von dem abgeschlossenen Run:
![Folder-Strucutre](./imgs/gitlab.png)

Auch hier ist eine "halbe" Ewigkeit verstrichen bis das Model fertig trainiert war(~1200 Minuten = ca 20h // im Vergleich zu 120 Tagen Training auf meiner Hardware aber eigentlich ganz annhembar. 

# Test with your own data


```python
instance = x_test_clean[56]
print(instance)
```

    From time to time it very advisable for the aristocracy to watch some silent film about the harsh life of the common people in order to remind themselves of the privileges and the comfortable life that they have enjoyed since the beginning of mankind or even before in comparison with the complicated and hard work that common people have to endure everyday since the aristocrats rule the world br br And that what happens in The Love Light the first film directed by Dame France Marion who will be famous afterwards in the silent and talkie world thanks overall to her work as screenwriter better for her certainly because her career as film director doesn impress this German count br br The film tells the story of Dame Angela Carlotti Dame Mary Pickford merry Italian girl who lives surrounded by picturesque squalor an important difference of opinion between upper and low classes aristocrats prefers to live surrounded by picturesque luxury she has two brothers and secret admirer but all she gives him in return is indifference Destiny begins to work hard and pretty soon war is declared and Dame Angela two brothers enlist and in the next reel both are dead But destiny is even crueller and Dame Angela meanwhile falls in love with German And to make things worse she doesn know that her Teutonic sweetie is spy and that the light signals that she sends to him every night from the lighthouse she maintains thinking that is love signal don mean Ich Liebe Dich but Sink Any Damn Italian Boat At Sea br br Fortunately for Dame Angela pretty soon her sweetie German spy will be found by the neighbours in her house in which she was hiding him not strange fact indeed because it is not an easy task for German to go unnoticed but the German spy will prefer to die before being captured by those Italians br br From that German love half Teutonic baby will born the wicked Destiny at full speed but greedy neighbour who has particular idea of motherhood will carry away her son with the consent of Catholic nun who has taken the Council of Trent to extremes fact that will put Dame Angela at the verge of insanity br br But meanwhile Dame Angela secret admirer has returned from the war and you can think that finally Dame Angela sorrowful life will improve tremendous mistake because Destiny has in store for her that the returned soldier is blind But as they say in Germany it may be blessing in disguise and finally Dame Angela will recover her son and will start new life with her blind sweetie in poor Italian village in what it is supposed to be happy ending for the common people br br As this German count said before it was much better for Dame Frances Marion that she continued her career as screenwriter because as can be seen in The Love Light she had lot of imagination to invent incredible stories ja wohl but completely different subject is to direct films and her silent debut lacks emotion and rhythm in spite of the effort of Dame Pickford to involve the audience with her many disgraces The nonexistent film narrative causes indifference in the spectator making this the kind of film where only Dame Pickford herself provides the interest and not her circumstances br br And now if you ll allow me must temporarily take my leave because this German Count must send Morse signals from the Schloss north tower to one of his Teutonic rich heiress 
    


```python
def sentiment(text):
    
    instance = tokenizer.texts_to_sequences(text)
    flat_list = []
    for sublist in instance:
        for item in sublist:
            flat_list.append(item)

    flat_list = [flat_list]
    instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)
    sentiment = model2.predict(instance)
    
    comment = 'meh'
    if sentiment > 0.85:
        comment = 'very good'
    elif sentiment > 0.75:
        comment = 'good'
    elif sentiment > 0.50:
        comment = 'moderate'
    return sentiment,comment
```


```python
model2 = handle_model("./results/02_model",False)
```

    loaded model from disk
    


```python
test1 = "I simply don't like this film."
print ( sentiment(test1))
```

    (array([[0.9579619]], dtype=float32), 'very good')
    


```python
test1 = "I hate this film."
print ( sentiment(test1))
```

    (array([[0.93643403]], dtype=float32), 'very good')
    


```python
test1 = "I love this film."
print ( sentiment(test1))
```

    (array([[0.93643403]], dtype=float32), 'very good')
    


```python
test1 = x_test_clean[13000]
print ( sentiment(test1))
```

    (array([[0.81808805]], dtype=float32), 'good')
    


```python
tend = datetime.now()
print(tend)
print("Der Lauf des gesamten Notebooks dauert: " + str(tend-tstart))
```

    2020-08-22 22:13:49.596099
    Der Lauf des gesamten Notebooks dauert: 0:13:33.705048
    
