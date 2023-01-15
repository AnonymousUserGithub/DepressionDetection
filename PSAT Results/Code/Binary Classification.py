import pandas as pd
import numpy as np
from ast import literal_eval

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences



'''''''''''''''' Read Original Dataset '''''''''''''''''''''''
# Required only if Train_Test split is not available.

# df = pd.read_csv("D:\\My\\Dataset Creation\\CLEF_Complete_n-grams (4000 keyphrases).csv")  # Diagnosed users are appended in last.
# df.drop(['Unnamed: 0'], inplace = True, axis=1)
# df.drop(['Subject_id','Text'], inplace = True, axis=1)
# df.info()
# # literal_eval picks each cell and convert it into a list. Then it will join all its elements with space and convert to string.
# df.loc[:,'Text_Phrases'] = df.loc[:,'Text_Phrases'].apply(lambda x : " ".join(literal_eval(x)))    # apply(literal_eval) (Saravanan\\Word Frequency)
# x = df['Text_Phrases'].values
# y = df['Y'].values

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)      #  stratify=y  




'''''''''''''''' Saving and Loading Train_Test Split'''''''''''''''''''''''

# train = pd.DataFrame(columns=['x_train','y_train'])
# train['x_train'] = x_train
# train['y_train'] = y_train
# train.to_csv("D:\\My\\Dataset Creation\\CLEF\\Train Test Split\\Train.csv")

# train = pd.DataFrame(columns=['x_test','y_test'])
# train['x_test'] = x_test
# train['y_test'] = y_test
# train.to_csv("D:\\My\\Dataset Creation\\CLEF\\Train Test Split\\Test.csv")


df1=pd.read_csv("D:\\My\\Dataset Creation\\CLEF\\Train Test Split\\Train.csv")
x_train = df1['x_Text'].values
y_train = df1['y_train'].values
df1=pd.read_csv("D:\\My\\Dataset Creation\\CLEF\\Train Test Split\\Test.csv")
x_test = df1['x_Text'].values
y_test = df1['y_test'].values



'''''''''''''''' Tokenizer '''''''''''''''''''''''

sentence_len = 2000   # sequence length or time steps    70 is avg length for phrase embedded sentences in Primate dataset
embedding_dim = 50


# prepare tokenizer
to_exclude = '!"#$%&()*+-/:;<=>@[\\]^`{|}~\t\n'    # remove these symbols.
t = Tokenizer(filters=to_exclude)
t.fit_on_texts(x_train)
vocab_size = len(t.word_index) + 1
x_train_indices = t.texts_to_sequences(x_train)
x_test_indices = t.texts_to_sequences(x_test)
x_train_indices_padded = np.asarray(x_train_indices)
x_test_indices_padded = np.asarray(x_test_indices)
# pad documents to a max length of 4 words

x_train_indices_padded = pad_sequences(x_train_indices_padded, maxlen=sentence_len, padding='post')
x_test_indices_padded = pad_sequences(x_test_indices_padded, maxlen=sentence_len, padding='post')
#x_encoded = x_encoded.reshape((x_encoded.shape[0],x_encoded.shape[1],n_features))


'''''''''''''''' Reading Vectors from Custom-trained .bin embedding file '''''''''''''''''''''''

# from gensim.models import Word2Vec
# # Reading from custom trained file Word2Vec_phrase50.bin so binary=True.
# model = Word2Vec.load('D:\\My\\Dataset Creation\\Word Vectors\\Trained\\NN_Keyphrase_Embedding\\KeyBERT_n-grams_Word2Vec_skg50.bin')
# vocab = list(model.wv.key_to_index)
# from numpy import zeros
# embedding_matrix = zeros((vocab_size, embedding_dim))
# for word, i in t.word_index.items():
#     if word in vocab:
#         embedding_vector = model.wv[word]  # Slightly different from getting vectors of pre-trained binary file GoogleNews-vectors-negative300.bin
#     else:
#         #print(word)
#         embedding_vector = np.zeros(embedding_dim, dtype = int)
#     embedding_matrix[i] = embedding_vector



'''''''''''''''' Reading Vectors from Pre-trained .txt embedding file '''''''''''''''''''''''

embedding_path = "D:\\My\\Dataset Creation\\Word Vectors\\Downloaded\\glove.6B.50d.txt" ## change 
# Create the word2vec dict from the dictionary
def get_word2vec(file_path):
    file = open(embedding_path, "r", encoding="utf8")
    if (file):
        word2vec = dict()
        split = file.read().splitlines()
        for line in split:
            key = line.split(' ',1)[0] # the first word is the key
            value = np.array([float(val) for val in line.split(' ')[1:]])
            word2vec[key] = value
        return (word2vec)
    else:
        print("invalid fiel path")

w2v = get_word2vec(embedding_path)
#print(Word2Vec.wv.most_similar('this'))
# get the embedding matrix from the embedding layer
from numpy import zeros
embedding_matrix = zeros((vocab_size, embedding_dim))
for word, i in t.word_index.items():
    embedding_vector = w2v.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
        

'''''''''''''''' SVM '''''''''''''''''''''''

from sklearn.svm import SVC  
model = SVC(kernel='linear') 
# Train the SVM, optimized by Stochastic Gradient Descent 
model.fit(x_train_indices_padded, y_train) # train_corpus_target is the correct values for each training data.
y_pred = model.predict(x_test_indices_padded)



# from sklearn.metrics import confusion_matrix
# confusion_matrix(y_test, y_pred)
# from time import time
# t0 = time()
# score_train = model.score(x_train_indices_padded,y_train)
# print(f"\nPrediction time (train): {round(time()-t0, 3)}s")
# t0 = time()
# score_test = model.score(x_test_indices_padded, y_test)
# print(f"\nPrediction time (test): {round(time()-t0, 3)}s")
# print("\nTrain set score:", score_train)
# print("\nTest set score:", score_test)



'''''''''''''''' Naive Bayes '''''''''''''''''''''''

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train_indices_padded, y_train)
y_pred = model.predict(x_test_indices_padded)


# from sklearn.metrics import confusion_matrix
# confusion_matrix(y_test, y_pred)
# from time import time
# t0 = time()
# score_train = model.score(x_train_indices_padded, y_train)
# print(f"\nPrediction time (train): {round(time()-t0, 3)}s")
# t0 = time()
# score_test = model.score(x_test_indices_padded, y_test)
# print(f"\nPrediction time (test): {round(time()-t0, 3)}s")
# print("\nTrain set score:", score_train)
# print("\nTest set score:", score_test)







'''''''''''''''' MLP '''''''''''''''''''''''

from keras.models import Sequential
from keras import layers

#embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim,weights=[embedding_matrix],input_length=sentence_len))
#model.add(layers.Flatten())
model.add(layers.GlobalMaxPool1D()) #either Flatten or GlobalMaxPool1D
model.add(layers.Dense(128, activation='tanh'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))     # 9 neurons as primate dataset contains output corresponding to 9 questions of PHQ-9.
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()





'''''''''''''''' textCNN '''''''''''''''''''''''



from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Embedding, GlobalMaxPooling1D
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim,weights=[embedding_matrix], input_length=sentence_len))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(20, activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()




'''''''''''''''' Execution '''''''''''''''''''''''

history=model.fit(x_train_indices_padded, y_train, validation_split= 0.2, epochs=20)    

loss, accuracy = model.evaluate(x_train_indices_padded, y_train, verbose=False)
print("Training Accuracy: %f" % (accuracy*100))
loss, accuracy = model.evaluate(x_test_indices_padded, y_test, verbose=False)
print("Testing Accuracy: %f" % (accuracy*100))




'''''''''''''''' Binary-Label Metrics '''''''''''''''''''''''

# confusion matrix for neural network
import sklearn
conf_matrix = sklearn.metrics.confusion_matrix(y_test, np.rint(y_pred))
print(conf_matrix)

from sklearn.metrics import accuracy_score, precision_score, recall_score
accuracy = accuracy_score(y_test, np.rint(y_pred))
print('Precision: %.3f' % accuracy)
precision = precision_score(y_test, np.rint(y_pred), average='binary')
print('Precision: %.3f' % precision)
recall = recall_score(y_test, np.rint(y_pred), average='binary')
print('Recall: %.3f' % recall)





'''''''''''''''' Binomial Test '''''''''''''''''''''''
# count number of true and false label samples in y_test.
unique, counts = np.unique(y_test, return_counts=True)
# unique, counts1 = np.unique(y_pred1, return_counts=True)


from scipy.stats import binom
# setting the values
# of n and p
n = 46
p = 0.1
# defining the list of r values
r_values = list(range(n + 1))
# obtaining the mean and variance 
mean, var = binom.stats(n, p)
# list of pmf values
#dist = [binom.pmf(r, n, p) for r in r_values ]
positive= [binom.pmf(r, n, p) for r in range(0,8)]
negative = [binom.pmf(r, n, p) for r in range(0,38)]
print("positive", sum(positive))
print("negative", sum(negative))




