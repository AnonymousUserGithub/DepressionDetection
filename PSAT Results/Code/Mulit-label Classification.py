import pandas as pd
import numpy as np
from ast import literal_eval

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences



'''''''''''''''' Read Original Dataset '''''''''''''''''''''''

# # #Required only if Train_Test split is not available.
# df = pd.read_csv("D:\\My\\Dataset Creation\\Primate\\Primate_dataset.csv") 
# df.drop(['Unnamed: 0'], inplace = True, axis=1)
# df.info()
# # #literal_eval picks each cell and convert it into a list. Then it will join all its elements with space and convert to string.
# df.loc[:,'Text_Phrases'] = df.loc[:,'Text_Phrases'].apply(lambda x : " ".join(literal_eval(x)))    # apply(literal_eval) (Saravanan\\Word Frequency)
# df.loc[:,'annotations'] = df.loc[:,'annotations'].apply(lambda x : literal_eval(x)) 
# x = df[['post_title', 'post_text', 'Text', 'Text_Phrases']]
# y = df['annotations']

# # average length of sentences.
# length = 0 
# for i in range(0, len(x)):
#     length = length + len(x[i].split())
# print("avg sentence length", length/len(x))

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)  


'''''''''''''''' Saving and Loading Train_Test Split'''''''''''''''''''''''

# train = pd.DataFrame(columns=['post_title', 'post_text', 'Text', 'Text_Phrases'])
# train[['post_title', 'post_text', 'Text', 'Text_Phrases']] = x_train[['post_title', 'post_text', 'Text', 'Text_Phrases']]
# train['y'] = y_train
# train.to_csv("D:\\My\\Dataset Creation\\Primate\\Train Test Split\\Train.csv")

# test = pd.DataFrame(columns=['post_title', 'post_text', 'Text', 'Text_Phrases'])
# test[['post_title', 'post_text', 'Text', 'Text_Phrases']] = x_test[['post_title', 'post_text', 'Text', 'Text_Phrases']]
# test['y'] = y_test
# test.to_csv("D:\\My\\Dataset Creation\\Primate\\Train Test Split\\Test.csv")


df1=pd.read_csv("D:\\My\\Dataset Creation\\Primate\\Train Test Split\\Train.csv")
x_train = df1['Text'].values
y_train = df1.loc[:,'y'].apply(lambda x : literal_eval(x)) 
# #single list contain values for all 9 classes.
ytrain=pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8])
for i in range(0,len(y_train)):     # arranging each question in unique column.
    #if i>0: break   
    for j in range(0,9):
        if y_train[i][j][1] == 'yes':
            ytrain.at[i,j] = 1 
        else: ytrain.at[i,j] = 0      
y_train = np.asarray(ytrain).astype('float32')


df1=pd.read_csv("D:\\My\\Dataset Creation\\Primate\\Train Test Split\\Test.csv")
x_test = df1['Text'].values
y_test = df1.loc[:,'y'].apply(lambda x : literal_eval(x)) 
# #single list contain values for all 9 classes.
ytest=pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8])
for i in range(0,len(y_test)):     # arranging each question in unique column.
    #if i>0: break   
    for j in range(0,9):
        if y_test[i][j][1] == 'yes':
            ytest.at[i,j] = 1 
        else: ytest.at[i,j] = 0
y_test = np.asarray(ytest).astype('float32')



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



'''''''''''''''' Multi-label SVM '''''''''''''''''''''''

from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC

svm = LinearSVC(random_state=42)   # Create the SVM
model = MultiOutputClassifier(svm, n_jobs=-1) # Make it an Multilabel classifier
model = model.fit(x_train_indices_padded, y_train)  # Fit the data to the Multilabel classifier
y_pred = model.predict(x_test_indices_padded)


# from time import time
# t0 = time()
# score_train = model.score(x_train_indices_padded, y_train)
# print(f"\nPrediction time (train): {round(time()-t0, 3)}s")
# t0 = time()
# score_test = model.score(x_test_indices_padded, y_test)
# print(f"\nPrediction time (test): {round(time()-t0, 3)}s")
# print("\nTrain set score:", score_train)
# print("\nTest set score:", score_test)

# from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
# conf_matrix = multilabel_confusion_matrix(y_test, y_pred)  # Generate multiclass confusion matrices
# import matplotlib.pyplot as plt
# cmd = ConfusionMatrixDisplay(conf_matrix[0], display_labels=np.unique(y_test)).plot()   # Plotting matrices: code
# plt.title('Confusion Matrix for label 1 (type)')
# plt.show()



'''''''''''''''' Multi-label Naive Bayes '''''''''''''''''''''''

from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB

naivebayes = GaussianNB()   # Create the SVM
model = MultiOutputClassifier(naivebayes, n_jobs=-1) # Make it an Multilabel classifier
model = model.fit(x_train_indices_padded, y_train)  # Fit the data to the Multilabel classifier
y_pred = model.predict(x_test_indices_padded)


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
model.add(layers.Dense(9, activation='sigmoid'))     # 9 neurons as primate dataset contains output corresponding to 9 questions of PHQ-9.
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
model.add(Dense(9, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()




'''''''''''''''' Execution '''''''''''''''''''''''

history=model.fit(x_train_indices_padded, y_train, validation_split= 0.2, epochs=20)    

loss, accuracy = model.evaluate(x_train_indices_padded, y_train, verbose=False)
print("Training Accuracy: %f" % (accuracy*100))
loss, accuracy = model.evaluate(x_test_indices_padded, y_test, verbose=False)
print("Testing Accuracy: %f" % (accuracy*100))

y_pred = model.predict(x_test_indices_padded)


'''''''''''''''' Multi-Label Metrics '''''''''''''''''''''''
# for Primate dataset


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, multilabel_confusion_matrix
accuracy = accuracy_score(y_test, np.rint(y_pred))
precision_score_sample = precision_score(y_test, np.rint(y_pred), average='samples')
recall_score_sample = recall_score(y_test, np.rint(y_pred), average='samples')
f1_score_sample = f1_score(y_test, np.rint(y_pred), average='samples')
print(f"Accuracy Score = {accuracy*100}")
print(f"Precision (sample) = {precision_score_sample*100}")
print(f"Recall (sample) = {recall_score_sample*100}")
print(f"F1 Score (sample) = {f1_score_sample}")


conf_matrix = multilabel_confusion_matrix(y_test, np.rint(y_pred))


import sklearn
conf_matrix = sklearn.metrics.multilabel_confusion_matrix(y_test, np.rint(y_pred))
print(conf_matrix)
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
precision = 0
recall = 0
for i in range(0,9):
    p_score = precision_score(y_test[i], np.rint(y_pred[i]), average='binary')
    precision = precision + p_score
    #print('Precision: %.3f' % p_score)  
    r_score = recall_score(y_test[i], np.rint(y_pred[i]), average='binary')
    recall = recall + r_score
    #print('Recall: %.3f' % r_score) 
print('Total Precision: %.3f' % precision)
print('Total Recall: %.3f' % recall)
print('Avg Precision: %.3f' % (precision/9))
print('Avg Recall: %.3f' % (recall/9))


'''''''''''''''' Binomial Test '''''''''''''''''''''''
unique, counts = np.unique(y_test, return_counts=True)
unique, counts1 = np.unique(y_pred1, return_counts=True)


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
dist = [binom.pmf(r, n, p) for r in r_values ]


positive= [binom.pmf(r, n, p) for r in range(0,9)]
negative = [binom.pmf(r, n, p) for r in range(0,37)]
print("positive", sum(positive))
print("negative", sum(negative))



'''''''''''''''' Multi-Label Confusion Matrix Visualization '''''''''''''''''''''''

labels = ["".join("PH_Question" + str(i)) for i in range(0, 9)]

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=10):
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("Confusion Matrix for the class - " + class_label)


fig, ax = plt.subplots(3, 3, figsize=(13, 8))
    
for axes, cfs_matrix, label in zip(ax.flatten(), conf_matrix, labels):
    print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])
    fig.tight_layout()
    
fig.show()
fig.savefig("D:\\My\\Dataset Creation\\Phrases Topics\\Attention Visulaization\\SVM-Primate-ConfMatrix2.pdf")





