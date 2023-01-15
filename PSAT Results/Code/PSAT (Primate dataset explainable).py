import numpy as np
import pandas as pd
from ast import literal_eval
import array as arr

from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import sklearn
from sklearn.model_selection import train_test_split 
from tensorflow.keras.layers import Embedding, Input, Dense, LSTM, Bidirectional, Attention
from tensorflow.keras.layers import Layer, Flatten, LayerNormalization, Concatenate
from keras import backend as K
from keras.models import Model


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
x_train = df1['Text_Phrases'].values
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
x_test = df1['Text_Phrases'].values
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




df_ontology = pd.read_csv("D:\\My\\Dataset Creation\\Phrases Topics\\KeyBERT_POS\\Depression Ontology3.csv")  # Diagnosed users are appended in last.
df_ontology.drop(['Unnamed: 0'], inplace = True, axis=1)



categories  = df_ontology.columns
sentence_len = 100   # sequence length or time steps   70 is avg length of phrase embedded sentences in Primate
query_len = 186
embedding_dim = 50


# prepare tokenizer
to_exclude = '!"#$%&()*+-/:;<=>@[\\]^`{|}~\t\n'
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


# # check number of keyphrases with more than 1 word.
# count=0
# for word, i in t.word_index.items():
#     if len(word.split('_'))>1: count = count+1

ontology = []
for i,category in enumerate(categories):   # Representing phrases of each class as stream of phrases.
    ontology.append( " ".join(df_ontology.loc[:,f'{category}'].dropna()))

concept_indices = t.texts_to_sequences(ontology)
concept_indices_padded = pad_sequences(concept_indices, maxlen=query_len, padding='post')



'''''''''''''''' Reading Vectors from Custom-trained .bin embedding file '''''''''''''''''''''''

# Reading from custom trained file Word2Vec_phrase50.bin so binary=True.
model = Word2Vec.load('D:\\My\\Dataset Creation\\Word Vectors\\Trained\\NN_Keyphrase_Embedding\\KeyBERT_n-grams_Word2Vec_skg50.bin')
vocab = list(model.wv.key_to_index)
from numpy import zeros
embedding_matrix = zeros((vocab_size, embedding_dim))
for word, i in t.word_index.items():
    if word in vocab:
        embedding_vector = model.wv[word]  # Slightly different from getting vectors of pre-trained binary file GoogleNews-vectors-negative300.bin
    else:
        #print(word)
        embedding_vector = np.zeros(embedding_dim, dtype = int)
    embedding_matrix[i] = embedding_vector




'''''''''''''''' Reading Vectors from Pre-trained .txt embedding file '''''''''''''''''''''''

# embedding_path = "D:\\My\\Dataset Creation\\Word Vectors\\Downloaded\\glove.6B.50d.txt" ## change 
# # Create the word2vec dict from the dictionary
# def get_word2vec(file_path):
#     file = open(embedding_path, "r", encoding="utf8")
#     if (file):
#         word2vec = dict()
#         split = file.read().splitlines()
#         for line in split:
#             key = line.split(' ',1)[0] # the first word is the key
#             value = np.array([float(val) for val in line.split(' ')[1:]])
#             word2vec[key] = value
#         return (word2vec)
#     else:
#         print("invalid fiel path")

# w2v = get_word2vec(embedding_path)
# #print(Word2Vec.wv.most_similar('this'))
# # get the embedding matrix from the embedding layer
# from numpy import zeros
# embedding_matrix = zeros((vocab_size, embedding_dim))
# for word, i in t.word_index.items():
#     embedding_vector = w2v.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector
        
        

'''''''''''''''' Bahdanau Attention '''''''''''''''''''''''
# simple function is in n-grams Based Explainability.py
# Bahdanau attention adds the query and value/key vectors while simple attention takes only single vector (key/query/value) and multiple with weight matrix and add bias matrix.
# Self-attention multiplies query and key/value vectors.
# https://medium.com/analytics-vidhya/neural-machine-translation-using-bahdanau-attention-mechanism-d496c9be30c3


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)   # initiate W1
        self.W2 = tf.keras.layers.Dense(units)   # initiate W2
        self.V = tf.keras.layers.Dense(1)        # initiate V
        #self.l1=1000
        #self.l2=1000
    def call(self, query, values): #query = Prev output, values = current input   
        # query and value matrix shapes are (None, 4000, 100) 
        query_with_time_axis = tf.expand_dims(query, 1)   # increase the query matrix by 1 dimension (None, 1, 4000, 100)
        # (None, None, 4000, 50) = (None, 1, 4000, 50) + (None, 4000, 50) if W1 and W2 units=50
        # (None, None, 4000, 50) converted to (None, None, 4000, 1) if V units=1
        score = self.V(tf.nn.tanh( self.W1(query_with_time_axis) + self.W2(values)))    # applying the score function proposed by bahdhanau
        #a = score[0][0][:self.l1]
        attention_weights = tf.nn.softmax(score, axis=1)   # apply softmax  (None, None, 4000, 1)
        #b = attention_weights[0][0][:self.l2]
        context_vector = attention_weights * values      # (None, None, 4000, 100)
        context_vector = tf.reduce_sum(context_vector, axis=1)   # (None, 4000, 100)  attention_output
        #return  context_vector, attention_weights, a, b
        return  context_vector, attention_weights

'''''''''''''''' Embedding Layer '''''''''''''''''''''''

word_input = Input(shape=(sentence_len), dtype='float32')   # Input layer with shape = 400 (400 = length of 1 new article)
print("word_input" , word_input.shape)
# Embedding layer = total no. of words, characteristics of each word, embedding matrix, maximum sentence length
embedding_layer = Embedding(input_dim = vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False)  # 
word_sequences = embedding_layer(word_input)   # embedding layer
print("word_sequences",word_sequences.shape)



# Embedding layer = total no. of words, characteristics of each word, embedding matrix, maximum sentence length
embedding_layer1 = Embedding(input_dim = vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length= query_len, trainable=False)  # 
concept_seq0 = embedding_layer1(concept_indices_padded[0])   # embedding layer
concept_seq1 = embedding_layer1(concept_indices_padded[1])   # embedding layer
concept_seq2 = embedding_layer1(concept_indices_padded[2])   # embedding layer
concept_seq3 = embedding_layer1(concept_indices_padded[3])   # embedding layer
concept_seq4 = embedding_layer1(concept_indices_padded[4])   # embedding layer
concept_seq5 = embedding_layer1(concept_indices_padded[5])   # embedding layer
concept_seq6 = embedding_layer1(concept_indices_padded[6])   # embedding layer
concept_seq7 = embedding_layer1(concept_indices_padded[7])   # embedding layer
concept_seq8 = embedding_layer1(concept_indices_padded[8])   # embedding layer
# concept_seq9 = embedding_layer1(concept_indices_padded[9])   # embedding layer
# concept_seq10 = embedding_layer1(concept_indices_padded[10])   # embedding layer
# concept_seq11 = embedding_layer1(concept_indices_padded[11])   # embedding layer
# concept_seq12 = embedding_layer1(concept_indices_padded[12])   # embedding layer
# concept_seq13= embedding_layer1(concept_indices_padded[13])   # embedding layer

print("word_sequences",concept_seq0.shape)


''''''''''''''''''''''' Self-Attention '''''''''''''''''''''''

post_self_attention_op, post_self_attention_wts = BahdanauAttention(50) (word_sequences,word_sequences) 
print("word_attention_op" , post_self_attention_op.shape)

addition = tf.add(post_self_attention_op, word_sequences)
print("addition" , addition.shape)
normalized_post = LayerNormalization(axis=1) (addition)
print("normalized_post" , normalized_post.shape)

'''''''''''''''''''''' Cross Attention  '''''''''''''''''''''

cross_attention_output0, cross_attention_score0 = Attention(name="cross_attention0")([normalized_post,concept_seq0],  return_attention_scores=True, training = True)
print("cross_attention_output0", cross_attention_output0.shape, "cross_attention_seq0", cross_attention_score0.shape)

cross_attention_output1, cross_attention_score1 = Attention(name="cross_attention1")([normalized_post,concept_seq1],  return_attention_scores=True, training = True)
cross_attention_output2, cross_attention_score2 = Attention(name="cross_attention2")([normalized_post,concept_seq2],  return_attention_scores=True, training = True)
cross_attention_output3, cross_attention_score3 = Attention(name="cross_attention3")([normalized_post,concept_seq3],  return_attention_scores=True, training = True)
cross_attention_output4, cross_attention_score4 = Attention(name="cross_attention4")([normalized_post,concept_seq4],  return_attention_scores=True, training = True)
cross_attention_output5, cross_attention_score5 = Attention(name="cross_attention5")([normalized_post,concept_seq5],  return_attention_scores=True, training = True)
cross_attention_output6, cross_attention_score6 = Attention(name="cross_attention6")([normalized_post,concept_seq6],  return_attention_scores=True, training = True)
cross_attention_output7, cross_attention_score7 = Attention(name="cross_attention7")([normalized_post,concept_seq7],  return_attention_scores=True, training = True)
cross_attention_output8, cross_attention_score8 = Attention(name="cross_attention8")([normalized_post,concept_seq8],  return_attention_scores=True, training = True)
# cross_attention_output9, cross_attention_score9 = Attention(name="cross_attention9")([normalized_post,concept_seq9],  return_attention_scores=True, training = True)
# cross_attention_output10, cross_attention_score10 = Attention(name="cross_attention10")([normalized_post,concept_seq10],  return_attention_scores=True, training = True)
# cross_attention_output11, cross_attention_score11 = Attention(name="cross_attention11")([normalized_post,concept_seq11],  return_attention_scores=True, training = True)
# cross_attention_output12, cross_attention_score12 = Attention(name="cross_attention12")([normalized_post,concept_seq12],  return_attention_scores=True, training = True)
# cross_attention_output13, cross_attention_score13 = Attention(name="cross_attention13")([normalized_post,concept_seq13],  return_attention_scores=True, training = True)

# addition1 = tf.add(cross_attention_attention_output0, word_sequences)
# normalized1 = LayerNormalization(axis=1) (addition1)


''''''''''''''''''''''''' Model Training '''''''''''''''''''''''''
concate = Concatenate()([cross_attention_output0, cross_attention_output1, cross_attention_output2, 
                         cross_attention_output3, cross_attention_output4, cross_attention_output5, cross_attention_output6,
                         cross_attention_output7, cross_attention_output8])

print('concate',concate.shape)
flatten = Flatten()(concate)   # Flatten layer  word_attention_op
predictions = Dense(9, activation='sigmoid')(flatten)   # output layer 
print('predictions2',predictions.shape)
model = Model(word_input, predictions)
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(x_train_indices_padded, y_train, validation_split= 0.2, epochs=20)

_, test_acc = model.evaluate(x_test_indices_padded, y_test)
print('\nPREDICTION ACCURACY (%):')
print( 'Test: %.3f' % ( test_acc*100))



'''''''''''''''' Multi-Label Metrics '''''''''''''''''''''''

import sklearn
y_pred = model.predict(x_test_indices_padded)
conf_matrix = sklearn.metrics.multilabel_confusion_matrix(y_test, np.rint(y_pred))
print(conf_matrix)

from sklearn.metrics import precision_score, recall_score


# for Primate dataset
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
fig.savefig("D:\\My\\Dataset Creation\\Phrases Topics\\Attention Visulaization\\PSAT-Primate-ConfMatrix.pdf")


'''''''''''''''' Attention Visualizaation '''''''''''''''''''''''




from IPython.core.display import display, HTML
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
    
def attention2color(attention_score):
    #print(attention_score)
    r = 255 - int(attention_score * 255)
    #print(r)
    color = rgb_to_hex((255, r, r))
    return str(color)


def max_weights(arr):
    max_wts = []
    print("arr len", len(arr))
    for i in arr:
        # print('i',len(i))
         max = np.argmax(i)
         #print('max', max)
         max_wts.append(i[max]) 
    return max_wts

def visualize_attention(idx):
    # Make new model for output predictions and attentions
    model_att = Model(inputs=model.input, outputs=[model.output, model.get_layer('cross_attention7').output])
    #idx = np.random.randint(low = 0, high=X_indices_padded.shape[0]) # Get a random test
    #idx =180
    print('idx :', idx)
    tokenized_sample = np.trim_zeros(x_train_indices_padded[idx]) # Get the tokenized text
    print("length of tokenized_sample", len(tokenized_sample))
    label_probs, attentions = model_att.predict(x_train_indices_padded[idx:idx+1]) # Perform the prediction
    #print(len(attentions[1][0]) )
    print("labels",(label_probs[0][0]))
    # Get decoded text and labels
    id2word = dict(map(reversed, t.word_index.items()))
    print("id2word type", type(id2word))
    decoded_text = [id2word[word] for word in tokenized_sample] 
    # Get classification
    #label = np.argmax((np.array(label_probs[0])>0.5).astype(int).squeeze()) # Only one
    #label2id = ['Not Fake', 'Fake']

    # Get word attentions using attenion vector
    token_attention_dic = {}
    max_score = 0.0
    min_score = 0.0
    max_wts = max_weights(attentions[0][0][:len(tokenized_sample)])
    #max_wts = max_weights(attentions[1][0][0][:len(tokenized_sample)])
    print("max_wts length:", len(max_wts))    
    
    for token, attention_score in zip(decoded_text, max_wts):
        #print(token, attention_score)
        if token in token_attention_dic.keys():         # checks if a token is repeating and if it is repeating then takes largest value of all existences for that token.
            score = token_attention_dic[token]
            #print(f" token is {token} and score is {score}")
            if score < attention_score:
                #print("new value greater", attention_score)
                token_attention_dic[token] = attention_score
        else:
            token_attention_dic[token] = attention_score
            
    a = sorted(token_attention_dic.items(), key=lambda x: x[1],reverse=True) 
    global tokens_list
    tokens_list=token_attention_dic.keys()
    print("length of a:",len(a)) 
    print(a[:20])    
    #print(token_attention_dic)
    # Build HTML String to viualize attentions
    html_text = "<hr><p style='font-size: large'><b>Text:  </b>"
    for token, attention in token_attention_dic.items():
        #print(attention)
        html_text += "<span style='background-color:{};'>{} <span> ".format(attention2color(attention), token)
    #html_text += "</p><br>"
    #html_text += "<p style='font-size: large'><b>Classified as:</b> "
    #html_text += label2id[label] 
    #html_text += "</p>"
    # Display text enriched with attention scores 
    display(HTML(html_text))
    Func = open("D:\\My\\Dataset Creation\\Phrases Topics\\KeyBERT_POS\\PSAT_CustomEmbedding_Category7.html","w")
    Func.write(html_text)
    Func.close()




visualize_attention(390)



# find id of the user document.
for i in range(0, len(x_test)):
    if x_test[i].startswith("Why do I have sudden"):
        print(i)
        
        
Why do I have sudden