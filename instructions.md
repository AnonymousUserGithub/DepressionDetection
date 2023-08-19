
## Requirements
  	Package
         Python 3.8
    Word Embedding’s
           Datasets
             CLEF eRisk
             Primate
             CAMS



## Working

The example is presented for the PRIMATE dataset. We have four files:

Keyphrase Extraction-1.py \\
Keyphrase Tagging-2.py \\
Phrase Embedding-3.py \\
PSAT-4(Primate dataset explainable).py \\


The key phrases with TF-IDF scores greater than 0.65 assigned by three phrase extraction algorithms, KeyBERT, KeyBART, and KeyBART + POSTags(Keyphrase Extraction-1.py), are examined manually for their relevance to the depression symptoms. The same process was followed for each dataset to develop the respective phrase list. So we will have four phrase lists in total.
The phrase list of a dataset will be used to tag the posts (Keyphrase Tagging-2.py) and manually develop the depression Knowledge Graph for that dataset based on the PHQ-9 questionnaire.
Using tagged posts, phrase embeddings will be trained by importing the Word2Vec skipgram model from the Genism library (Phrase Embedding-3.py). These embeddings will convert the text data into vector form in the PSAT model.
All resources required by PSAT are ready (PSAT-4(Primate dataset explainable).py).

## Phrase Extraction:
Keyphrase Extraction-1.py will be used for the purpose.
KeyBERT Keyphrase Extraction
The model first creates BERT embeddings of the text documents. For this model, a document is a collection of all posts of a social media user.
In the next step, BERT phrase embeddings for predefined length n-grams (static) are found.
The model calculates a cosine similarity score between the phrase embeddings (second step) and the document embeddings (first step) to find the best key phrases.
(1,3) is supplied as n-grams to the extract_keywords method of the KeyBERT class in the Keybert library of Python.

txt1 = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3), top_n=300, stop_words='english', use_mmr=True, diversity=0.7 #

It is applied at the user level, i.e., for each user document, key phrases are found and stored in different files #

for i in range(0,len(df.Text)):   #
    print("Loop is at: ", i)
    if i<134: continue 
    posts = nltk.sent_tokenize(df.Text[i])   print("length of posts is: ", len(posts))      
    doc = "\n".join(posts)
    # Maximal Marginal Relevance ---- Minimize redundancy and maximize the diversity of results
    txt1 = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3), top_n=300, stop_words='english',        use_mmr=True, diversity=0.7) #0.7 high diversity
    txt = pd.DataFrame(txt1, columns =['Phrase', 'Score'])      # convert list of tuples into dataframe 
    txt.to_csv(f"{path}KeyBERT_{i}.csv")


KeyBERT Files from all users will be merged into a single file in “KeyBERT Merge Files” part of the Keyphrase Extraction-1.py.

## Merge CSV files of each user from KeyBERT. 

import pandas as pd
import os

files = os.listdir("...\\KeyBERT\\")
df1 = pd.DataFrame(columns=['Phrase','Score'])
for i in range(0,len(files)):                 
    my_list = re.split(r'_|\.', files[i])     
    file_no = re.findall(r'\d+', my_list[1])  
    if int(file_no[0]) in range(0,79):            
        try:
            df = pd.read_csv(f"...\\New\\KeyBERT\\{files[i]}") 
            df.drop(['Unnamed: 0'], inplace = True, axis=1)
            df1 = pd.concat([df1,df], axis=0)
        except FileNotFoundError:
            pass
df1.to_csv(f"...\\TotalKeyBERT.csv")


## KeyBERT + POSTags

To consider the grammatical structure of the sentences, we have supplied the pos-tagged user documents to KeyBERT.
Along with user documents, KeyphraseCountVectorizer from the keyphrase_vectorizers library in Python is supplied to the extract_keywords method instead of n_grams range to apply the method on dynamic range n_grams.

kw_model = KeyBERT() #load model
vectorizer = KeyphraseCountVectorizer()  #Init default vectorizer.
print(vectorizer.get_params())    #Print parameters
document_keyphrase_matrix = vectorizer.fit_transform(doc1).toarray()
print(document_keyphrase_matrix)
keyphrases = vectorizer.get_feature_names_out() #After learning the keyphrases, they can be returned.
print(keyphrases)

## Maximal Marginal Relevance ---- Minimize redundancy and maximize the diversity of results

txt = kw_model.extract_keywords(doc1, vectorizer=KeyphraseCountVectorizer(),top_n=5000, stop_words='english', use_mmr=True, diversity=0.7) #0.7 high diversity

It is applied at the user level, i.e., for each user document, key phrases are found.

df = pd.read_csv("...\\Primate_Complete.csv")  #Read Primate
df.info()
#df['Text'] = df['Text'].str.replace("[^a-zA-Z#]", " ") #remove quotations
df['Text']= df['Text'].apply(lambda x : " ".join([w.lower() for w in x.split()]))
doc1=[]
for i in range(0,len(df)):  
    posts = nltk.sent_tokenize(df.Text[i])  # Tokenize user document (all posts) into sentences.
    #len(posts)
    for post in posts:
        #print(post)
        doc1.append(post)   


 
## KeyBART

It is applied on the post level, i.e., for each post, key phrases are extracted.


keyphrases = []
for i in range(0,len(df.Text)):  # Selects all posts of a user or one row from the dataset
    #print("Loop is at: ", i)
    #if i<50: continue
    posts = nltk.sent_tokenize(df.Text[i])  # Tokenize user documents (all posts) into sentences. 
    number_of_posts = len(posts)
    print(f"For user number {i}, number of posts are: {number_of_posts}")
    
    for j in range(0, number_of_posts):     # Post level topic extraction
        #if len(posts[j])>1024: continue
        print(f'j is at {j}.....{number_of_posts}....{i}:')
        # posts[j] = re.sub(r'http\S+', '', posts[j])
        posts[j] = " ".join([w.lower() for w in posts[j].split() if len(w)>1])
        if len(posts[j])>1024: continue
        keyphrases.append(generator(posts[j]))


The “ml6team/keyphrase-generation-keybart-inspec” model is considered while applying the KeyBART model.


from transformers import Text2TextGenerationPipeline, AutoModelForSeq2SeqLM, AutoTokenizer
class KeyphraseGenerationPipeline(Text2TextGenerationPipeline):
    def __init__(self, model, keyphrase_sep_token=";", *args, **kwargs):
        super().__init__(
            model=AutoModelForSeq2SeqLM.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )
        self.keyphrase_sep_token = keyphrase_sep_token

    def postprocess(self, model_outputs):
        results = super().postprocess( model_outputs=model_outputs )
        return [[keyphrase.strip() for keyphrase in result.get("generated_text").split(self.keyphrase_sep_token) if keyphrase != ""] for result in results]
                               
## Load pipeline
model_name = "ml6team/keyphrase-generation-keybart-inspec"
generator = KeyphraseGenerationPipeline(model=model_name)


KeyBART does not provide any score, so the frequency for each phrase is calculated in “KeyBART Phrase Count” part of the Keyphrase Extraction-1.py.

import pandas as pd
file1 = open('...\\KeyBART.txt', 'r')
Lines = file1.readlines()
keyphrases = []
for i,line in enumerate(Lines):
    #print(line)
    #if i>1: break
    line = [phrase.strip() for phrase in line.split(',')] # Remove spaces from the start and end of phrase.
    keyphrases.extend(line)
keyphrases_occurrence = {item:keyphrases.count(item) for item in keyphrases}
df_keyphrases_occurrence = pd.DataFrame(columns=['Keyphrase','Occurred'])   
i=0
for k,v in keyphrases_occurrence.items():  # Faster approach to convert dictionary into dataframe.
    df_keyphrases_occurrence.at[i,'Keyphrase'] = k
    df_keyphrases_occurrence.at[i, 'Occurred'] = v
    i=i+1
df_keyphrases_occurrence.to_csv("...\\KeyBART1.csv")   
value = {i for i in keyphrases_occurrence if keyphrases_occurrence[i]==1}


## Filtration
Keyphrases from KeyBART, KeyBERT, KeyBERT POS” of the Keyphrase Extraction-1.py file place all top 3000 phrases from three algorithms into a single file.


import pandas as pd
df = pd.DataFrame(columns=['Bigram','Score'])
df1 = pd.read_csv("...\\KeyBERT.csv")
df1 = df1.sort_values('Score', ascending=False).drop_duplicates('Bigram', keep="first")
df = pd.concat([df,df1[:3000]], ignore_index=True)
df1 = pd.read_csv("...\\KeyBART.csv") 
df1 = df1.sort_values('Score', ascending=False).drop_duplicates('Bigram', keep="first")
df = pd.concat([df,df1[:3000]], ignore_index=True)
df1 = pd.read_csv("...\\KeyBERT_POS.csv")
df1 = df1.sort_values('Score', ascending=False).drop_duplicates('Bigram', keep="first")
df = pd.concat([df,df1[:3000]], ignore_index=True)

Duplicate phrases will be removed to keep the pool of unique phrases.
df = df.sort_values('Score', ascending=False).drop_duplicates('Bigram', keep="first")  
In the next step, phrases with a length of less than three and stopwords will be removed.
df1 = pd.DataFrame(columns=['Bigram','Score'])
for i in range(0,len(df)):
    print(i)
    phrase = df.Bigram[i]
    if len(phrase)>=3 and phrase not in (stop_words):   
        if len(phrase.split())==1:
            if phrase in english_vocab: 
                df1.at[i,'Bigram'] = phrase
                df1.at[i,'Score'] = df.Score[i]
        else:
            df1.at[i,'Bigram'] = phrase
            df1.at[i,'Score'] = df.Score[i]

df1.to_csv("...\\Total_KeyPhrases.csv") 

In the last step, we will check for n-grams in the extracted keyphrases, i.e., the phrases having continuous words in a sentence. This is done in the “Extract n-grams Keyphrases from Total_KeyPhrases” part of the Keyphrase Extraction-1.py.
import pandas as pd

## It is searching for n-grams i.e. it considers all words of a keyphrase are continuous in the text.

df_keyphrase = pd.read_csv("...\\Total_KeyPhrases.csv") 
#df_keyphrase.drop(['Unnamed: 0'], inplace = True, axis=1)
bigrams = list(set(df_keyphrase.Bigram))   # Removing duplicate phrases. 
found = []
found_not = []
for i, keyphrase in enumerate(bigrams):  # Extract sentences having the top keyphrases.
    #if i>1: break
    print(i)
    #print(keyphrase)
    j = 0
    for one_user_posts in all_users_posts: 
        for post in one_user_posts:    
            if f'{keyphrase}' in post:  
                j = 1                       
    if j==0: found_not.append(keyphrase)  
    else:    found.append(keyphrase)
df_result = df_keyphrase[df_keyphrase['Bigram'].isin(found)]    

keyphrases = list(set(df_result.Bigram[:4000])) 
keyphrases = [w for w in keyphrases if len(w)>=3]
keyphrases = list(set(keyphrases))
keyphrases = [word for word in keyphrases if word not in (stop_words)]   # (4790 keyphrases)

df1.to_csv("...\\Total_KeyPhrases.csv") 


PHQ-9 Depression Ontology: Keyphrases received in the filtration step are mapped to the PHQ-9 questionnaire manually.
Each question of the PHQ-9 questionnaire is taken as a category in the ontology, i.e., nine classes for the PHQ-9 questionnaire.
Some depression-related phrases don’t fit into these classes. To handle those phrases, three classes: "Talking disease symptoms, diagnosis," "Antidepressants," and "Relationship related" are also created.
1000 depression-specific phrases were collected by manually examining the diagnosed group’s phrases extracted by KeyBERT, KeyBERT with POSTags, and KeyBART. These phrases are mapped into one of the classes out of twelve and treated as instances of these classes.
The "Talking disease symptoms, diagnosis" class contains the largest number of phrases, 240. This length will be considered while creating key and value vectors in the cross-attention network.
 
## Phrase Tagging:
Keyphrase Tagging-1.py will be used for the purpose. 
User posts will be tokenized, and remove extra spaces or quotation marks.


all_users_posts= []
for i in range(0,len(df.Text)):  # df[df.columns[3]]
    #if i>0: break
    #print("Loop is at: ", i)
    posts = nltk.sent_tokenize(df.Text[i])  # Tokenize user document (all posts) into sentences. CLEF_Control and CLEF_Diagnosed has 940329 and 101293 sentences respectively.
    doc=[] 
    for post in posts:
        post= post.lower() 
        post = contractions.fix(post)  
        post = re.sub("[^a-zA-Z#]", " ", post) # Remove quotations and symbols other than alphabtes.
        post = re.sub(" +", " ", post) # Remove extra spaces
        doc.append(post)  
        #doc1="\n".join(doc)
    all_users_posts.append(doc)

Next, we check user posts for the available phrases from the Total_KeyPhrases.csv file created in Keyphrase Extraction-1.py and tag them.

# Embed each user document with n-grams(keyphrases) and save them to Primate.csv. Only n-grams(keyphrases) which are found in posts checked in previous loop are used in embedding.

list1 = []    
for i, one_user_posts in enumerate(all_users_posts): 
    print(i)
    post_keyphrase = [] 
    for post in one_user_posts:    # Searches a phrase in one post of a user at a time. (post as a document)
        for j, keyphrase in enumerate(keyphrases):   
            #print(j)
            if f'{keyphrase}' in post:  # It is searching for n-grams i.e. it considers all words of a keyphrase are continuous in the text.
                post_keyphrase.append("_".join(keyphrase.split()))             # But words of a keyphrase may not be continuously present in the text thats why it is not finding many keyphrases which are stored in found_not list.
    list1.append(post_keyphrase)

df.insert(loc = 2, column = 'Text_Phrases', value=0)
df['Text_Phrases'] = df['Text_Phrases'].astype('object')
df['Text_Phrases'] = list1
df.to_csv("...\\Primate_pharse_tagged.csv") 



## Phrase Embedding:  
Available word embeddings contain only a few phrases. So we trained our phrase embeddings using the phrase embedded dataset in this case Primate. Phrase Embedding-3.py carries all code required for the purpose. 

Import required files including the Word2Vec model from the Gensim library.

from gensim.models import Word2Vec
import pandas as pd
from ast import literal_eval
Phrase-tagged user posts are tokenized into sentences and passed to the Word2Vec model.
df=pd.read_csv("...\\Primate_pharse_tagged.csv")
df.loc[:,'Text_Phrases'] =df.loc[:,'Text_Phrases'].apply(lambda x : literal_eval(x))
sentences= df.Text_Phrases
sentences = [x.split(" ") for x in sentences if str(x) != 'nan']  # For non-nan strings, convert string of phrases into list of phrases for each user.


# Sentences is list of lists i.e. for each user we have a list of phrases.
model = Word2Vec(sentences, vector_size=50, min_count=1, workers=6, sg=1)
# summarize the loaded model
print(model)
# summarize vocabulary

# Calculated vectors will be stored back in a .bin file.

words=[]
words = list(model.wv.key_to_index)
#print(words)

print(model.wv.get_vecattr("clear", "count"))

X = model.wv[model.wv.key_to_index]

# Save model

model.save('...\\phrase_embedding.bin')


## PSAT Model: 
PSAT-4(Primate dataset explainable).py contains various modules executed by the proposed PSAT model. This file is for the Primate dataset which is a multilabel dataset.
Primate dataset is already provided as a train and test split. The first step is to import the train.csv and test.csv. Then the label will be encoded into vector form.

df1=pd.read_csv("...\\Train.csv")
x_train = df1['Text_Phrases'].values
y_train = df1.loc[:,'y'].apply(lambda x : literal_eval(x)) 
# single list contain values for all 9 classes.
ytrain=pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8])
for i in range(0,len(y_train)):     # arranging each question in unique column.
    #if i>0: break   
    for j in range(0,9):
        if y_train[i][j][1] == 'yes':
            ytrain.at[i,j] = 1 
        else: ytrain.at[i,j] = 0      
y_train = np.asarray(ytrain).astype('float32')


df1=pd.read_csv("...\\Test.csv")
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

Load the PHQ-9 depression ontology.


df_ontology = pd.read_csv("...\\Depression Ontology.csv")  # Diagnosed users are appended in last.
#df_ontology.drop(['Unnamed: 0'], inplace = True, axis=1)

Import and load the phrase vectors from phrase_embedding.bin (developed in Phrase Embedding.py) into embedding_matrix.

# Reading from custom trained file Word2Vec_phrase50.bin so binary=True.


model = Word2Vec.load('...\\phrase_embedding.bin')
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

## Create and compile the PSAT model.



'''''''''''''''' Bahdanau Attention '''''''''''''''''''''''
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

## '''''''''''''''' Embedding Layer '''''''''''''''''''''''
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
concept_seq9 = embedding_layer1(concept_indices_padded[9])   # embedding layer
concept_seq10 = embedding_layer1(concept_indices_padded[10])   # embedding layer
concept_seq11 = embedding_layer1(concept_indices_padded[11])   # embedding layer
#concept_seq12 = embedding_layer1(concept_indices_padded[12])   # embedding layer
#concept_seq13= embedding_layer1(concept_indices_padded[13])   # embedding layer
print("word_sequences",concept_seq0.shape)

''''''''''''''''''''''' Self-Attention '''''''''''''''''''''''
post_self_attention_op, post_self_attention_wts = BahdanauAttention(50) (word_sequences,word_sequences) 
print("word_attention_op" , post_self_attention_op.shape)
addition = tf.add(post_self_attention_op, word_sequences)
print("addition" , addition.shape)
normalized_post = LayerNormalization(axis=1) (addition)
print("normalized_post" , normalized_post.shape)

'''''''''''''''''''''' Cross Attention  '''''''''''''''''''''
#Attention()[query, value/key]
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
cross_attention_output9, cross_attention_score9 = Attention(name="cross_attention9")([normalized_post,concept_seq9],  return_attention_scores=True, training = True)

''''''''''''''''''''''''' Model Training '''''''''''''''''''''''''
concate = Concatenate()([cross_attention_output0, cross_attention_output1, cross_attention_output2, cross_attention_output3, 
                         cross_attention_output4, cross_attention_output5, cross_attention_output6, cross_attention_output7, 
                         cross_attention_output8, cross_attention_output9])

print('concate',concate.shape)
flatten = Flatten()(concate)   # Flatten layer  word_attention_op
predictions = Dense(9, activation='sigmoid')(flatten)   # output layer 
print('predictions2',predictions.shape)
model = Model(word_input, predictions)
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(x_train_indices_padded, y_train, validation_split= 0.2, epochs=20)

In the next step, various multi-label metrics will be calculated in the “Multi-Label Metrics” part of the file.
“Attention Visualization” will create an HTML file highlighting the phrases of a user post based on the attention score assigned to them by different attention layers. 
display(HTML(html_text))
 Uncomment below line for Cross attention layers visualization in loop
 
#Func = open(f"...\\PSAT_Primate_Phrases{idx}_CustomEmbedding_Category{category}.html","w")  
Func = open(f"...\\SelfAttention_Primate_Test{idx}_Phrases_CustomEmbedding.html","w")  # Print 
Func.write(html_text)
Func.close()

The user post id and attention layer should be passed in the visualize_attention() function (PHQ-9 Class Visualizer), like 140 and cross_attention0 passed in the following example.
visualize_attention(140,'cross_attention0')



 
 



