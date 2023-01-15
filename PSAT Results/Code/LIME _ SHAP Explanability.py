import pandas as pd
import numpy as np
from ast import literal_eval
import sklearn
import sklearn.ensemble
import sklearn.metrics

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline


df = pd.read_csv("D:\\My\\Dataset Creation\\CLEF_Complete_n-grams (4000 keyphrases).csv")  # Diagnosed users are appended in last.
df.drop(['Unnamed: 0'], inplace = True, axis=1)
df.drop(['Subject_id','Text_Phrases'], inplace = True, axis=1)
df.info()
df.loc[:,'Text'] = df.loc[:,'Text'].apply(lambda x : " ".join(literal_eval(x)))  
x = df['Text'].values
y = df['Y'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify=y)  

post = """Why do I have sudden "bursts" of depressionI know the title probably doesn't make sense but I 
stopped working for a while to peruse a business idea i had (which failed) and now I'm about to go back 
into the work force I'm only 19 and I have these moments where i just feel lost and like I failed my 
family and friends as my business is what i dedicated my life to for the past 6 months and most of that 
time was me sitting in my room trying to get it off the ground floor. I'm really nervous about getting a 
job again as i haven't had a real one that entire time am I just overthinking it or will it be not as bad 
as i think. """

x_test=np.insert(x_test,46,post) 
y_test=np.insert(y_test,46,1) 


# # find id of the user document.
# for i in range(0, len(x_train)):
#     if x_train[i].startswith("guess nothing attack"):
#         print(i)

# # find id of the user document.
# for i in range(0, len(x_test)):
#     if y_test[i]==0:
#         print(i)
        
        
        
'''''''''''''''' LIME Explanations '''''''''''''''''''''''

vectorizer = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}', ngram_range=(1, 1), stop_words = 'english', binary=True)
train_vectors = vectorizer.fit_transform(x_train)
test_vectors = vectorizer.transform(x_test)


print(vectorizer.get_feature_names()[10])

logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(train_vectors, y_train)
pred = logreg.predict(test_vectors)
accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, average='weighted')
recall = recall_score(y_test, pred, average='weighted')
f1 = f1_score(y_test, pred, average='weighted')
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))





c = make_pipeline(vectorizer, logreg)
class_names=['control','diagnosed']
explainer = LimeTextExplainer(class_names=class_names)

idx = 46
exp = explainer.explain_instance(x_test[idx], c.predict_proba, num_features=6)
print('Document id: %d' % idx)
print('Predicted class =', class_names[logreg.predict(test_vectors[idx]).reshape(1,-1)[0,0]])
print('True class: %s' % class_names[y_test[idx]])


print ('Explanation for class %s' % class_names[1])
print ('\n'.join(map(str, exp.as_list(label=1))))



exp = explainer.explain_instance(x_test[idx], c.predict_proba, num_features=6, top_labels=2)
print(exp.available_labels())

exp.show_in_notebook(text=False)

exp.show_in_notebook(text=y_test[idx], labels=(0,))

exp.save_to_file("D:\\My\\Dataset Creation\\Phrases Topics\\Attention Visulaization\\Primate Sample\\LIME_Primate.html")



'''''''''''''''' SHAP Explanations '''''''''''''''''''''''


import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import keras.backend as K
K.set_session
import shap

class TextPreprocessor(object):
    def __init__(self, vocab_size):
        self._vocab_size = vocab_size
        self._tokenizer = None
        self.to_exclude = '!"#$%&()*+-/:;<=>@[\\]^`{|}~\t\n'    # remove these symbols.
    def create_tokenizer(self, text_list):
        tokenizer = Tokenizer(num_words = self._vocab_size)
        tokenizer = Tokenizer(filters=self.to_exclude,num_words = self._vocab_size)
        tokenizer.fit_on_texts(text_list)
        self._tokenizer = tokenizer
    def transform_text(self, text_list):
        text_matrix = self._tokenizer.texts_to_matrix(text_list)
        return text_matrix



VOCAB_SIZE = 1000
class_names=['control','diagnosed']
num_tags = 1        # As only two class so we need only one neuron in the output layer.
train_post = x_train
test_post = x_test
processor = TextPreprocessor(VOCAB_SIZE)
processor.create_tokenizer(train_post)
train_post = processor.transform_text(train_post)
test_post = processor.transform_text(test_post)

def create_model(vocab_size, num_tags):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(50, input_shape = (VOCAB_SIZE,), activation='relu'))
    model.add(tf.keras.layers.Dense(25, activation='relu'))
    model.add(tf.keras.layers.Dense(num_tags, activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    return model

model = create_model(VOCAB_SIZE, num_tags)      # As only two class so we need only one neuron in the output layer.
model.fit(train_post, y_train, epochs = 2, batch_size=128, validation_split=0.1)
print('Eval loss/accuracy:{}'.format(model.evaluate(test_post, y_test, batch_size = 128)))


attrib_data = train_post[:100]
shape_explainer = shap.DeepExplainer(model, attrib_data)
num_explanations = 46
shap_vals = shape_explainer.shap_values(test_post[:num_explanations])
words = processor._tokenizer.word_index
word_lookup = list()

for i in words.keys():
    word_lookup.append(i)

word_lookup = [''] + word_lookup
shap.summary_plot(shap_vals, feature_names=word_lookup, class_names=class_names)












