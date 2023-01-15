import pandas as pd
import numpy as np
from ast import literal_eval

import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm




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


# from sklearn.model_selection import train_test_split
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



df_train = pd.read_csv("D:\\My\\Dataset Creation\\Primate\\Train Test Split\\Train.csv")
df_train.loc[:,'y'] = df_train.loc[:,'y'].apply(lambda x : literal_eval(x)) 
for i in range(0,len(df_train)):     # arranging each question in unique column.
    result = []
    #if i>0: break   
    for j in range(0,9):
        if df_train['y'][i][j][1] == 'yes':
            result.append(1) 
        else: result.append(0)
    df_train.at[i,'y'] = result


df_test = pd.read_csv("D:\\My\\Dataset Creation\\Primate\\Train Test Split\\Test.csv")
df_test.loc[:,'y'] = df_test.loc[:,'y'].apply(lambda x : literal_eval(x)) 
for i in range(0,len(df_test)):     # arranging each question in unique column.
    result = []
    #if i>0: break   
    for j in range(0,9):
        if df_test['y'][i][j][1] == 'yes':
            result.append(1) 
        else: result.append(0)
    df_test.at[i,'y'] = result


train_dataset = df_train.loc[:1599, ['Text','y']]
test_dataset = df_test.loc[:399, ['Text','y']]


'''''''''''''''' Tokenizer '''''''''''''''''''''''

from transformers import BertModel, BertTokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

# from transformers import RobertaModel, RobertaTokenizer
# model_name = 'robert-base'
# tokenizer = RobertaTokenizer.from_pretrained(model_name)

# from transformers import LongformerModel, LongformerTokenizer
# model_name = "allenai/longformer-base-4096"
# tokenizer = LongformerTokenizer.from_pretrained(model_name)


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

num_classes=9
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-05


class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.Text
        self.targets = self.data.y
        self.max_len = max_len
    def __len__(self):
        return len(self.text)
    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(text,None,add_special_tokens=True,max_length=self.max_len,pad_to_max_length=True,return_token_type_ids=True)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return {'ids': torch.tensor(ids, dtype=torch.long), 'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float) }


training_set = MultiLabelDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = MultiLabelDataset(test_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0 }
test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 0  }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)




'''''''''''''''' BERT '''''''''''''''''''''''

class BertClass(torch.nn.Module):
    def __init__(self):
        super(BertClass, self).__init__()
        self.l1 = BertModel.from_pretrained(model_name)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, num_classes)
    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


model = BertClass()
model.to(device)


'''''''''''''''' RoBERTa '''''''''''''''''''''''

class RoBertaClass(torch.nn.Module):
    def __init__(self):
        super(RoBertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained(model_name)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, num_classes) 
    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = RoBertaClass()
model.to(device)


'''''''''''''''' Longformer '''''''''''''''''''''''

class LongformerClass(torch.nn.Module):
    def __init__(self):
        super(LongformerClass, self).__init__()
        self.l1 = LongformerModel.from_pretrained(model_name)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, num_classes)
    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    
 
model = LongformerClass()
model.to(device)
   
    


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
def train_model(epoch):
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)
        outputs = model(ids, mask, token_type_ids)
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _%1000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}') 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


for epoch in range(EPOCHS):
    train_model(epoch)

def validation(model, testing_loader):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

outputs, targets = validation(model, testing_loader)
outputs = np.array(outputs) >= 0.5



'''''''''''''''' Multi-Label Metrics '''''''''''''''''''''''

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, multilabel_confusion_matrix
accuracy = accuracy_score(targets, outputs)
f1_score_micro = f1_score(targets, outputs, average='micro')
f1_score_macro = f1_score(targets, outputs, average='macro')
f1_score_sample = f1_score(targets, outputs, average='samples')
recall_score_sample = recall_score(targets, outputs, average='samples')
precision_score_sample = precision_score(targets, outputs, average='samples')
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Micro) = {f1_score_micro}")
print(f"F1 Score (Macro) = {f1_score_macro}")
print(f"F1 Score (sample) = {f1_score_sample}")
print(f"Recall (sample) = {recall_score_sample}")
print(f"Precision (sample) = {precision_score_sample}")

conf_matrix = multilabel_confusion_matrix(targets, outputs)



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
fig.savefig("D:\\My\\Dataset Creation\\Phrases Topics\\Attention Visulaization\\MLP-Primate-ConfMatrix2.pdf")


