import re
import transformers
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from torchcrf import CRF
import joblib
import preprocess

BASE_MODEL_PATH = "./KeyPhraseModel/bert-base-uncased"
MODEL_PATH = "./KeyPhraseModel/BERT-BiLSTM-CRF"
MAX_LEN = 256
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True
)

with open('vietnamese-stopwords.txt', 'r', encoding='utf-8') as file:
    stop_words = file.read().splitlines()

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


class EntityModel(nn.Module):
    def __init__(self, num_tag):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.bert = transformers.BertModel.from_pretrained(BASE_MODEL_PATH,return_dict=False)
        self.bilstm= nn.LSTM(768, 1024 // 2, num_layers=1, bidirectional=True, batch_first=True)

        self.dropout_tag = nn.Dropout(0.3)

        self.hidden2tag_tag = nn.Linear(1024, self.num_tag)

        self.crf_tag = CRF(self.num_tag, batch_first=True)
        
    # def forward(self, ids, mask, token_type_ids, target_tag):
    #     x, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
    #     h, _ = self.bilstm(x)

    #     o_tag = self.dropout_tag(h)
    #     tag = self.hidden2tag_tag(o_tag)
    #     mask = torch.where(mask==1, True, False)

    #     loss_tag = - self.crf_tag(tag, target_tag, mask=mask, reduction='token_mean')
    #     loss=loss_tag

    #     return loss
    def encode(self, ids, mask, token_type_ids, target_tag):
        # Bert - BiLSTM
        x, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        h, _ = self.bilstm(x)

        # drop out
        o_tag = self.dropout_tag(h)
        # o_pos = self.dropout_pos(h)

        # Hidden2Tag (Linear)
        tag = self.hidden2tag_tag(o_tag)
        mask = torch.where(mask==1, True, False)
        tag = self.crf_tag.decode(tag, mask=mask)

        return tag
    
    
class EntityDataset:
    def __init__(self, texts, tags,enc_tag):
        # texts: [["hi", ",", "my", "name", "is", "abhishek"], ["hello".....]]
        # pos/tags: [[1 2 3 4 1 5], [....].....]]
        self.texts = texts
        # self.pos = pos
        self.tags = tags
        self.enc_tag=enc_tag

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        # pos = self.pos[item]
        tags = self.tags[item]

        ids = []
        # target_pos = []
        target_tag =[]

        for i, s in enumerate(text):
            inputs = TOKENIZER.encode(
                str(s),
                add_special_tokens=False
            )
            # abhishek: ab ##hi ##sh ##ek
            input_len = len(inputs)
            ids.extend(inputs)
            # target_pos.extend([pos[i]] * input_len)
            target_tag.extend([tags[i]] * input_len)

        ids = ids[:MAX_LEN - 2]
        # target_pos = target_pos[:MAX_LEN - 2]
        target_tag = target_tag[:MAX_LEN - 2]

        ids = [102] + ids + [103]
        # target_pos = [0] + target_pos + [0]
        o_tag=self.enc_tag.transform(["O"])[0]
        target_tag = [o_tag] + target_tag + [o_tag]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = MAX_LEN - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        # target_pos = target_pos + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            # "target_pos": torch.tensor(target_pos, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
            # "words":torch.tensor(words,dtype=torch.int)
        }
        
def predict_sentence(model, sentence, enc_tag):
    sentence = sentence.split()
    test_dataset = EntityDataset(
        texts=[sentence],
        # pos=[[0] * len(sentence)],
        tags=[[0] * len(sentence)],
        enc_tag=enc_tag
    )

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)

        tag = model.encode(**data)
        tag = enc_tag.inverse_transform(tag[0])
        # pos = enc_pos.inverse_transform(pos[0])

    return tag

import numpy as np
def reverse_tokenize(ids, tags):
    tokens = []
    tags_list = []
    for token_id, tag in zip(ids, tags):
        token = TOKENIZER.decode(token_id)
        token = token.replace(' ', '')
        token_array = np.array(list(token))
        token_string = ''.join(token_array)
        if token_string.startswith('##'):
            token_string = token_string.replace('##', '')
            if tokens:
                tokens[-1] += token_string
                # Nếu từ bắt đầu bằng '##', ta vẫn giữ nguyên tag của từ trước đó
                tags_list[-1] = tag
        else:
            tokens.append(token_string)
            tags_list.append(tag)
    # return list(zip(tokens, tags_list))
    return list(tokens)
    


meta_data = joblib.load("./KeyPhraseModel/BERT-BiLSTM-CRF/meta.bin")
enc_tag = meta_data["enc_tag"]
keyphrase_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
model_bert_keyphrase = EntityModel(3)
model_bert_keyphrase.load_state_dict(torch.load('./KeyPhraseModel/BERT-BiLSTM-CRF/trained_model_3.bin', map_location=torch.device('cpu')))
model_bert_keyphrase.eval()
device = torch.device("cpu")
model_bert_keyphrase.to(device)

def keyphraseExtraction(text):
    text = preprocess.remove_stopwords(text)
    text = preprocess.remove_punctuation(text)
    tokenized_sentence = TOKENIZER.encode(text)
    tags = predict_sentence(model_bert_keyphrase, text, enc_tag)

    reversed_tokens = reverse_tokenize(tokenized_sentence, tags)
    return reversed_tokens

