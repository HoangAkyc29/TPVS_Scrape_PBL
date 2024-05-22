import re
import transformers
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from TorchCRF import CRF
import joblib
import preprocess
import os
from score_extract import get_sentence_score_and_info, combine_dicts
#pip scikit-learn

def base_model_path():
  current_directory = os.path.dirname(os.path.abspath(__file__))  # Lấy thư mục chứa chương trình đang chạy
  model_directory = os.path.join(current_directory, "KeyPhraseModel")
  model_path = os.path.join(model_directory, "bert-base-uncased")
  return model_path

def bin_path(bin_name = 'trained_model_3.bin'):
  current_directory = os.path.dirname(os.path.abspath(__file__))  # Lấy thư mục chứa chương trình đang chạy
  model_directory = os.path.join(current_directory, "KeyPhraseModel")
  model_path = os.path.join(model_directory, "BERT-BiLSTM-CRF")
  train_model_path = os.path.join(model_path, bin_name)
  return train_model_path

MAX_LEN = 256

def tokenizer():
    return transformers.BertTokenizer.from_pretrained(
    base_model_path(),
    do_lower_case=True
)

def vie_stopwords_path():
  current_directory = os.path.dirname(os.path.abspath(__file__))  # Lấy thư mục chứa chương trình đang chạy
  return os.path.join(current_directory, "vietnamese-stopwords.txt")

with open(vie_stopwords_path(), 'r', encoding='utf-8') as file:
    stop_words = file.read().splitlines()

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


class EntityModel(nn.Module):
    def __init__(self, num_tag):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.bert = transformers.BertModel.from_pretrained(base_model_path(),return_dict=False)
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
            inputs = tokenizer().encode(
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
        token = tokenizer().decode(token_id)
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
    return list(tokens), list(tags_list)
    


meta_data = joblib.load(bin_path("meta.bin"))
enc_tag = meta_data["enc_tag"]
keyphrase_tokenizer = AutoTokenizer.from_pretrained(base_model_path(), use_fast=False)
model_bert_keyphrase = EntityModel(3)
model_bert_keyphrase.load_state_dict(torch.load(bin_path(), map_location=torch.device('cpu')))
model_bert_keyphrase.eval()
device = torch.device("cpu")
model_bert_keyphrase.to(device)

def remove_punctuation(text):
  """Removes punctuation from a string.

  Args:
      text: The string to remove punctuation from.

  Returns:
      A new string with all punctuation characters removed.
  """

  # Define punctuation characters
  punctuation = "!\"#$%&()*+,./:;<=>?@[\\]^_`{|}~"

  # Use translation table to remove punctuation
  no_punct_text = text.translate(str.maketrans('', '', punctuation))

  return no_punct_text

def entitesExtraction(text):
    # text = preprocess.remove_stopwords(text)
    # text = remove_stopwords(text)
    text = remove_punctuation(text)
    tokenized_sentence = tokenizer().encode(text)
    tags = predict_sentence(model_bert_keyphrase, text, enc_tag)
    reversed_tokens, reversed_tags = reverse_tokenize(tokenized_sentence, tags)
    return text, reversed_tokens, reversed_tags

def TextEntities_Score(text):
    sentences = text.split('.')
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    dicts_sentiment = {}
    dicts_adj_feature = {}
    list_noun_feature = []
    list_proper_noun_feature = []
    for sentence in sentences:
        sentence_text, reversed_tokens, reversed_tags = entitesExtraction(sentence)
        dict_sentiment, noun_feature, dict_adj_feature, proper_noun_feature = get_sentence_score_and_info(sentence_text, reversed_tokens, reversed_tags)
        dicts_sentiment = combine_dicts(dicts_sentiment, dict_sentiment)
        dicts_adj_feature = combine_dicts(dicts_adj_feature, dict_adj_feature)
        list_noun_feature.extend(noun_feature)
        list_proper_noun_feature.extend(proper_noun_feature)
    return dicts_sentiment, dicts_adj_feature, list_noun_feature, list_proper_noun_feature


# text = "Cho tôi 1 địa điểm du lịch với công viên nước, các con vật, thiên nhiên mát mẻ. Đặc biệt là chó, con trai tôi rất thích chó. Và ở gần đó tôi muốn có cả những khách sạn chuẩn 5 sao, tiện nghi, sang trọng để gia đình tôi có thể có trải nghiệm tốt nhất trong thời gian tại Việt Nam."
# print(entitesExtraction(text))
# # sentence_text, reversed_tokens, reversed_tags = entitesExtraction("Cho tôi 1 địa điểm du lịch hang động, với các đồi núi lớn")
# # dict_sentiment, noun_feature, dict_adj_feature, proper_noun_feature = get_sentence_score_and_info(sentence_text, reversed_tokens, reversed_tags)
# # print(dict_sentiment)

# dicts_sentiment, dicts_adj_feature, list_noun_feature, list_proper_noun_feature = TextEntities_Score(text)
# print(dicts_sentiment)
# print(dicts_adj_feature)
# print(list_proper_noun_feature)