import py_vncorenlp
import os
from sentiment_analysis import analyze_sentences
# vncorenlp_path = "./VN_core_NLP"

# if not os.path.exists(vncorenlp_path):
#     os.makedirs(vncorenlp_path)
# try:
#     py_vncorenlp.download_model(save_dir=vncorenlp_path)
# except Exception as e:
#    print(e)

def load_vncorenlp_model():
  current_directory = os.path.dirname(os.path.abspath(__file__))  # Lấy thư mục chứa chương trình đang chạy
  model_directory = os.path.join(current_directory, "VN_core_NLP")
  return py_vncorenlp.VnCoreNLP(save_dir=model_directory)

model = load_vncorenlp_model()

def get_features(sentence, tokens, tags):
    # Loại bỏ token [CLS] và [SEP] trong tokens
    tokens = tokens[1:-1]

    # Tạo danh sách các từ trong câu
    words = sentence.split()

    # Tạo danh sách từ với lowercase
    words_lower = [word.lower() for word in words]

    # Thay thế các tokens trong câu thành từ tương ứng lowercase
    vie_tokens = [word.lower() for word in words_lower if word != '[CLS]' and word != '[SEP]']

    # Loại bỏ giá trị đầu và cuối của tags
    tags = tags[1:-1]

    # Ánh xạ vie_tokens với tags
    mapped_tags = []
    i = 0
    for token in tokens:
        if token.startswith("##"):
            continue
        mapped_tags.append(tags[i])
        i += 1

    # Tạo features
    features = []
    current_feature = []
    for token, tag in zip(vie_tokens, mapped_tags):
        if tag == 'B-P':
            if current_feature:
                features.append(" ".join(current_feature))
                current_feature = []
            current_feature.append(token)
        elif tag == 'I-P' and current_feature:
            current_feature.append(token)
        else:
            if current_feature:
                features.append(" ".join(current_feature))
                current_feature = []

    if current_feature:
        features.append(" ".join(current_feature))

    return features

def classify_word(text):
  result = next(iter(model.annotate_text(text).values()))
  textlabel = 1
  for i in range(10):
    try:
      word_instance = result[i]
      if word_instance == None:
        break
    except Exception as e:
      break
    word_type = word_instance['posTag']
    word_label = word_instance['nerLabel']
    if word_label == "B-LOC" or word_label == "B-PER":
      textlabel = 3
      break
    elif word_type == "A" or word_type == "V":
      textlabel = 2
    else:
      textlabel = 1
      break
  return textlabel

def classify_words_list(word_list):
    result_1 = []
    result_2 = []
    result_3 = []

    for word in word_list:
        classification = classify_word(word)
        if classification == 1:
            result_1.append(word)
        elif classification == 2:
            result_2.append(word)
        elif classification == 3:
            result_3.append(word)

    return result_1, result_2, result_3

def check_substring(A, B):
    for string in B:
        if A in string:
            return True
    return False

def get_adj_words(text, list_noun_feature):
  result = next(iter(model.annotate_text(text).values()))
  list_adj = []
  list_adj_index = []
  for i in range(100):
    try:
      word_instance = result[i]
      if word_instance == None:
        break
    except Exception as e:
      break
    word_type = word_instance['posTag']
    word_label = word_instance['nerLabel']
    word_form = word_instance['wordForm'].replace("_", " ")
    if word_label == "B-LOC" or word_label == "B-PER":
      continue
      break
    elif word_type == "A" and check_substring(word_form, list_noun_feature) == False:
      list_adj.append(word_form)
      list_adj_index.append(i)
  return list_adj, list_adj_index

def sort_words(sentence, list_noun, list_adj):
    list_features = list_adj + list_noun
    sort_features = []
    sort_features_with_index = []

    words = sentence.split()

    for index, word in enumerate(words):
        for feature_word in list_features:
            feature_word_list = feature_word.split()
            if feature_word_list[0] != word:
                continue
            if words[index:index+len(feature_word_list)] == feature_word_list:
                sort_features_with_index.append([feature_word, index])
                sort_features.append(feature_word)
                break

    return sort_features_with_index, sort_features

def find_nth_occurrence(lst, item, n):
    indices = [index for index, value in enumerate(lst) if value == item]
    if len(indices) >= n:
        return indices[n - 1]
    else:
        return -1

def mota_danh_tu(list_noun, list_adj, sort_word):
  """
  Hàm tạo mô tả cho danh từ bằng cách kết hợp danh từ với các tính từ phù hợp.

  Args:
    list_noun (list): Danh sách danh từ.
    list_adj (list): Danh sách tính từ.
    sort_word (list): Danh sách kết hợp danh từ và tính từ theo thứ tự.

  Returns:
    dict: Từ điển kết hợp danh từ với mô tả (tính từ).
  """

  dict_mota = {}
  seen_nouns = set()  # Tập hợp lưu trữ các danh từ đã được xử lý

  for i, noun in enumerate(list_noun):

    # Tìm vị trí của danh từ hiện tại trong danh sách sort_word
    previous_count = list_noun[:i].count(noun) + 1
    noun_index = find_nth_occurrence(sort_word, noun, previous_count)


    # Khởi tạo danh sách tính từ cho danh từ hiện tại
    adj_list = []

    # Duyệt qua các từ tiếp theo trong sort_word
    for j in range(noun_index + 1, len(sort_word)):
      word = sort_word[j]

      # Kiểm tra xem từ tiếp theo có phải là tính từ hay không
      if word in list_adj:
        adj_list.append(word)
      elif len(adj_list) > 0:
        # Nếu không phải tính từ, dừng lặp và lưu danh sách tính từ đã thu thập
        break
      else:
        continue

    # # Thêm mô tả (danh sách tính từ) cho danh từ hiện tại vào dict_mota
    # if noun in seen_nouns:
    #   dict_mota[noun].append(adj_list)
    # else:
    seen_nouns.add(noun)  # Thêm noun vào tập hợp đã xử lý
    if noun in dict_mota:
      dict_mota[noun].extend(adj_list)
    else:
      dict_mota[noun] = adj_list


  return dict_mota


def process_sentence(entity, adj, sentence, list_noun, list_adj):
    # Xóa các từ trong list_word mà có chứa trong entity_with_adj
    list_features = list_adj + list_noun
    list_features = [word for word in list_features if word not in entity]
    list_features = [word for word in list_features if word not in adj]

    words = sentence.split()
    entity_split = entity.split()
    adj_split = adj.split()
    flag1 = 0
    flag2 = 0
    sub_sentence = []
    for index, word in enumerate(words):
        if entity_split[0] != word and flag1 == 0:
              continue
        if words[index:index+len(entity_split)] == entity_split:
          sub_sentence = []
          flag1 = 1
        if words[index-len(adj_split):index] == adj_split:
          flag1 = 0
          break
        for feature_word in list_features:
            feature_word_split = feature_word.split()
            if words[index:index+len(feature_word_split)] == feature_word_split:
              flag2 = len(feature_word_split)
              break
        if flag1 == 1 and flag2 == 0:
          sub_sentence.append(word)
        if flag2 > 0:
          flag2 = flag2 - 1
    sub_sentence_string = ' '.join(sub_sentence)
    return sub_sentence_string

def process_data(dict_data, sentence, list_noun, list_adj):
  """
  Hàm xử lý dữ liệu dictionary bằng cách sử dụng hàm `process_sentence`.

  Args:
    dict_data: Dictionary chứa dữ liệu cần xử lý.
    process_sentence: Hàm nhận hai giá trị `entity` và `adj` và trả về một câu.

  Returns:
    Một dictionary mới với các giá trị được thay thế bằng các câu được tạo ra bởi hàm `process_sentence`.
  """
  new_dict_data = {}
  for key, values in dict_data.items():
    new_sentences = []
    for adj in values:
      new_sentences.append(process_sentence(key, adj, sentence, list_noun, list_adj))
    new_dict_data[key] = new_sentences
  return new_dict_data

# sentence = "Cho tôi 1 địa điểm du lịch hang động với các đồi núi lớn"
# tokens = ['[CLS]', 'cho', 'toi', '1', 'đia', 'điem', 'du', 'lich', 'hang', 'đong', 'voi', 'cac', 'đoi', 'nui', 'lon', '[SEP]'] 
# tags =  ['O', 'O', 'O', 'O', 'O', 'B-P', 'I-P', 'I-P', 'I-P', 'I-P', 'O', 'O', 'B-P', 'I-P', 'O', 'O']

def get_entity_data(sentence, tokens, tags):
  features = get_features(sentence, tokens, tags)
  list_noun_feature = classify_words_list(features)[0]
  proper_noun_feature = classify_words_list(features)[2]
  list_adj_feature = get_adj_words(sentence,list_noun_feature)[0]
  if len(list_noun_feature) == 1:
    # Chỉ có một tính năng danh từ
    noun_feature = list_noun_feature[0]  # Lấy tính năng danh từ duy nhất
    dict_mota = {noun_feature: list_adj_feature}
    new_dict_data = {noun_feature: [sentence]}
  elif len(list_noun_feature) == 0:
    dict_mota = {}
    new_dict_data = {}
  else:
    sort_features_with_index, sort_features = sort_words(sentence, list_noun_feature, list_adj_feature)
    dict_mota = mota_danh_tu(list_noun_feature, list_adj_feature, sort_features)
    new_dict_data = process_data(dict_mota, sentence, list_noun_feature, list_adj_feature)
  # print(new_dict_data)
  return new_dict_data, list_noun_feature, dict_mota, proper_noun_feature

def get_sentence_score_and_info(sentence, tokens, tags):
  entity_dict, list_noun_feature, dict_adj_feature, proper_noun_feature = get_entity_data(sentence, tokens, tags)
  dict_sentiment = analyze_sentences(entity_dict)
  return dict_sentiment, list_noun_feature, dict_adj_feature, proper_noun_feature

def combine_dicts(*dicts):
    combined_dict = {}
    for d in dicts:
        for key, value in d.items():
            if key in combined_dict:
                combined_dict[key].extend(value)
            else:
                combined_dict[key] = value
    return combined_dict

