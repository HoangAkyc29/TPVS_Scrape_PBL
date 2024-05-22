from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

def Sentiment_Analysis(sentence):
    # Lấy đường dẫn tới thư mục chứa model
    current_directory = os.path.dirname(os.path.abspath(__file__))  # Lấy thư mục chứa chương trình đang chạy
    model_directory = os.path.join(current_directory, "BERT_Sentiment")  # Kết hợp với tên thư mục chứa model
    model_subdirectories = os.listdir(model_directory)  # Lấy danh sách tất cả các thư mục con trong thư mục chứa model

    # Lấy đường dẫn đến thư mục chứa model (được giả định là thư mục con đầu tiên trong 'BERT_Sentiment')
    if model_subdirectories:
        model_subdirectory = model_subdirectories[0]  # Lấy tên thư mục con đầu tiên
        model_path = os.path.join(model_directory, model_subdirectory)  # Kết hợp với đường dẫn gốc
    else:
        print("Không tìm thấy thư mục chứa model.")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    input = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
    output = model(**input)
    prediction = torch.nn.functional.softmax(output.logits, dim=-1)
    
    score = prediction[0][0].item() * (-1) + prediction[0][1].item() * 1
    return score

def analyze_sentences(data_dict):
    scores = {}
    for category, sentences_list in data_dict.items():
        category_scores = []
        for sentence in sentences_list:
            score = Sentiment_Analysis(sentence)
            category_scores.append(score)
        scores[category] = category_scores
    return scores

