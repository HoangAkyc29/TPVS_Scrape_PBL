import os
import json

def extract_reviews_to_txt_unique(file_path, output_file):
    unique_reviews = set()  # Tạo một tập hợp để lưu trữ các đánh giá duy nhất

    with open(output_file, 'w', encoding='utf-8') as txt_file:
        # Duyệt qua từng file trong thư mục
        for root, dirs, files in os.walk(file_path):
            for file_name in files:
                if file_name.endswith('.json'):
                    file_path = os.path.join(root, file_name)
                    # Đọc file JSON
                    with open(file_path, 'r', encoding='utf-8') as json_file:
                        data = json.load(json_file)
                        # Duyệt qua từng dictionary trong list
                        for item in data:
                            # Kiểm tra xem có trường 'reviews' trong từng dictionary không
                            if 'reviews' in item:
                                reviews = item['reviews']
                                # Duyệt qua từng review trong trường 'reviews'
                                for review in reviews:
                                    # Kiểm tra xem review đã tồn tại trong tập hợp chưa
                                    if review not in unique_reviews:
                                        # Ghi review vào file txt và thêm vào tập hợp
                                        txt_file.write(review + '\n')
                                        unique_reviews.add(review)

# Sử dụng hàm
# extract_reviews_to_txt_unique(r"E:\PBL_Crawler\Scrape_Data\attraction", 'reviews.txt')

import csv

def handle_txt_to_csv(file_path):
  """
  Hàm đọc file txt, xử lý và tạo file csv.

  Args:
    file_path: Đường dẫn đến file txt.

  Returns:
    None.
  """
  # Mở file txt và đọc từng dòng
  with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

  # Khởi tạo list lưu trữ các dòng sau khi xử lý
  processed_lines = []

  # Duyệt qua từng dòng trong file
  for line in lines:
    # Loại bỏ khoảng trắng đầu và cuối dòng
    stripped_line = line.strip()

    # Loại bỏ dòng nếu có ít hơn 40 ký tự
    if len(stripped_line) < 40:
      continue

    # Loại bỏ text "Read more" hoặc "Đọc thêm" (nếu có)
    processed_line = stripped_line.replace('Read more', '').replace('Đọc thêm', '').replace('\n', ' ')

    # Thêm dòng đã xử lý vào list
    processed_lines.append(processed_line)

  # Tạo file csv và ghi dữ liệu
  with open('processed_data.csv', 'w', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['review', 'label', 'approve'])  # Ghi tiêu đề cột
    for line in processed_lines:
        writer.writerow([line, '', ''])  # Ghi dữ liệu, 2 cột label và approve để rỗng

# Ví dụ sử dụng
file_path = r'E:\PBL_Crawler\reviews.txt'  # Thay đổi đường dẫn file txt cho phù hợp
handle_txt_to_csv(file_path)

print('Đã tạo file csv thành công!')