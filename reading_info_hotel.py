from bs4 import BeautifulSoup
import json
import os
from datetime import datetime
import re
def json_to_html(input_json_path, output_html_path):
    """
    Đọc nội dung từ một file JSON, lấy giá trị của "content" từ phần tử đầu tiên trong danh sách "results",
    và tạo một file HTML từ giá trị này.

    :param input_json_path: Đường dẫn đến file JSON đầu vào.
    :param output_html_path: Đường dẫn đến file HTML đầu ra.
    """
    try:
        # Đọc file JSON
        with open(input_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Kiểm tra cấu trúc của JSON để đảm bảo "results" là một danh sách và có ít nhất một phần tử
        if not isinstance(data.get('results'), list) or not data['results']:
            raise ValueError("'results' phải là một danh sách không rỗng.")

        # Lấy giá trị của "content" từ phần tử đầu tiên trong danh sách "results"
        first_result = data['results'][0]
        if 'content' not in first_result:
            raise KeyError("'content' không có trong phần tử đầu tiên của 'results'.")

        content = first_result['content']
        url = first_result['url']
        # Ghi giá trị này vào một file HTML
        with open(output_html_path, 'w', encoding='utf-8') as file:
            file.write(content)

        print(f"Đã tạo file HTML thành công: {output_html_path}")
        return url

    except KeyError as e:
        print(f"Lỗi: Không tìm thấy key {e} trong file JSON.")
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file {e}.")
    except json.JSONDecodeError as e:
        print(f"Lỗi: JSON không hợp lệ. Chi tiết lỗi: {e}")
    except ValueError as e:
        print(f"Lỗi: {e}")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")


def process_links(links):
    processed_links = []
    seen_links = set()

    for link in links:
        # Chỉ giữ lại các đường link cho dòng subtext "Hotel_Review", "Attraction_Review" hoặc "Restaurant_Review"
        if "Hotel_Review" in link or "Attraction_Review" in link or "Restaurant_Review" in link:
            # Xóa phần "#REVIEWS" khỏi đường link
            link = link.replace("#REVIEWS", "")

            # Nếu đường link bắt đầu bằng dấu "/", thêm string "https://www.tripadvisor.com" vào trước đường link đó
            if link.startswith("/"):
                link = "https://www.tripadvisor.com" + link

            # Kiểm tra và loại bỏ những đường link bị trùng lặp
            if link not in seen_links:
                processed_links.append(link)
                seen_links.add(link)

    return processed_links


def parse_hotel_page(html):
    soup = BeautifulSoup(html, 'html.parser')

    # Tên hotel
    hotel_name_tag = soup.find('h1', class_='biGQs _P rRtyp')
    hotel_name = hotel_name_tag.text.strip() if hotel_name_tag else 'N/A'

    # Thông tin chung
    general_info_div = soup.find('div', class_='fgplF')
    if general_info_div:
        general_info_spans = general_info_div.find_all('span', class_='biGQs _P pZUbB KxBGd')
        general_info = [span.text.strip() for span in general_info_spans]
    else:
        general_info = []

    # Thông tin sơ bộ về Hotel và các review
    summary_and_reviews = soup.find_all('div', class_=lambda x: x and '_T FKffI' in x)
    if summary_and_reviews:
        hotel_summary = summary_and_reviews[0].text.strip()
        reviews = []
        for div in summary_and_reviews[1:]:
            review_text = div.text.strip() if div.text.strip() else ''  # Thêm vào text trống nếu div không có text
            reviews.append(review_text)
    else:
        hotel_summary = 'N/A'
        reviews = []

    # Thống kê đánh giá số sao
    star_ratings_spans = soup.find_all('span', class_='QErCz')
    star_ratings = [int(re.sub(r'\D', '', span.text.strip())) for span in star_ratings_spans] if star_ratings_spans else [0, 0, 0, 0, 0]

    # Bậc location của địa điểm
    location_div = soup.find('div', class_='yrwyh f k O Cj Pf PN Ps PA')
    if location_div:
        location_spans = location_div.find_all('span')
        location = list({span.text.strip() for span in location_spans if span.text.strip()})
    else:
        location = []
    
    # Lấy text từ tất cả thẻ span có class 'iSNGb _R Me S4 H3 Cj' và tạo thành list 'reviews_time'
    reviews_time_spans = soup.find_all('span', class_='iSNGb _R Me S4 H3 Cj')
    reviews_time = [span.text.strip() for span in reviews_time_spans]

    # Lấy tất cả các đường dẫn "href" từ các thẻ <a>
    href_tags = soup.find_all('a', href=True)
    html_links = [tag['href'] for tag in href_tags]
    html_links = process_links(html_links)

    return {
        'hotel_name': hotel_name,
        'general_info': general_info,
        'hotel_summary': hotel_summary,
        'reviews': reviews,
        'star_ratings': star_ratings,
        'location': location,
        'reviews_time': reviews_time,
        'html_links': html_links
    }

def read_html_hotel(html_path = r"format.html"):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    input_json_path = os.path.join(current_directory, 'response.json')
    # input_json_path = r"E:\PBL_Crawler\response.json"
    output_html_path = os.path.join(current_directory, html_path)
    url = json_to_html(input_json_path, output_html_path)
    with open(html_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    result = parse_hotel_page(html_content)
    result['url'] = url
    save_hotel_data(result)
    # print(result.get('html_links', []))
    return result.get('html_links', [])

def save_hotel_data(result):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    storage_directory = os.path.join(current_directory, 'Scrape_Data', 'hotel')

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(storage_directory, exist_ok=True)

    # Sử dụng hotel_name để đặt tên file, nếu rỗng thì đặt là "Undefined-<current_time>"
    hotel_name = result.get('hotel_name', '').strip()
    if not hotel_name or hotel_name== 'N/A':
        hotel_name = f"Undefined-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Đảm bảo tên file an toàn
    safe_hotel_name = "".join(c for c in hotel_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    file_name = f"{safe_hotel_name}.json"
    file_path = os.path.join(storage_directory, file_name)

    # Đọc nội dung hiện có của file JSON, nếu file tồn tại
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Thêm kết quả mới vào dữ liệu hiện có
    data.append(result)

    # Lưu toàn bộ dữ liệu vào file JSON
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"Đã lưu dữ liệu của khách sạn vào file: {file_path}")

# read_html_hotel()