import requests
import json

TRANSLATION_ERROR = "Lỗi dịch"
TRANSLATION_PROVIDER = "Google Translate"
translate_from = "vi"
translate_to = "en"

def translate_text(text, translate_from = "auto", translate_to = "vi", auth_key=None):
    if auth_key:
        url = f"https://translation.googleapis.com/language/translate/v2?format=text&target={translate_to}&key={auth_key}"
        data = {"q": [text], "source": translate_from}
        response = requests.post(url, json=data)
        if response.status_code == 200:
            translation = json.loads(response.text)["data"]["translations"][0]["translatedText"]
            return True, translation
        else:
            return False, f"{TRANSLATION_ERROR}: {response.text}"
    else:
        url = f"https://translate.google.com/m?sl={translate_from}&tl={translate_to}&q={text}"
        response = requests.get(url)
        if response.status_code == 200:
            start = response.text.find("result-container\">") + 18
            end = response.text.find('<', start)
            translation = response.text[start:end]
            return True, translation
        else:
            return False, f"{TRANSLATION_ERROR}: {response.text}"