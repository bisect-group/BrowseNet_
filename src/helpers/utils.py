import re
from unidecode import unidecode

def preprocess_entity(phrase):
    return re.sub('[^A-Za-z0-9 ]', ' ', phrase.lower()).strip()

def clean_text(input_text):
    """
    input_text: str
    returns: cleaned text - (no punctuation, strip leading/trailing spaces, articles, and convert to lowercase)
    """
    cleaned_text = unidecode(input_text)
    cleaned_text = cleaned_text.replace('-', ' ')
    cleaned_text = cleaned_text.replace('/', ' ')
    cleaned_text = re.sub(r"\b(a|an|the)\b", " ", cleaned_text) 
    cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', cleaned_text)  
    cleaned_text = cleaned_text.lower()  
    cleaned_text = ' '.join([word.strip() for word in cleaned_text.split()]) 
    return cleaned_text