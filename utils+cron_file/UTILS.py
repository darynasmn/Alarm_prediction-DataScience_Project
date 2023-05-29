def remove_names_and_dates(page_html_text,BeautifulSoup):
    from bs4 import BeautifulSoup
    parsed_html = BeautifulSoup(page_html_text)
    p_lines = parsed_html.findAll('p')
    
    min_sentences_word_count = 20
    p_index=0
    
    for p_line in p_lines:
        strong_lines = p_line.findAll('strong')
        if not strong_lines:
            p_index += 1
            continue
            
        for s in strong_lines:
            if len(s.text.split(' '))>= min_sentences_word_count:
                break
        else:
            p_index +=1
            continue
        break
    for i in range(0, p_index):
        page_html_text = page_html_text.replace(str(p_lines[i]), '')
    return page_html_text
def remove_one_letter_word(data):
    from nltk.tokenize import word_tokenize
    words = word_tokenize(str(data))
    
    new_text = ""
    for w in words:
        if len(w) > 1:
            new_text = new_text + " " + w
    
    return new_text

def convert_lower_case(data):
    import numpy as np
    return np.char.lower(data)

def remove_stop_words(data):
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    stop_stop_words = {"no","not"}
    stop_words = stop_words - stop_stop_words
    
    from nltk.tokenize import word_tokenize
    words = word_tokenize(str(data))
    
    new_text=''
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def remove_punctuation(data):
    import numpy as np
    symbols = "!\"#$%&()*+—-./:;<=>?@[\]^_`{|}~\n"
    
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data,'  ', ' ')
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    import numpy as np
    return np.char.replace(data, "'", "")

def stemming(data):
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def lemmatizing(data):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + lemmatizer.lemmatize(w)
    return new_text

def conver_numbers(data):
    import numpy as np
    from nltk.tokenize import word_tokenize
    from num2words import num2words
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        if w.isdigit():
            if int(w)<1000000000000:
                w = num2words(w)
            else:
                w=''
        new_text = new_text + ' ' + w
    new_text = np.char.replace(new_text,'-',' ')
    
    return new_text

def remove_url_string(data):
    import re
    from nltk.tokenize import word_tokenize
    words = word_tokenize(str(data))
    
    new_text = ''
    for w in words:
        w = re.sub(r'^https?:\/\/.*[\r\n]*', '', str(w), flags = re.MULTILINE)
        w = re.sub(r'^http?:\/\/.*[\r\n]*', '', str(w), flags = re.MULTILINE)
        
        new_text = new_text + ' ' + w
        
    return new_text

def remove_names_and_dates(page_html_text,BeautifulSoup):
    from bs4 import BeautifulSoup
    parsed_html = BeautifulSoup(page_html_text)
    p_lines = parsed_html.findAll('p')
    
    min_sentences_word_count = 20
    p_index=0
    
    for p_line in p_lines:
        strong_lines = p_line.findAll('strong')
        if not strong_lines:
            p_index += 1
            continue
            
        for s in strong_lines:
            if len(s.text.split(' '))>= min_sentences_word_count:
                break
        else:
            p_index +=1
            continue
        break
    for i in range(0, p_index):
        page_html_text = page_html_text.replace(str(p_lines[i]), '')
    return page_html_text