import pandas as pd
import string
from nltk.corpus import stopwords

class SCOTUS:
    def __init__(self, usecols=['author_name', 'category', 'per_curiam', 'case_name', 'year_filed', 'text']):
        self.since_1970 = pd.read_csv('scotus/opinions_since_1970.csv', usecols=usecols)
        self.all_opinions = pd.read_csv('scotus/all_opinions.csv', usecols=usecols)


class OpinionPreprocessing:
    def __init__(self, text):
        self.text = text
        
        # Clean and tokenize
        # obvious-ly => obvious ly (lemmatization)
        words = ''.join((filter(lambda x: x in string.printable, text))).replace('\n', ' ')
        # print(type(words), len(words))
        words = words.replace('–', '')
        # print(type(words), len(words))
        table = str.maketrans('', '', string.punctuation+'’‘'+'“”'+'–'+string.digits+'­')
        words = [w.translate(table).lower() for w in words.split()]
        # print(type(words), len(words))
        stop_words = stopwords.words('english')
        words = list(filter(lambda w: w not in stop_words, words))
        # print(type(words), len(words))
        words = list(filter(lambda w: w.isalpha(), words))
        
        self.words = words
        pass
