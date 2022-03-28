import pandas as pd
import string
from nltk.corpus import stopwords
from collections import Counter
from itertools import combinations
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, norm
import numpy as np
from math import log
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

class SCOTUS:
    def __init__(self, usecols=['author_name', 'category', 'per_curiam', 'case_name', 'federal_cite_one', 'year_filed', 'scdb_decision_direction', 'text']):
        self.since_1970 = pd.read_csv('scotus/opinions_since_1970.csv', usecols=usecols)
        self.all_opinions = pd.read_csv('scotus/all_opinions.csv', usecols=usecols)


class Opinion:
    def __init__(self, text=''):
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

# https://osf.io/2tvh5/
gender_words = {
            "female": "she her mommy mom girl mother lady sister mama momma sis grandma herself".split(),
            "male": "he his daddy dad boy father guy brother dada papa bro grandpa himself".split(), 
            "femalenew": "hers women woman daughter wife gal female girlfriend queen grandmother aunt granddaughter niece gentlewoman sorority bachelorette princess dame gentlewomen stepmother ms madam bride stepdaughter maiden godmother grandma missus heroine motherhood sororities ma".split(),
            "malenew": "him man men son husband guy male boyfriend king grandfather uncle grandson nephew lad gentleman fraternity bachelor prince dude gentlemen stepfather sir bloke groom stepson suitor godfather fella hero fatherhood fraternities pa".split(),
            "femalemax": "her hers she women woman herself daughter mother wife gal girl sister female mom girlfriend queen grandmother aunt granddaughter niece lady gentlewoman sorority bachelorette princess dame gentlewomen stepmother mommy ms madam bride stepdaughter maiden godmother grandma missus heroine motherhood sororities mama ma".split(),
            "malemax": "he his him man men himself son father husband guy boy brother male dad boyfriend king grandfather uncle grandson nephew lad gentleman fraternity bachelor prince dude gentlemen stepfather daddy sir bloke groom stepson suitor godfather grandpa fella hero fatherhood fraternities papa pa".split()
}

all_gender_words = list(set([word for gender in gender_words.values() for word in gender]))

# Removes punctuation, numbers, non-gendered stopwords from string and returns tokenized array
# Str -> [Str]
def clean_text(text, remove_stopwords=True):
    words = ''.join((filter(lambda x: x in string.printable, text))).replace('\n', ' ')
    # print(type(words), len(words))
    words = words.replace('–', '')
    # print(type(words), len(words))
    table = str.maketrans('', '', string.punctuation+'’‘'+'“”'+'–'+string.digits+'­')
    words = [w.translate(table).lower() for w in words.split()]
    # remove stopwords but keep gender terms
    if remove_stopwords:
        stop_words = [stopword for stopword in stopwords.words('english') if stopword not in all_gender_words]
        words = list(filter(lambda w: w not in stop_words, words))
    words = list(filter(lambda w: w.isalpha(), words))
    return words

class ProjectUtils:
    def __init__(self):
        pass
    
    def summarize_opinions_metadata(df_opinions):
        print("Number of opinions: {}".format(len(df_opinions.index)))
        case_name_counts = df_opinions.value_counts('case_name')
        print("Number of unique cases: {}".format(len(list(set(df_opinions['case_name'])))))
        print(case_name_counts)
        print('----')
        category_value_counts = df_opinions.value_counts('category')
        print(category_value_counts)
        print('----')
        author_name_counts = df_opinions.value_counts('author_name')
        print("Number of unique justices: {}".format(len(author_name_counts)))
        print(author_name_counts)
        print('----')
        # Values:
        # 1	conservative
        # 2	liberal
        # 3	unspecifiable
        print(df_opinions.value_counts('scdb_decision_direction'))
        pass
    
    # Removes punctuation, numbers, non-gendered stopwords from string and returns tokenized array
    # Str -> [Str]
    def clean_text(text):
        return clean_text(text)
    
    # Uni- Bi-gram counters
    def count_grams(texts_tokenized, cx = Counter(), cxy = Counter()):
        for x in texts_tokenized:
            cx[x] += 1
        for x, y in map(sorted, combinations(texts_tokenized, 2)):
            cxy[(x, y)] += 1
        return cx, cxy

    def model_matrices(unigram_counter, bigram_counter):
        # index to/from token maps
        x2i, i2x = {}, {}
        for i, x in enumerate(unigram_counter.keys()):
            x2i[x] = i
            i2x[i] = x
        
        sx = sum(unigram_counter.values())
        sxy = sum(bigram_counter.values())

        ppmi_counter = Counter()
        cooccurrence_counter = Counter()

        rows, cols = [], []
        ppmi_data = []
        cooccurence_data = []
        for (x, y), n in bigram_counter.items():
            rows.append(x2i[x])
            cols.append(x2i[y])

            ppmi_data.append(max(0, log((n / sxy) / (unigram_counter[x] / sx) / (unigram_counter[y] / sx))))
            ppmi_counter[(x, y)] = ppmi_data[-1]

            cooccurence_data.append(bigram_counter[(x,y)])
            cooccurrence_counter[(x,y)] = cooccurence_data[-1]
        ppmi = csc_matrix((ppmi_data, (rows, cols)))
        cooccurence = csc_matrix((cooccurence_data, (rows, cols)))
        # return ppmi, ppmi_counter, cooccurence, cooccurrence_counter
        return ppmi, cooccurence, x2i, i2x
        
    def model_svd(matrix, k=10, normalize=True):
        def normalize_matrix(matrix):
            norms = np.sqrt(np.sum(np.square(matrix), axis=1, keepdims=True))
            matrix /= np.maximum(norms, 1e-7)
            return matrix
        matrix = matrix.asfptype()
        print(type(matrix))
        U, S, VT = svds(matrix, k=k)
        print('U shape', U.shape)
        print('S shape', S.shape)
        print('VT shape', VT.shape)
        if (normalize):
            U = normalize_matrix(U)
        return {"U": U, "S": S, "VT": VT}  
    
    def wv_cosine_similarity(x_vec, y_vec):
        return cosine_similarity(x_vec, y_vec)#[0][0]
    
    #     VISUALIZATIONS
    #     count : 'type_count' | 'token_count'
    def plot_corpora_scatter(df_opinions, type_token='type_count'):
        # count_types = lambda row : len(set(clean_text(row['text'])))
        # count_tokens = lambda row : len(clean_text(row['text']))
        df_opinions = df_opinions.assign(
            type_count= df_opinions['text'].apply(lambda text: len(set(clean_text(text, remove_stopwords=False)))),
            token_count= df_opinions['text'].apply(lambda text: len(clean_text(text, remove_stopwords=False)))
        )
        print(df_opinions.head())
        
        majority_liberal = df_opinions[
            (df_opinions['category']=='majority') &
            (df_opinions['scdb_decision_direction']==2.0)
        ]
        majority_conservative = df_opinions[
            (df_opinions['category']=='majority') &
            (df_opinions['scdb_decision_direction']==1.0)
        ]
        dissenting_liberal = df_opinions[
            (df_opinions['category']=='dissenting') &
            (df_opinions['scdb_decision_direction']==2.0)
        ]
        dissenting_conservative = df_opinions[
            (df_opinions['category']=='dissenting') &
            (df_opinions['scdb_decision_direction']==1.0)
        ]
        
        styles = {
            'm_l': (majority_liberal, '^', '#1f77b4'),
            'm_c': (majority_conservative, 'v', '#d62728'),
            'd_l': (dissenting_liberal, '^', '#1f77b4'),
            'd_c': (dissenting_conservative, 'v', '#d62728'),
        }
        
        for opinions, marker, color in styles.values():
            year_filed_types = [(row['year_filed'], row[type_token]) for i, row in opinions.iterrows()]

            x = list()
            y = list()
            for year, count in year_filed_types:
                x.append(year)
                y.append(count)

            plt.scatter(x, y, s=100, marker=marker, c=color, linewidths=2)

        if type_token=='type_count':
            plt.title("Corpora type counts by year")
            plt.ylabel("Number of types")
        else:
            plt.title("Corpora token counts by year")
            plt.ylabel("Number of tokens")
            
        years = range(min(df_opinions['year_filed']), max(df_opinions['year_filed'])+1)
        plt.xlabel("Year")
        plt.xticks(years)
        plt.show()
        
        
        
        

        
    
class ProjectVisualizations:
    def __init__(self):
        pass
    
    
    def plot_corpora_scatter(df_opinions):
        year_filed_counts = Counter(list(
            [(row['year_filed'], ) for row in df_opinions.iterrows()]
        ))

        x = list()
        y = list()
        for year, count in year_filed_counts.items():
            x.append(year)
            y.append(count)

#         years = range(min(x), max(x)+1)
#         counts = [year_filed_counts[year] for year in years]

#         plt.bar(x, y, width=1.0)
#         plt.title("ACLU landmark cases (some missing) opinions published by year")
#         plt.xlabel("Year")
#         plt.ylabel("Number of opinions published (any category, e.g. dissenting)")
#         plt.show()


if __name__ == "__main__":
    pass