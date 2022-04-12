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
from wordcloud import WordCloud
from gensim.models import KeyedVectors

from wefe.metrics import WEAT
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel

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

# (Charlesworth, 2021)
# https://osf.io/2tvh5/
charlesworth_2021_words = {
    "female": "she her mommy mom girl mother lady sister mama momma sis grandma herself".split(),
    "male": "he his daddy dad boy father guy brother dada papa bro grandpa himself".split(), 
    "femalenew": "hers women woman daughter wife gal female girlfriend queen grandmother aunt granddaughter niece gentlewoman sorority bachelorette princess dame gentlewomen stepmother ms madam bride stepdaughter maiden godmother grandma missus heroine motherhood sororities ma".split(),
    "malenew": "him man men son husband guy male boyfriend king grandfather uncle grandson nephew lad gentleman fraternity bachelor prince dude gentlemen stepfather sir bloke groom stepson suitor godfather fella hero fatherhood fraternities pa".split(),
    "femalemax": "her hers she women woman herself daughter mother wife gal girl sister female mom girlfriend queen grandmother aunt granddaughter niece lady gentlewoman sorority bachelorette princess dame gentlewomen stepmother mommy ms madam bride stepdaughter maiden godmother grandma missus heroine motherhood sororities mama ma".split(),
    "malemax": "he his him man men himself son father husband guy boy brother male dad boyfriend king grandfather uncle grandson nephew lad gentleman fraternity bachelor prince dude gentlemen stepfather daddy sir bloke groom stepson suitor godfather grandpa fella hero fatherhood fraternities papa pa".split(),
    "home": "baby house home wedding kid family marry".split(), 
    "work": "work office job business trade activity act money".split(), 
    "homemax": "baby house home wedding kid family marry marriage parent caregiving children cousins relative kitchen cook".split(), 
    "workmax": "work office job business trade money manage executive professional corporate corporation salary career hire hiring".split(),
    "reading": "book read write story word writing reading tale".split(), 
    "math": "puzzle number count math counting calculator subtraction addition".split(), 
    "readingmax": "book read write story word writing reading tale novel literature narrative sentence paragraph phrase diary notebook fiction nonfiction".split(), 
    "mathmax": "puzzle number count math counting calculator subtraction addition multiplication division algebra geometry calculus equations computation computer".split(), 
    "arts": "art dance dancing sing singing paint painting song draw drawing".split(), 
    "science": "science scientist chemistry physic engineer space spaceship astronaut chemical microscope".split(), 
    "artsmax": "art dance dancing dancer jig sing singing singer paint painting portrait sketch picture painter song draw drawing poet poetry symphony melody music musician sculpture create artistry designer compose".split(), 
    "sciencemax": "science scientist chemistry chemist physic doctor medicine engineer space spaceship astronaut chemical microscope technology Einstein NASA experiment astronomy data analyze atom molecule lab".split(), 
    "instruments": "guitar banjo drum tuba bell fiddle harpsichord piano flute horn violin".split(), 
    "weapons": "weapon arrow club gun spear sword dynamite rifle tank bomb knife cannon slingshot whip".split(), 
    "good": "good happiness happy fun fantastic lovable magical delight joy relaxing honest excited laughter lover cheerful".split(), 
    "bad": "bad torture murder abuse wreck die disease disaster mourning virus killer nightmare stress kill death".split(),
}

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
        print(df_opinions.drop_duplicates(subset=['case_name']).value_counts('scdb_decision_direction'))
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
    
    #     GENSIM KEYED VECTORS FOR WEAT TEST
    def svd_keyed_vectors(svd, x2i, k=1000):
        model_kv = KeyedVectors(k)
        keys = [word for k, word in enumerate(x2i)]
        vectors = [svd['U'][x2i[word]] for k, word in enumerate(x2i)]
        # for k, word in enumerate(x2i):
            # model_kv.add_vector(word, svd['U'][x2i[word]])
        model_kv.add_vectors(keys, vectors)
        return model_kv
    
    def kv_weat_test(model_kv, query, lost_vocab_threshold=0.9):
        model_we = WordEmbeddingModel(model_kv, 'SCOTUS') 
        metric = WEAT()
        result = metric.run_query(
            query, model_we,
            lost_vocabulary_threshold=lost_vocab_threshold,
            calculate_p_value=True, p_value_iterations=10000
        )
        print(result)
        return result
    
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
            
        years = range(min(df_opinions['year_filed']), max(df_opinions['year_filed'])+1, 5)
        plt.xlabel("Year")
        plt.xticks(years)
        plt.show()
        
    def plot_ngram_barchart(ngram_counter, top=20):
        most_common = ngram_counter.most_common()[:top]
        fig, ax = plt.subplots()
        # Example data
        tokens = list(map(lambda token_freq: token_freq[0], most_common))
        y_pos = np.arange(len(tokens))
        freqs = list(map(lambda token_freq: token_freq[1], most_common))

        ax.barh(y_pos, freqs, align='center')
        ax.set_yticks(y_pos, labels=tokens)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Frequency')
        # ax.set_title(f'{author_name} ngram frequency')
        plt.show()
        
    def plot_unigram_wordcloud(ugram_counter):
        wordcloud_unigrams = dict([[k,v] for k,v in ugram_counter.items()])
        wordcloud = WordCloud(background_color=None, mode='RGBA', max_words=100).generate_from_frequencies(wordcloud_unigrams)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
        
    def most_common_unigram_heatmap(ugram_counter, bigram_counter, top=10):
        min_token_len = 3
        most_common = ugram_counter.most_common(100)
        top_unigrams = [k for k, v in most_common if len(k) >= min_token_len][:top]
        i2t = dict([[k,v] for k,v in enumerate(top_unigrams)])
        heatmap_bigram = np.zeros((top,top), int)
        for i in range(top):
            for j in range(top):
                heatmap_bigram[i,j] = bigram_counter[(i2t[i], i2t[j])]
        heatmap_bigram = np.log(heatmap_bigram + 1)
        ax = sns.heatmap(heatmap_bigram, linewidth=0.5,  xticklabels=top_unigrams, yticklabels=top_unigrams, cbar_kws={'label': 'log(freq + 1)'})
        plt.title(f'Heatmap of bigrams for top {top} unigrams (min {min_token_len} chars)')
        plt.show()
        return heatmap_bigram

    def most_common_bigram_heatmap(bigram_counter, top=10):
        min_token_len = 3
        most_common = bigram_counter.most_common(100)
        tokens = []
        for bigram, freq in most_common:
            for token in bigram:
                if token not in tokens and len(token) >= min_token_len:
                    tokens.append(token)
        top_tokens = tokens[:top]

        i2t = dict([[k,v] for k,v in enumerate(top_tokens)])
        heatmap_bigram = np.zeros((top,top), int)
        for i in range(top):
            for j in range(top):
                heatmap_bigram[i,j] = bigram_counter[(i2t[i], i2t[j])]
        ax = sns.heatmap(heatmap_bigram, linewidth=0.5, xticklabels=top_tokens, yticklabels=top_tokens, cbar_kws={'label': 'freq'})
        plt.title(f'Heatmap of top {top} bigrams (min {min_token_len} chars)')
        plt.show()
        return heatmap_bigram
        

        

if __name__ == "__main__":
    pass