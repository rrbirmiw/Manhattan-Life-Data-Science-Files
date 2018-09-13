import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import pandas as pd
from pathlib import Path
import nltk


def lemmatize_stemming(text):
    stemmer = PorterStemmer()
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 0:
            result.append(lemmatize_stemming(token))
    return result

def preprocess_helper(train_text_df):
        data_text = train_text_df[[0]]
        data_text['index'] = data_text.index
        processed_doc = data_text[0].map(preprocess)
        dictionary =  gensim.corpora.Dictionary(processed_doc)
        return(processed_doc, dictionary)

class LDA_Classifier():
    def __init__(self, num_topics, training_text_df):
        self.num_topics = num_topics
        self.train_text_df = training_text_df
        self.dictionary = None
        self.processed_doc = None
        self.lda_model = None

    def preprocess(self):
        (self.processed_doc, self.dictionary) = preprocess_helper(self.train_text_df)

    def train(self):
        bow_corpus = [self.dictionary.doc2bow(doc) for doc in self.processed_doc]
        self.lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=self.num_topics, id2word=self.dictionary, passes=2, workers=2)

    def classify(self, unseen_text_string):

        bow_vector = self.dictionary.doc2bow(preprocess(unseen_text_string))
        for index, score in sorted(self.lda_model[bow_vector], key=lambda tup: -1*tup[1]):
            print("Score: {}\t Topic: #{} {}".format(score, index, self.lda_model.print_topic(index, 5)))

        print(".....NOW RETURNING THE ARGMAX TOPIC BASED ON SCORE...")
        return(sorted(self.lda_model[bow_vector], key=lambda x: x[1])[-1][0])

def example_run():
    root = Path(__file__).parents[1]
    print(root)
    icd_mapping = pd.read_csv(root / "relevant_data_files" / "icd_code_description_map.csv",delimiter=",")

    data_text = icd_mapping[['Des']]
    data_text['index'] = data_text.index
    data_text.columns=[0,'index']

    my_lda_classifier = LDA_Classifier(num_topics=25,training_text_df=data_text )
    my_lda_classifier.preprocess()
    my_lda_classifier.train()

    test_string = "typhoid fever and swelling"
    (classified_topic) =  my_lda_classifier.classify(test_string)
    print(classified_topic)

if __name__ == "__main__":
    nltk.download('wordnet')
    np.random.seed(2018)
    example_run()
