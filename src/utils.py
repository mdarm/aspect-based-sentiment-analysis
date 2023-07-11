import re
import nltk
import argparse
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup


# Default running parameters
MODEL_NAME = 'logistic_regression'
VECTORISER = 'count_vectoriser'
MAX_FEATURES = None
MAX_DF = 1.0
MIN_DF = 1


# Argument parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=MODEL_NAME, type=str, help='Machine Learning model to use; choose one of "logistic_regression", "random_forest", "svm" and "multinomial_nb"' )
    parser.add_argument('--vectoriser', default=VECTORISER, type=str, help='Vectoriser type; choose one of "count_vectoriser" and "tfidf"' )
    parser.add_argument('--max_features', default=MAX_FEATURES, type=int, help='If not None, build a vocabulary that only considers the top max_features ordered by term frequency across the corpus.' )
    parser.add_argument('--max_df', default=MAX_DF, type=float, help='Ignore terms that have a document frequency strictly higher than the given threshold; applies to both vectorisers.')
    parser.add_argument('--min_df', default=MIN_DF, type=int, help='Ignore terms that have a document frequency strictly lower than the given threshold; applies to both vectorisers.')

    # Parse the aforementioned arguments
    opt = parser.parse_args()

    return opt


# Tokenization of text and set English stopwords
tokenizer = ToktokTokenizer()
nltk.download('stopwords')
stopword_list = nltk.corpus.stopwords.words('english')


# Removing the html strips (if any)
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


# Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


# Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


# Removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9-$\s]'
    text=re.sub(pattern, ' ', text)
    return text


# Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text


# Removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
