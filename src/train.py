import os
import csv
import joblib
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from utils import strip_html, remove_between_square_brackets, denoise_text, remove_special_characters, simple_stemmer, remove_stopwords
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from gensim.models import Word2Vec
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def xml_to_dataframe(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Prepare lists to store the data
    review_ids = []
    sentence_ids = []
    texts = []
    categories = []
    polarities = []

    # Extract data from XML
    for review in root.findall('.//Review'):
        review_id = review.get('rid')
        for sentence in review.findall('.//sentence'):
            sentence_id = sentence.get('id')
            text = sentence.find('text').text
            opinions = sentence.find('Opinions')
            if opinions is not None:
                for opinion in opinions.findall('Opinion'):
                    category = opinion.get('category')
                    polarity = opinion.get('polarity')
                    review_ids.append(review_id)
                    sentence_ids.append(sentence_id)
                    texts.append(text)
                    categories.append(category)
                    polarities.append(polarity)
            else:
                review_ids.append(review_id)
                sentence_ids.append(sentence_id)
                texts.append(text)
                categories.append('')
                polarities.append('')

    # Create a DataFrame from the lists
    dataframe = pd.DataFrame({
        'Review ID': review_ids,
        'Sentence ID': sentence_ids,
        'Text': texts,
        'Category': categories,
        'Polarity': polarities
    })

    return dataframe


def train_model(dataframe, model_name = 'logistic_regression', vectoriser_name = 'count_vectoriser', max_features = None, ngram_range = (1, 1), max_df = 1, min_df = 1):
    # Shuffle the DataFrame
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    
    # Define the valid Polarity values
    valid_polarities = ['neutral', 'positive', 'negative']

    # Filter the DataFrame
    dataframe = dataframe[dataframe['Polarity'].isin(valid_polarities)]

    # Concatenate 'Text' and 'Category' columns (implicitly account for aspect)
    dataframe['Text_Category'] = dataframe['Text'] + " " + dataframe['Category']

    # Remove noise, special characters; perform stemming and remove stop words
    dataframe['Text_Category'] = dataframe['Text_Category'].apply(denoise_text) 
    dataframe['Text_Category'] = dataframe['Text_Category'].apply(remove_special_characters) 
    dataframe['Text_Category'] = dataframe['Text_Category'].apply(simple_stemmer) 
    dataframe['Text_Category'] = dataframe['Text_Category'].apply(remove_stopwords) 

    # Target variable is 'Polarity' and the feature is 'Text' & 'Aspect'
    if vectoriser_name == 'count_vectoriser':
        vectorizer = CountVectorizer(min_df       = min_df,
                                     max_df       = max_df,
                                     max_features = max_features,
                                     ngram_range  = ngram_range 
                                     )
        X = vectorizer.fit_transform(dataframe['Text_Category'])
    elif vectoriser_name == 'tfidf':
        vectorizer = TfidfVectorizer(min_df       = min_df,
                                     max_df       = max_df,
                                     max_features = max_features,
                                     ngram_range  = ngram_range 
                                     )
        X = vectorizer.fit_transform(dataframe['Text_Category'])
    elif vectoriser_name == 'word2vec':
        # Train a Word2Vec model
        sentences = [row.split() for row in dataframe['Text_Category']]
        vectorizer = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
        vectorizer.save("word2vec.model")

        # Average the word vectors for each sentence
        X = np.array([np.mean([vectorizer.wv[word] for word in sentence], axis=0) for sentence in sentences])

    # Labeling the sentient data
    encode = {
        'neutral': 0,
        'positive': 1,
        'negative': 2
    }

    # Transformed sentiment data
    y = dataframe['Polarity'].apply(lambda label: encode[label])

    # Create a model instance
    if model_name == 'logistic_regression':
        model = LogisticRegression(class_weight='balanced')
    elif model_name == 'random_forest':
        model = RandomForestClassifier(class_weight = 'balanced')
    elif model_name == 'svm':
        model = SVC(class_weight = 'balanced')
    elif model_name == 'multinomia_nb':
        model = MultinomialNB()

    
    # Train the model
    model.fit(X, y)
    print(model)

    # Save the trained model and vectorizer to disk
    joblib.dump(model, 'trained_model.pkl')
    if vectoriser_name != 'word2vec':
        joblib.dump(vectorizer, 'vectorizer.pkl')
