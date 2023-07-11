import os
import joblib
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from split import load_xmls
from test import test_model
from train import xml_to_dataframe, train_model
from sklearn.model_selection import KFold


def run_experiment(indices, n_splits=10, model_name='logistic_regression', vectoriser_name='count_vectoriser', max_features = None, ngram_range = (1, 1), min_df = 1, max_df = 1):
    # Initialize the KFold cross-validator
    kf = KFold(n_splits=n_splits)
        
    # Initialize a list to store the F1 scores for each fold
    f1_scores = []
    f1_individual_scores = []

    # Perform cross-validation
    for train_idx, test_idx in kf.split(indices):
        # Add 1 to the indices
        train_idx = [index + 1 for index in train_idx]
        test_idx = [index + 1 for index in test_idx]
        print(train_idx, test_idx) 
        # Initialize an empty DataFrame to store training data
        all_data = pd.DataFrame()

        # Process each XML file for training
        for index in train_idx:
            xml_file = f'part{index}.xml'
            if os.path.exists(xml_file):
                data = xml_to_dataframe(xml_file)
                all_data = pd.concat([all_data, data])

        # Train the model
        train_model(all_data, model_name = model_name, vectoriser_name=vectoriser_name, max_features = max_features, ngram_range = ngram_range, min_df = min_df, max_df = max_df)

        # Load the saved model and vectorizer
        model = joblib.load('trained_model.pkl')
        if vectoriser_name == 'word2vec':
            vectorizer = Word2Vec.load("word2vec.model")
        else:
            vectorizer = joblib.load('vectorizer.pkl')

        # Read the XML file and convert it to a DataFrame
        xml_file = f'part{test_idx[0]}.xml'
        if os.path.exists(xml_file):
            test_data = xml_to_dataframe(xml_file)

        # Test the model and append the F1 score to the list
        f1_overall, f1_individual = test_model(test_data, model, vectorizer, vectoriser_name)
        f1_scores.append(f1_overall)
        f1_individual_scores.append(f1_individual)

    # Return the average F1 score
    return sum(f1_scores) / len(f1_scores), [sum(score[i] for score in f1_individual_scores) / len(f1_individual_scores) for i in range(len(f1_individual_scores[0]))]
