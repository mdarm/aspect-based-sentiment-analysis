import os
import joblib
import utils
import pandas as pd
from split import load_xmls
from test import test_model
from experiments import run_experiment
from train import xml_to_dataframe, train_model


def run(model_name, vectoriser_name, max_features, min_df, max_df):
    # Load and split XML files.
    load_xmls('ABSA16_Restaurants_Train_SB1_v2.xml')

    # Files for training and testing
    indices = [x + 1 for x in range(10)]

    # Run the cross-validation experiments and print the average & individual F1 scores
    avg_f1, idv_f1 = run_experiment(indices, n_splits=10,
                                    model_name=model_name,
                                    vectoriser_name=vectoriser_name,
                                    max_features = max_features,
                                    ngram_range = (1, 1),
                                    min_df = min_df,
                                    max_df = max_df)

    print(f'Average F1 score: {avg_f1}')
    print(f'Sentiment F1 scores: Neutral: {idv_f1[0]}, Positive: {idv_f1[1]}, Negative: {idv_f1[2]}')


if __name__ == "__main__":

    opt = utils.parse_args()

    run(model_name=opt.model_name, vectoriser_name=opt.vectoriser, max_features=opt.max_features, min_df=opt.min_df, max_df=opt.max_df)
