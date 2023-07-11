import re
from sklearn.metrics import f1_score
from utils import strip_html, remove_between_square_brackets, denoise_text, remove_special_characters, simple_stemmer, remove_stopwords


def test_model(dataframe, model, vectorizer):

    # Define the valid Polarity values and drop everything else
    valid_polarities = ['neutral', 'positive', 'negative']
    dataframe = dataframe[dataframe['Polarity'].isin(valid_polarities)]

    # Concatenate 'Text' and 'Category' columns (implicitly account for aspect)
    dataframe['Text_Category'] = dataframe['Text'] + " " + dataframe['Category']

    # Remove noise, special characters; perform stemming and remove stop words
    dataframe['Text_Category'] = dataframe['Text_Category'].apply(denoise_text) 
    dataframe['Text_Category'] = dataframe['Text_Category'].apply(remove_special_characters) 
    dataframe['Text_Category'] = dataframe['Text_Category'].apply(simple_stemmer) 
    dataframe['Text_Category'] = dataframe['Text_Category'].apply(remove_stopwords) 

    # Transform the text data with the loaded vectorizer
    X_test = vectorizer.transform(dataframe['Text_Category'])

    # Labeling the sentient data
    encode = {
        'neutral': 0,
        'positive': 1,
        'negative': 2
    }

    # Transformed sentiment data
    y_test = dataframe['Polarity'].apply(lambda label: encode[label])

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate and return the F1 score
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_individual = f1_score(y_test, y_pred, average=None)

    return f1_weighted, f1_individual
