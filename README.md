# Aspect-Sentiment-Analysis

This repository contains the source code for an Aspect Sentiment Analysis project. The project involves reading data from an XML file, splitting the data into ten subfiles, and subsequently training a model using nine of these files. Both the vectoriser and the model are saved for future use. The remaining XML file is used for testing purposes and performing an 10-fold cross-validation.

## Code Structure

The code is initiated from [main.py](src/main.py).

## Requirements

To ensure all functionalities of the project run as expected, the following Python libraries need to be installed:

- `joblib==1.0.1`
- `pandas==1.2.4`
- `scikit-learn==0.24.2`
- `nltk==3.6.2`
- `gensim==4.3.1`
- `beautifulsoup4==4.9.3`

You can install these libraries by running:

```python
pip install -r requirements.txt
```

## Setting up the project
The project requires you to have XML files for training and testing. For this purpose, we have a function `load_xmls` in [split.py](split.py) that takes as argument the name of your XML file and splits it into 10 subfiles. 


## Training
The [train.py](src/train.py) script loads the 9 XML files, vectorises them, and fits them to a model. Both the vectoriser and the model are saved for future use.

## Testing
The [test.py](src/test.py) script uses the remaining XML file, transforms it using the saved vectoriser, and fits it to the saved model.

## Cross-Validation
The [experiments.py](src/experiments.py) script performs 10-fold cross-validation for various vectorisers and models, and returns the average F1 score along with the class individual score.

## Executing the code
To execute the code, use the command:

```python
python main.py --model_name [model_name] --vectoriser [vectoriser] --max_features [max_features] --min_df [min_df] --max_df [max_df]
```

Example of execution with a personalized configuration:

```python
python main.py --model_name random_forest --vectoriser word2vec --max_features 2000 --min_df 1 --max_df 0.5
```

## [main.py](src/main.py) parameters:
The parameters for [main.py](main.py) are defined in [utils.py](utils.py).
- `model_name`: the name of the model to be used for training and testing; can be 'logistic_regression', 'random_forest', 'svm' and 'multinomial_nb'
- `vectoriser`: the name of the vectoriser to be used for training and testing; can be 'count_vectoriser', 'tfidf' and 'word2vec'
- `max_features`: the maximum number of features the vectorizer should consider; takes integer values
- `min_df`: the minimum document frequency for the Count Vectorizer; takes values from 0.0 to 1.0
- `max_df`: the maximum document frequency for the Count Vectorizer; takes values from 0.0 to 1.0

**Note:** All default running options, as well as the stemming\tokenisation\text-cleaning, can be found within the [utils.py](src/utils.py)
