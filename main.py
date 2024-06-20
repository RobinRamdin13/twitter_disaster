import os 
import re
import spacy
import pandas as pd
import spacy_cleaner

from tqdm import tqdm 
from typing import List, Union
from spacy import tokens
from pandas import DataFrame
from functools import partialmethod
from spacy_cleaner.processing import removers, mutators, transformers
from spacy_cleaner.processing.evaluators import Evaluator

# disable the tqdm output from spacy_cleaner
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

class MentionEvaluator(Evaluator):
    """Evaluates mentions"""

    def evaluate(self, tok: tokens.Token)-> bool:
        """If the given token is a mention

        Args:
            tok (tokens.Token): token to evaluate

        Returns:
            bool: `True` is the token is a mention. `False` if not
        """
        if str(tok)[0] == '@':
            print(f'@ in {str(tok)}')
            return True
        else:
            return False

def replace_mention_token(
    tok: tokens.Token, replace: str='') -> Union[str, tokens.Token]:
    """If the token is a mention, replace it with empty string.

    Args:
      tok: A `spaCy` token.
      replace: The replacement string.

    Returns:
      The replacement string or the original token.
    """
    # replace = str(tok)[1:]
    return transformers.Transformer(
        MentionEvaluator(), replace
    ).transform(tok)

# define global variables 
random_state = 12345

# define data types
data_type = {
    'id':int,
    'keyword':str,
    'text':str,
    'location':str,
    'target':int}

# define the x and y variables
xlabel = ['id', 'keyword', 'text', 'location']
ylabel = ['target']

# load the nlp model 
nlp = spacy.load('en_core_web_lg')

# instantiate the text cleaning pipeline 
cleaning_pipeline = spacy_cleaner.Cleaner(
    nlp, # spacy model
    removers.remove_email_token, # remove emails
    # replace_mention_token,
    removers.remove_url_token,  # remove urls 
    removers.remove_stopword_token, # remove stop words
    removers.remove_punctuation_token, # remove punctuations 
    removers.remove_number_token, # remove numbers 
    mutators.mutate_lemma_token, # replace tokens by their lemma
)

def main(train_path:str, test_path:str)-> None:
    # ingest the csv files into panda dataframes
    df_train = pd.read_csv(train_path, dtype=data_type, index_col=0)
    df_test = pd.read_csv(test_path)
    
    # # clean the text using spacy
    # df_train['clean_text'] = df_train.text.apply(lambda x: cleaning_pipeline.clean([x.lower()]))
    # df_test['clean_text'] = df_test.text.apply(lambda x: cleaning_pipeline.clean([x.lower()]))

    # # feature extraction
    # # text preprocessing 
    testing = df_train.text.tolist()[-4]
    print(testing)
    print(cleaning_pipeline.clean([testing.lower()]))
    # df_train.text = df_train.text.apply(lambda x: filter_token(nlp(x))) # convert text to documents
    # print(df_train)
    return

if __name__ =="__main__":
    cwd = os.getcwd()
    train_path = os.path.join(cwd, 'data/train.csv')
    test_path = os.path.join(cwd, 'data/test.csv')
    main(train_path, test_path)