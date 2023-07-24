import pandas as pd 
import os 
from pyarabic.araby import strip_tashkeel, tokenize
import string
import arabicstopwords.arabicstopwords as stp

def train_test_split():
    '''
    This function aims to read all the stories dataset 
    and create a testa and train splits with proprotion 0.8,0.2 resp. 
    '''
    path = './dataset/'
    filenames = os.listdir(path)
    for file in filenames:
        if file.startswith("stories"):
            data = pd.read_csv(path + file, usecols=lambda col: col != 'Unnamed: 0')

            split_index = int(len(data) * 0.8)

            # Split the data into train and test sets
            train_data = data[:split_index]
            test_data = data[split_index:]

            # Append the train and test sets to CSV files

            with open('train.csv', 'a',encoding='utf-8') as f:
                train_data.to_csv(f, header=f.tell() == 0, index=False)
            with open('test.csv', 'a',encoding='utf-8') as f:
                test_data.to_csv(f, header=f.tell() == 0, index=False)


def encode_category(train, test):

    '''
    This function uses label encoding to convert your categorical labels into numerical values.
    
    '''
    labels = train['topic'].unique()
    encode = {}
    
    for i, l in enumerate(labels):
        encode[l] = i
    
    train['topic'] = train['topic'].map(encode)
    test['topic'] = test['topic'].map(encode)



    



def preprocess_arabic_text(text):
    '''
    This function aims to tokenize, remove punctuation, tashkeel, Arabic stop words.
    '''
    # Tokenization
    tokens = tokenize(text)

    # Remove punctuation and non-Arabic characters
    arabic_tokens = [token for token in tokens if all(c not in string.punctuation for c in token)]

    # Remove diacritics (Tashkeel)
    stripped_tokens = [strip_tashkeel(token) for token in arabic_tokens]
    
   
    
    arabic_stop_words = set(stp.stopwords_list())

    # Remove Arabic stop words
    cleaned_tokens = [token for token in stripped_tokens if token not in arabic_stop_words]

    res=''
    for word in cleaned_tokens:
        res+=' '
        res+=word

    return res



