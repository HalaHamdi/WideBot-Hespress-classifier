# Hespress Classification
NLP Python code that classifies arabic stories into one of 11 topics such as economics , politiques, sports..etc 

---
## Preprocessing Process
* Splitting the test and training datasets , so that tests set is 20% 
* Test set is only used to record metric and no data leakage  (fresh used)
* Encoding the label from 0 -> 11
* tokenize 
* remove stop words, tashkeel & punctuations

---


## Feature Extraction
* TFIDF
* BOW

---

## Models 
* SVM
* Linear regression
* XGboost
* Decision Trees
- All of which were tried and parameter tunning were made to find suitable hyper parameters.
- GridSearch was  used to find the best model parameters ;however, it took more than 200 mins and was aborted. 
---

## Future Work

1. using NN to extract features such as using Arabert to provide embeddings
2. Use Word2vec or CBOW to give more embeddings
3. Utilize the other columns in the features such as the author name or title 
4. Building CNN to better enhance the models 
5. Explore more of Deep networks for model building






   
