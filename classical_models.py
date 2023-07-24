from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import numpy as np

def svm_model(X_train,y_train,X_test):
    classifier = SVC(kernel='rbf')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    return y_pred

def DTmodel(X_train,y_train,X_test):
    clf=DecisionTreeClassifier()
    clf.fit(X=X_train,y=y_train)
    y_pred=clf.predict(X=X_test)
    return y_pred


def xgboost_model(X_train,y_train,X_test):
    xgb=XGBClassifier()
    xgb_model=xgb.fit(X_train,y_train)
    y_pred=xgb_model.predict(X_test)
    return y_pred

def LRmodel(Xtrain,y_train,X_test):
    clf=LogisticRegression(max_iter=300)
    clf.fit(X=Xtrain,y=y_train)
    y_pred=clf.predict(X=X_test)
    return y_pred

def GridSearch(X_train,y_train,X_test,y_test):

    # List of C values
    # C_range = np.logspace(-1, 1, 3)
    C_range=[0.01, 0.1, 10]
    print(f'The list of values for C are {C_range}')
    # List of gamma values
    gamma_range = np.logspace(-1, 1, 3)
    print(f'The list of values for gamma are {gamma_range}')

    # Define the search space
    param_grid = { 
        # Regularization parameter.
        "C": C_range,
        # Kernel type
        "kernel": ['rbf','linear'],
        }
    # Set up score
    scoring = ['accuracy']

    # Set up the k-fold cross-validation
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)


    svc = SVC()
    # Define grid search
    grid_search = GridSearchCV(estimator=svc, 
                            param_grid=param_grid, 
                            scoring=scoring, 
                            refit='accuracy', 
                            n_jobs=-1, 
                            cv=kfold, 
                            verbose=0)
    # Fit grid search
    grid_result = grid_search.fit(X_train, y_train)

    # Print the best accuracy score for the training dataset
    print(f'The best accuracy score for the training dataset is {grid_result.best_score_:.4f}')
    # Print the hyperparameters for the best score
    print(f'The best hyperparameters are {grid_result.best_params_}')
    # Print the best accuracy score for the testing dataset
    print(f'The accuracy score for the testing dataset is {grid_search.score(X_test, y_test):.4f}')



def get_metrics(y_true, y_pred, avg='weighted', to_print=True):
    '''get metrics for classification'''
    precision = precision_score(y_true, y_pred, average=avg)
    recall = recall_score(y_true, y_pred, average=avg)
    f1 = f1_score(y_true, y_pred, average=avg)
    accuracy = accuracy_score(y_true, y_pred)
    
    if to_print:
        # Print overall metrics
        print('Overall Metrics:')
        print('Precision (weighted): %.3f' % precision)
        print('Recall (weighted): %.3f' % recall)
        print('F1 Score (weighted): %.3f' % f1)
        print('Accuracy: %.3f' % accuracy)
        
        # Print metrics for each class
        class_report = classification_report(y_true, y_pred)
        print('\nMetrics for Each Class:')
        print(class_report)
        
    return precision, recall, f1, accuracy
