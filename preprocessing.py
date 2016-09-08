### Pre-Processing
#=====================================================================================================










### GridSearch
#=====================================================================================================
# Gridsearch is used for searching for the best parameters
from sklearn.grid_search import GridSearchCV

# Setup our GridSearch Parmaters
search_parameters = {
    'fit_intercept':  [True, False], 
    'normalize':      [False, True]
}

# Intialize a blank model object
lm = LinearRegression()

# Initialize gridsearch
estimator = grid_search.GridSearchCV(lm, search_parameters, cv=5, verbose=1, n_jobs=4)

# Fit some data!
results = estimator.fit(trainX, trainY)

results.param_grid      #Displays parameters used
results.best_score_     #Best score achieved
results.best_estimator_   #Reference to model with best score. Is usable / callable.
results.best_params_    #The parameters that have been found to perform with the best score.
results.grid_scores_    #Display score attributes with cooresponding parameters
#=====================================================================================================










### Normalization, Standardization
#=====================================================================================================
# Normalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
# fit_transform calculates the pre-scaled mean and standard deviation, can use .mean_ and .scale_
scaled_data = scaler.fit_transform(df)


#-----------------------------------------------------------------------------------------------------


# Standardization
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()
# fit_transform calculates the pre-scaled mean and standard deviation, can use .mean_ and .scale_
scaled_data = scaler.fit_transform(df)
#=====================================================================================================










### Preprocessing Techniques
#=====================================================================================================
## Finding OPTIMAL parameter
"""
Lasso, Ridge and Elastic Net are better for smaller datasets
because they take up too much memory and slow down
"""
# Lasso (Complete with GridSearch)
from sklearn.linear_model import Lasso
lasso = Lasso()
parameters
	(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, 
	max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')

from sklearn.linear_model import LassoCV

parameters
	(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=False, precompute='auto', 
	max_iter=1000, tol=0.0001, copy_X=True, cv=None, verbose=False, n_jobs=1, positive=False, 
	random_state=None, selection='cyclic')

# Create a features and their coefficients in a dataframe
feature_importance = pd.DataFrame({ 'features':X.columns, 
                                   'coefficients':model.coef_
                                  })

feature_importance.sort_values('coefficients', ascending=False, inplace=True)
feature_importance


#-----------------------------------------------------------------------------------------------------


# Ridge (Complete with GridSearch)
from sklearn.linear_model import Ridge
ridge = Ridge()
parameters
	(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, 
	tol=0.001, solver='auto', random_state=None

from sklearn.linear_model import RidgeCV

parameters
	(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False, scoring=None, 
	cv=None, gcv_mode=None, store_cv_values=False)

# Create a features and their coefficients in a dataframe
feature_importance = pd.DataFrame({ 'features':X.columns, 
                                   'coefficients':model.coef_
                                  })

feature_importance.sort_values('coefficients', ascending=False, inplace=True)
feature_importance


#-----------------------------------------------------------------------------------------------------


# Elastic Net (Complete with GridSearch)
from sklearn.linear_model import ElasticNet
en = ElasticNet()
parameters
	(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, 
	copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')

from sklearn.linear_model import ElasticNetCV

parameters	
	(l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=False, 
	precompute='auto', max_iter=1000, tol=0.0001, cv=None, copy_X=True, verbose=0, n_jobs=1, 
	positive=False, random_state=None, selection='cyclic')

# Create a features and their coefficients in a dataframe
feature_importance = pd.DataFrame({ 'features':X.columns, 
                                   'coefficients':model.coef_
                                  })

feature_importance.sort_values('coefficients', ascending=False, inplace=True)
feature_importance


#-----------------------------------------------------------------------------------------------------


# Stochastic Gradient Descent
"""
SGD is better for median and larger datasets.
"""
from sklearn.linear_model import SGDRegressor, SGDClassifier

# SGDRegressor - For Regression
# SGDClassifier - For Categorical

sgd_params = {
    'loss':['squared_loss'],
    'penalty':['l1','l2'],
    'alpha':np.linspace(0.01, 10000, 100)
}

# Create a features and their coefficients in a dataframe
feature_importance = pd.DataFrame({ 'features':X.columns, 
                                   'coefficients':model.coef_
                                  })

feature_importance.sort_values('coefficients', ascending=False, inplace=True)
feature_importance
#=====================================================================================================










### NLP
#=====================================================================================================
# Basic way, cannot handle large amounts of data
from collections import Counter
print Counter(string_of_words.lower().split())

# Count Vectorizer 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
cvec = CountVectorizer()
cvec.fit([spam])
CountVectorizer(analyzer=u'word', binary=False, decode_error=u'strict',
        dtype=<type 'numpy.int64'>, encoding=u'utf-8', input=u'content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 1), preprocessor=None, stop_words=None,
        strip_accents=None, token_pattern=u'(?u)\\b\\w\\w+\\b',
        tokenizer=None, vocabulary=None)

# Creates a df that lists numbers of uniques and the unique 'string'
df  = pd.DataFrame(cvec.transform([spam]).todense(),
             columns=cvec.get_feature_names())

df.transpose().sort_values(0, ascending=False).head(10).transpose()

# Hash Vectorizer, more for big data
from sklearn.feature_extraction.text import HashingVectorizer
hvec = HashingVectorizer()
hvec.fit([spam])

df  = pd.DataFrame(hvec.transform([spam]).todense())
df.transpose().sort_values(0, ascending=False).head(10).transpose()

# Breaks up sentences and puts them into an array
from nltk.tokenize import PunktSentenceTokenizer
easy_text = "I went to the zoo today. What do you think of that? I bet you hate it! Or maybe you don't"
sent_detector = PunktSentenceTokenizer()
sent_detector.sentences_from_text(easy_text)

"""
Out[6]: 
['I went to the zoo today.',
 'What do you think of that?',
 'I bet you hate it!',
 "Or maybe you don't"]
"""

# Auto stems the best way
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
print stemmer.stem('Swimmed')
print stemmer.stem('Swimming')
"""
Output:
Swim
Swim
"""

# Stop words are to remove words that are very common and provide no information on text content
from nltk.corpus import stopwords
stop = stopwords.words('english')
sentence = "this is a foo bar sentence"
print [i for i in sentence.split() if i not in stop]
"""
Output:
['foo', 'bar', 'sentence']
"""
#=====================================================================================================






