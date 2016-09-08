#### Roadmap for Statistical Modeling
"""
Made for General Assembly's Data Science Immersive Cohort 2
Roadmap/Cheatsheet for Data Input, Learning, Munging, Exploring, Modeling, Definitions, Web Scraping
"""





### Uploading the CSV file
#=====================================================================================================
import pandas as pd
import numpy as np

df = pd.read_csv('../../../example.csv')
#=====================================================================================================





### Learn about the data
#=====================================================================================================
df.info() # Tells you the shape and the types
df.shape # Tells you the shape of the dataframe
df.dtypes # Tells you the data type for each column
df.head() # Prints out first 5 rows
df.tail() # Prints out last 5 rows
df.columns # Prints out all column names
df['col1'].unique() # Prints out all unique values in col1
df['col1'].value_counts() # Prints out each unique value and # of instances
#=====================================================================================================





### Clean the data
#=====================================================================================================
## Change Yes/No, True/False into 0 and 1
df['col1'].map(lambda x: 1 if x == 'Yes' else 0)

## Remove all $ signs, or any strings, in dataframes and change to a float, if needed, with .apply
def remove_dollar_sign(x):
	if '$' in x:
		return float(x.replace('$', ''))
	else:
		return x

df['col'].apply(remove_dollar_sign)

# Simple way to change all values in column to different type with .apply
df['col'].apply(float)

## Rename columns
df.rename(columns={'col1':'new_col1', 'col2':'new_col2'}, inplace=True)

# Another way
update_columns = ['new_col1', 'new_col2']
new_df = pd.DataFrame[data=df, columns=update_columns]

## Drop/Edit Nan Values
# Drop rows with NaN values
df.dropna()
df['col1'].dropna()

# Edit NaN Values
df.fillna('new_value')
df['col1'].fillna('new_value')

## Create new column
df['new_col'] = np.mean(df['col1']) / np.mean(df['col2'])

## Masking
# new_col has to be True AND col1 has to be 0 OR col2 does not equal to 'you'
new_mask = (df['new_col'] == True) & (df['col1'] == 0) | (df['col2'] != 'you')
df[new_mask]

## Quick way to create new dataframe with select columns from another dataset
new_df = df[['col1', 'col2', 'col3']]

## Dealing with outliers
# Removing
def reject_outliers(data, m=1.5):
    data[abs(data - np.mean(data)) > m * np.std(data)]

# Dropping columns that start with 'objective'
df.drop([c for c in df.columns if c.startswith('objective')], axis=1, inplace=True)

## Indexing
"""
.loc - indexes with the labels for rows and columns
.iloc - indexes with the integer positions for rows and columns
.ix - indexes with both lebals and integer positions
"""
#=====================================================================================================





### Hypothesis, EDA and Graphing
#=====================================================================================================
## Hypothesis
# This is where you setup your null hypothesis for testing.
# A preferable null hypothesis would be:
	# In the winter season, whiskey sales are sold 20% higher than any other alcohol.


#-----------------------------------------------------------------------------------------------------


## EDA
from scipy import stats
stats.mode(a)

# correlation matrix
df.corr() # method='pearson' / 'spearman' / 'kendall'
# covariance matrix
df.cov()
# displays all descriptive statistics
df.describe()
# displays all unique values in 1 column and counts them
df['col'].value_counts()
# displays all unique values in 1 column
df['col'].unique()

x = [1,2,3,4,5]
# returns the mean of the array
np.mean(x)
# returns the median of the array
np.median(x)
# returns the sum of the array
np.sum(x)
# returns the size/shape of the array
np.size(x)
# returns the variance of the array
np.var(x)
# returns the standard deviation of the array
np.std(x)
# returns the square root of the array
np.sqrt(x)
# returns the count occurance of the array 
.count(x)

## Groupby
new_df = df.groupby(['col1'])[['col2', 'col3']].mean().reset_index()
new_df.sort_values('col2', axis=1)
# Groupby col1 with the mean values of col2 and col3, reset the index and sorted by col2
# Must have .size() .mean() .sum() etc... for groupby to work

## Pivot Tables - Long to Wide
df_wide = pd.pivot_table(df_long, # The Data frame you want to convert
                        columns=['col'], # The values in the long df you want to assign for the wide dataframe
                        values='value', # The values in the long df you want to pivot to the wide dataframe
                        index=['subject_id'], # The columns in the long df you want to become the index for the wide dataframe
                        aggfunc=np.mean, # Aggregate function that defaults to the mean, can put own function in, works like .apply 
                        fill_value=np.nan) # Fills in all empty values as assigned value

# Pivot Table example
   A   B   C      D
0  foo one small  1
1  foo one large  2
2  foo one large  2
3  foo two small  3
4  foo two small  3
5  bar one large  4
6  bar one small  5
7  bar two small  6
8  bar two large  7

table = pivot_table(df, values='D', index=['A', 'B'], columns=['C'], aggfunc=np.sum, fill_value='hi')

# Output:
          small  large
foo  one  1      4
     two  6      'hi'
bar  one  5      4
     two  6      7

## Melt() - Wide to Long
df_long = pd.melt(df_wide, # The Data frame you want to convert
                  id_vars=['col1','col2'], # The identifiers for the other columns
                  value_vars=, # The value that identifies to each id_vars
                  var_name=, # The column name for value_vars
                  value_name=) # The column name for the values for each value_vars

# Melt example
df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
                   'B': {0: 1, 1: 3, 2: 5},
                   'C': {0: 2, 1: 4, 2: 6}})

   A  B  C
0  a  1  2
1  b  3  4
2  c  5  6

pd.melt(df, id_vars=['A'], value_vars=['B'], var_name='myVarname', value_name='myValname')

# Output:
   A myVarname  myValname
0  a         B          1
1  b         B          3
2  c         B          5


## Merging
new_df = df1.merge(df2, on='Id', how='left')
new_df = pd.merge(df1, df2, on='Id', how='right')
# Multiple merges in 1 line
new_df = df1.merge(df2, on='Id', how='left').merge(df3, on='Name', how='inner').merge(df4, on='Password', how='outer')


#-----------------------------------------------------------------------------------------------------


### Graphing
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')

# Plot within the jupyter notebook
%matplotlib inline
# Basically, the HD version
%config InlineBackend.figure_format = 'retina'

# Create a figure size
fig = plt.figure(figsize=(7,7))
ax = fig.gca()

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

## Scatterplot
# Seaborn
ax = sns.regplot(x=x, y=y, data=df, color='green', marker='+', ci=68, x_estimater=x, y_jitters=0.1, 
                 x_bins=5, truncate=True, logistic=False)
# Matplotlib
ax.scatter(x, y, s=150, c='blue', label='accepted')
ax.scatter(x, y, s=100, c='orange', label='rejected')

ax.set_ylabel('y label', fontsize=16)
ax.set_xlabel('x label', fontsize=16)
ax.set_title('I am title', fontsize=20)

ax.set_xlim([2.,5.])
ax.set_ylim([-0.1, 1.1])

plt.legend(loc='upper left')
plt.show()

# Add labels to plots
for label, x, y in zip(labels, x, y):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'orange', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    
plt.show()
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

## Histogram
# Seaborn
ax = sns.distplot(x, name='x variable', fit=norm, kde=False, vertical=False, color='yellow')
# Matplotlib
ax.hist(x, bin=5, alpha=0.7) # Hist for 2 datasets, input [x,y]
ax.set_ylabel('frequency', fontsize=16)
ax.set_xlabel('something', fontsize=16)
ax.set_title('I am title', fontsize=20)

ax.set_xlim([0.,50.])
ax.set_ylim([0, 100])

plt.show()

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

## Lineplot
# Seaborn
ax = sns.pointplot(x=x, y=y, hue='gender',data=df, marker=['o','x'], linestyle=['-','--'], join=True, 
                   color='#bb3f3f', order=['Dinner','Lunch'], estimator=np.median, capsize=.2)
# Matplotlib
ax = sns.plot(x,y)
ax.set_ylabel('y', fontsize=16)
ax.set_xlabel('x', fontsize=16)
ax.set_title('title', fontsize=20)

ax.set_xlim([0.,50.])
ax.set_ylim([0, 100])

plt.show()

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

## Heatmap (Correlate)
# Seaborn
mean_corr = bcw_mean[[x for x in bcw_mean.columns if x not in 'id']].corr()

# Set the default matplotlib figure size:
plt.rcParams['figure.figsize']=(9,7)

# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(mean_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Plot the heatmap with seaborn.
# Assign the matplotlib axis the function returns. This will let us resize the labels.
ax = sns.heatmap(mean_corr, mask=mask)
# Matplotlib
## It's difficult, use seaborn pls

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

## Barplot
# Seaborn
ax = sns.barplot(x=x, y=y, data=df, hue='gender', order=['Dinner', 'Lunch'], estimator=np.median, color='b')
# Matplotlib
ax.bar(x, y, width=1.5, color='blue')
ax.set_ylabel('y', fontsize=16)
ax.set_xlabel('x', fontsize=16)
ax.set_title('title', fontsize=20)

ax.set_xlim([0.,50.])
ax.set_ylim([0, 100])

plt.show()

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

## Pairplot
# Seaborn
ax = sns.pairplot(df, hue='gender', markers=['x','o'], size=3, vars=['Height', 'Weight'], kind='reg')

# Create customizable Pairplot
grid = sns.PairGrid(subjective)
grid = grid.map_lower(sns.regplot)
grid = grid.map_diag(plt.hist)
grid = grid.map_upper(sns.kdeplot, cmap='Blues', shade=True, shade_lowest=False)

plt.show()

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

## Boxplot
# Seaborn
ax = sns.boxplot(x=x, y=y, data=df, hue='gender', orient='h', fliersize=2)
ax = sns.swarmplot(x=x, y=y, data=df, color='.25')
# Another variation
sns.boxplot(x=x, data=rv_df, color='limegreen')
sns.swarmplot(x=x, data=rv_df, color='orange', linewidth=.3)
plt.show()
# Matplotlib
ax.barplot(data)


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


## Hexagon Plot
from scipy.stats import kendalltau
sns.set(style="ticks")

# fig = plt.figure(figsize=(12,10))
# ax = fig.gca()

x = rv_df.median_rv_price
y = rv_df.mean_rv_price

sns.jointplot(x, y, kind="hex", stat_func=kendalltau, color="#4CB391 ")
plt.show()
#=====================================================================================================





### Modeling
#=====================================================================================================
## Setting up Patsy
import patsy

formula = 'target ~ predictor1 + predictor2 + predictor3 + predictor4 ... + predictor100 - 1'
y, X    = patsy.dmatrices(formula, df=df, return_type='dataframe')

# Since patsy makes y into a 2D array/dataframe with the index, have to change y into a 1D array/List
y = y.values
# or
y = np.ravel(y)
# or
y = df['target']


#-----------------------------------------------------------------------------------------------------


## Normalization, Standardization, Lasso, Ridge, Elastic Net
# Normalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
# fit_transform calculates the pre-scaled mean and standard deviation, can use .mean_ and .scale_
scaled_data = scaler.fit_transform(df)


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


# Standardization
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()
# fit_transform calculates the pre-scaled mean and standard deviation, can use .mean_ and .scale_
scaled_data = scaler.fit_transform(df)


#-----------------------------------------------------------------------------------------------------


## Setting up train-test split
from sklearn.cross_validation import train_test_split

trainX, testX, trainY, testY = train_test_split(X, y, train_size=0.7, stratify=y) # Can also use test_size
print trainX.shape, testX.shape
print trainY.shape, trainY.shape
# trainX and testX should be a data from like (1000,15)
# trainY and testY should be a 1D array or list like (1000,)


#-----------------------------------------------------------------------------------------------------


## GridSearch
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


#-----------------------------------------------------------------------------------------------------


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


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


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


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


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


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


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


#-----------------------------------------------------------------------------------------------------


## Modeling Types
                ########################################################     
                #       Continuous        |        Categorical         #
                ########################################################
                #    Linear Regression    |    k-Nearest Neighbors     #
# Supervised    #                         |    Logistic Regression     #
                #                         |   Support Vector Machine   #
                #                         |       Decision Tree        #
                #                         |      Ensemble Methods      #
                #======================================================#
                #          LDA            |          K-Means           #
                #          PCA            |        Hierarchical        #
# Unsupervised  #                         |          DBSCAN            #
                #                         |                            #
                ########################################################


                #########################################################     
                #       Continuous         |        Categorical         #
                #########################################################
# Supervised    #       Regression         |       Classification       #
                #=======================================================#
# Unsupervised  # Dimensionality Reduction |         Clustering         #
                #########################################################


## Modeling - this is where you decide which model to use
# The parameters found in GridSearch can be used for regression analysis
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


"""
SUPERVISED CONTINUOUS (2.1 Deeper into Loss Functions for graphing)
"""
# Linear Regression
from sklearn.linear_model import LinearRegression

# Assign lm as the LinearRegression function
lm = LinearRegression()
# Fits trainX and trainY with the lm model and assigns to a variable
model = lm.fit(trainX, trainY)
# Returns Predicted Y values
predictions = model.predict(testX)
# Plots your True Y with Predicted Y
plt.scatter(testY, predictions)
# R-squared
score = model.score(testX, testY)


# Stats Models
import statsmodels.api as sm
model = sm.OLS(y, X).fit()
model.params

# Provides you with a full summary of the Linear Regression Model
summary = model.summary()
summary

# Create a features and their coefficients in a dataframe
feature_importance = pd.DataFrame({ 'features':X.columns, 
                                   'coefficients':model.coef_
                                  })

feature_importance.sort_values('coefficients', ascending=False, inplace=True)
feature_importance


#-----------------------------------------------------------------------------------------------------


"""
SUPERVISED CATEGORICAL
"""
# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Assign knn as a KNeighborsClassifier function
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform') # uniform or distance
# Fits trainX and trainY with the knn model and assigns to a variable
model = knn.fit(trainX, trainY)
# ?
predictions = model.predict(testX)
# ?
score = model.score(testX, testY)


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


# Logistic Regression
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
%matplotlib inline

def logistic_regression_calculation(predictors, target, title='Your Prediction'):
    
    ### Test-Train split 70-30
    trainX, testX, trainY, testY = train_test_split(predictors, target, train_size=0.7, stratify=target)
    print 'trainX shape: ', trainX.shape, '\ntestX shape:', testX.shape
    print 'trainY shape: ', trainY.shape, '\ntestY shape:', testY.shape
    
    ### Setup LogisticRegression modeling
    # Create LogisticRegression function cross validated 5 times
    logreg = LogisticRegressionCV(cv=5)
    # Fit the data points into the LogisticRegression model
    model = logreg.fit(trainX, trainY)
    # Predict Probability
    probabilities = model.predict_proba(testX)
    # Score the model
    score = model.score(testX, testY)
    print 'Model Score: ', score
    
    ### Plot the data
    # Creating a blank set of objects to store my confusion matrix metrics here
    FPR = dict()
    TPR = dict()
    ROC_AUC = dict()

    # I am assigning the 1st offsets to my FPR / TPR from the 2nd set of probabiliies from my
    # .predict_proba() predictions
    # This data is what will be plotted once we throw it to our figure
    FPR[1], TPR[1], _ = roc_curve(testY, probabilities[:, 1])
    ROC_AUC[1] = auc(FPR[1], TPR[1])

    # 1. Initialize a blank plot, aspect 11x9
    plt.figure(figsize=[11,9])
    # 2. Plot my false and true rates (returned from roc_curve function)
    plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
    # 3. Plotting a dotted line diagonally, representing the .5
    plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('Receiver operating characteristic for %s' %title, fontsize=18)
    plt.legend(loc="lower right")
    plt.show()

logistic_regression_calculation(X, y, title='over 200k predictions')


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


## Support Vector Machine

from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report

svm = SVC(C=np.logspace(1e-4, 1e1, 20), kernel=['linear','rbf'])

svm.fit(X, Y)

Y_pred = svm.predict(X)

# Baseline Accuracy
y.value_counts() / len(y)

# Average Accuracy Score
lin_model = SVC(kernel='linear')

scores = cross_val_score(lin_model, Xn, y, cv=5)
sm = scores.mean()
ss = scores.std()
print "Average score: {:0.3} +/- {:0.3}".format(sm, ss)

# Classification Report and Confusion Matrix
def print_cm_cr(y_true, y_pred):
    """prints the confusion matrix and the classification report"""
    confusion = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print confusion
    print
    print classification_report(y_true, y_pred)


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


## CARTs (Classification and Regression Trees)
# Classification
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='gini', max_depth=None)

classifier.fit(X, Y)

Y_pred = classifier.predict(X)
"""
Decision trees can give us feature importances.
The higher the number, the more important the predictor is to deciding splits at nodes.
The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature.
"""
feature_importances = classifier.feature_importances_

# Regression
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(criterion='mse', max_depth=5)

regressor.fit(X, Y)

Y_pred = regressor.predict(X)

feature_importances = regressor.feature_importances_

# Create a feature and their importances in a dataframe
feature_importance = pd.DataFrame({ 'feature':X.columns, 
                                   'importance':dctc_best.feature_importances_
                                  })

feature_importance.sort_values('importance', ascending=False, inplace=True)
feature_importance


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


## Latent Dirichlet Allocation (LDA)
from gensim import corpora, models, matutils
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import pandas as pd

doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health."

# compile sample documents into a list
documents = [doc_a, doc_b, doc_c, doc_d, doc_e]

# Fit the documents into a count vectorizer (TFIVectorizer)
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
X.todense()

docs = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())

vocab = {v: k for k, v in vectorizer.vocabulary_.iteritems()}
vocab1 = {i:vocab[i] for i in vocab if i>72}

# remove words that appear only once
frequency = defaultdict(int)

for text in documents:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]
          for text in documents]

# Create gensim dictionary object
dictionary = corpora.Dictionary(texts)

# Create corpus matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# Setup LDA Model and parameters
lda = models.LdaModel(
    matutils.Sparse2Corpus(X, documents_columns=False),
    # or use the corpus object created with the dictionary in the previous frame!
    # corpus, 
    num_topics  =  3,
    passes      =  20,
    id2word     =  vocab1
    # or use the gensim dictionary object!
    # id2word     =  dictionary
)

# Prints out number of topics and the 5 words with highest scores
lda.print_topics(num_topics=3, num_words=5)

# Generates overall score for each topic
lda.get_document_topics(corpus[0])


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


"""
Unsupervised Categorical
"""
## Clustering
# K-Means
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

k = 3
model = KMeans(n_clusters = k)
model.fit(df)

labels = model.labels_
centroids = model.cluster_centers_

print labels, centroids

score = silhouette_score(df, labels, metric='euclidean')
print score

# Scale if required
from sklearn.preprocessing import scale

df = pd.read_csv(csv_file)
df[features] = scale(df[features])

# Run K-Means against our 2 features
k = 7
model = KMeans(n_clusters = k)
model.fit(df[features].values)

# Check our results
# Assign clusters back to our dataframe
df['cluster'] = model.labels_

# Get our centrois
centroids    =  model.cluster_centers_
cc           =  pd.DataFrame(centroids)

# Setup some sweet colors for plotting (for later)
# colors = {'D':'red', 'E':'blue', 'F':'green'}
base_colors  =  ['r', 'g', 'b', 'orange', 'purple', 'gold', 'darkred']
colors       =  [base_colors[centroid] for centroid in model.labels_]

# update x,y of our final clusters to plot later
fig, ax      =  plt.subplots(figsize=(8,8))

# Plot the scatter of our points with calculated centroids
ax.scatter(df[features[0]], df[features[1]], c=colors)
ax.scatter(cc[0], cc[1], c=base_colors, s=100) # cc.index

# And our score
print "Silhouette Score: ", silhouette_score(df[features], df['cluster'], metric='euclidean')


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


"""
Unsupervised Continuous
"""
## Principal Component Anaylsis
# Eigenvalues and Eigenvectors using NumPy
eig_vals, eig_vecs = np.linalg.eig(demo_noage_corr)
print eig_vals
print eig_vecs

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Always Standardize before performing a PCA
ss = StandardScaler()
df_n = ss.fit_transform(df)

pca = PCA(n_components=5) # n_components cannot be more than your labels/columns
pca.fit(df_n)

# Prints out the Eigenvalues
components = pca.components_

# Prints out each PC and its Eigenvalue
print subjective.columns.values

for i, pc in enumerate(['PC1', 'PC2', 'PC3', 'PC4', 'PC5']):
    print pc
    for col, weight in zip(subjective.columns.values, subj_components[i]):
        print col, weight
    print '=======================================================\n'


# Plots the ratio between all Eigenvalue scores
tot = sum(pca.explained_variance_ratio_)
var_exp = [(i / tot)*100 for i in sorted(pca.explained_variance_ratio_, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

component_number = range(len(pca.explained_variance_ratio_))

plt.plot(component_number, cum_var_exp, lw=7, color='blue')
plt.bar(component_number, var_exp, lw=3, width=0.9, color='orange')

plt.axhline(y=0, linewidth=5, color='grey', ls='dashed')
plt.axhline(y=100, linewidth=3, color='grey', ls='dashed')

ax = plt.gca()
ax.set_xlim([0,5])
ax.set_ylim([-5,105])

ax.set_ylabel('cumulative variance explained', fontsize=16)
ax.set_xlabel('component', fontsize=16)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12) 
    
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12) 
    
ax.set_title('component vs cumulative variance explained\n', fontsize=20)

plt.show()


#-----------------------------------------------------------------------------------------------------


## Cross Validation
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics

# Perform 6-fold cross validation
scores = cross_val_score(model, X, y, cv=6) #parameters (model, predictor/dataset, target, folds)
print 'Cross-validated scores:', scores

# Make cross validated predictions
predictions = cross_val_predict(model, X, y, cv=6) #parameters same as cross_val_score
plt.scatter(y, predictions)

# Calculated accuracy
accuracy = metrics.r2_score(y, predictions)
print 'Cross-predicted Accuracy:', accuracy


## Manual Cross Validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

fold = StratifiedKFold(y, n_folds=5)

def my_cross_val_score(model, X, y):
    scores = []
    for train_i, test_i in fold:
        X_test = X.iloc[test_i,:]
        y_test = y[test_i]
        X_train = X.iloc[train_i,:]
        y_train = y[train_i]

        scores.append(lr.fit(X_train, y_train).score(X_test, y_test))
    print scores

my_cross_val_score(lr, X, y)


#-----------------------------------------------------------------------------------------------------


# Bootstrapping (Confidence Interval)
X = sample_data
X_length = len(X)

bootstrap_iterations = 1000

bootstrap_sample_medians = []

for i in range(bootstrap_iterations):
    random_indicies = np.random.choice(range(X_length), size=X_length, replace=True)
    X_random_sample = X[random_indices]
    bootstrapped_median = np.median(X_random_sample)
    bootstrap_sample_medians.append(bootstrapped_median)

average_median = np.mean(bootstrap_sample_medians)
median_5th_pctl = np.percentile(bootstrap_sample_medians, 5)
median_95th_pctl = np.percentile(bootstrap_sample_medians, 95)


#-----------------------------------------------------------------------------------------------------


## Pre-Processing Methods
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





## Pipeline
#=====================================================================================================


# cPickle helps you save files into small files
import cPickle

df = pd.DataFrame({'a':[1,2,3,4], 'b':[1,2,4,5]})

# 'w' - write
f = open('/Users/edwardlee/Desktop/small_df.p', 'w')
cPickle.dump(df, f)
f.close()

# 'r' - read
f = open('/Users/edwardlee/Desktop/small_df.p', 'r')
loaded_list = cPickle.load(f)
f.close()

## Pipeline
# Pipeline is good for chaining steps together so that you do not need to re-run the same codes on
# multiple data points. You can even write your own class and set it as a pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

ss = StandardScaler()
lr = LogisticRegression()

lr_pipe = Pipeline(steps=[['scaler', ss], ['lr_model', lr]])
#=====================================================================================================





## SQL
#=====================================================================================================
from sqlalchemy import create_engine

# Creates an engine/connection to the SQL table
engine = create_engine('postgresql://dsi_student:gastudents@dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com:5432/northwind')
# postgresql - SQL
# dsi_student - username
# gastudents - password
# @...amazonaws.com - server ip
# 5432 - port
# northwind - table/file name

# Saves SQL table into a pandas dataframe
pd.read_sql('SELECT * FROM table', con=engine)

sql = """
SELECT * FROM table
"""
pd.read_sql(sql)

## SQL commands
# Explanation
""""
CREATE TABLE - Create a new table
INSERT INTO - Insert a new row into a table
DELETE - Delete a row
UPDATE - Update the value in a row
SELECT - Select a row/or all(*)
DISTINCT - Return unique values only
FROM - Select a table
JOIN - Join another table (INNER JOIN, LEFT JOIN (default), RIGHT JOIN, FULL JOIN)
ON - Join another table on specific column
WHERE - Conditional format to extract specific criterions
LIKE - Conditional format to search for a specific pattern in a WHERE clause
GROUP BY - Group by a column of rows
HAVING - HAVING clause is used with GROUP BY, works like WHERE clause except for aggregate functions
ORDER BY - Order by a column of rows
LIMIT - Limit the number of entires
"""

# Examples
"""
CREATE TABLE table_name
(
column_name1 data_type(size),
column_name2 data_type(size),
column_name3 data_type(size),
....
);

INSERT INTO table_name (column1,column2,column3,...);

DELETE FROM table_name
WHERE some_column=some_value;

UPDATE table_name
SET column1=value1,column2=value2,...
WHERE some_column=some_value;

SELECT DISTINCT(table1.col1), table2.col1, table1.col2, SUM(table2.col2) FROM table1
JOIN table2
ON table1.col2=table2.col2
WHERE table2.col1 LIKE "s%"
GROUP BY table2.col2
HAVING table1.col1 > 10
ORDER BY table1.col1
LIMIT 5;
"""

# Cartesian Product
"""
SELECT col1, col2, col3, col4, col5, col6, col7, col8, col9 FROM table1, table2, table3
"""
#=====================================================================================================





### Web Scraping
#=====================================================================================================
from scrapy.selector import Selector
from scrapy.http import HtmlResponse

# Example selector with commonly used tags
Selector(text=HTML).xpath('/html/body/ul/li[@class = "alpha"][@id = "id"]/text()').extract()

# Scrap all the links on a website, ex. for www.datatau.com
//td[@class='title'][2]/a/@href

## Get HTML data
import requests

# Get Request
response = requests.get("http://www.datatau.com")
HTML = response.text
# view the first 500 characters of the HTML index document for DataTau
HTML[0:500]

# Contains is a (if 'string' in x) statement
//td[@class='subtext']/span[contains(@id,'score')]/text()

# Looking for the more link
//a[text()="More"]/@href

# Other ways
best1        = Selector(text=HTML).xpath('/html/body/div/p/a[@class="bestof-link"]')
nested_best1 = best1.xpath('./span[@class="bestof-text"]/text()').extract()
print nested_best1

# Through command line
mkdir scrapy_projects # Creates a new directory
scrapy startproject craigslist # Within the directory, pre-generates all the necessary files
"""
Creates pre-generated files from scrapy

craigslist/
    scrapy.cfg
    craigslist/
        __init__.py
        items.py
        pipelines.py
        settings.py
        spiders/
            __init__.py
            ...

scrapy.cfg: the project configuration file
craigslist/: the project’s python module, you’ll later import your code from here.
craigslist/items.py: the project’s items file.
craigslist/pipelines.py: the project’s pipelines file.
craigslist/settings.py: the project’s settings file.
craigslist/spiders/: a directory where you’ll later put your spiders.
"""
scrapy shell http://sfbay.craigslist.org/search/sfc/apa # Any webpage works for the shell
scrapy crawl craigslist -o apts.csv # -o saves file to apts.csv
#=====================================================================================================





### Definitions
#=====================================================================================================


## Correlation (matrix) vs Covariance (matrix)
# Correlation (matrix)
"""

"""
# Covariance (matrix)
"""

"""


#-----------------------------------------------------------------------------------------------------


## Pearson vs Spearman vs Kendall
# Pearson
"""

"""
# Spearman
"""

"""
# Kendall
"""

"""

#-----------------------------------------------------------------------------------------------------


# Machine Learning Models
# What is it?
"""

"""
# Linear Regression
"""

"""
# Logistic Regression
"""

"""
# K-Nearest Neighbors
"""

"""
# Support Vector Machines
"""

"""
# CARTs (Decision Trees)
"""

"""
# Random Forests (Bagging and Boosting)
"""

"""
# LDA
"""

"""
# Clustering k-Means
"""

"""
# Principal Component Analysis (PCA)
"""

"""


#-----------------------------------------------------------------------------------------------------


## Bias vs Variance
# What is it?
"""
Bias - I've accepted that my model will not perfectly model the dataset.
"""
# Linear Regression
"""

"""
# Logistic Regression
"""

"""
# K-Nearest Neighbors
"""

"""
# Support Vector Machines
"""

"""
# CARTs (Decision Trees)
"""

"""
# Random Forests (Bagging and Boosting)
"""

"""
# LDA
"""

"""
# Clustering k-Means
"""

"""
# Principal Component Analysis (PCA)
"""

"""


#-----------------------------------------------------------------------------------------------------


## Confidence Interval vs P-Value vs Bootstrapping
# Confidence Interval
"""

"""
# P-Value
"""

"""
# Bootstrapping
"""

"""
# How does it all come together?
"""

"""


#-----------------------------------------------------------------------------------------------------


## Law of Large Numbers, Central Limit Theorem, t-tests, Bayesian vs Frequentist
# Law of Large Numbers
"""
Performing the same experiment a large number of times and taking the average of the results will 
result in a convergence to the true expected value of the experiment.
"""
# Central Limit Theorem
"""
In probability theory, the central limit theorem (CLT) states that, given certain conditions, 
the arithmetic mean of a sufficiently large number of iterates of independent random variables, 
each with a well-defined (finite) expected value and finite variance, will be approximately
normally distributed, regardless of the underlying distribution.
"""
# T-tests
"""

"""
# Bayesian vs Frequentist
"""
Bayesian - P(true mean|data)
I have collected fixed data which I use to update my inference of the probability, 
which is called my posterior distribution
Thus, there is a distribution of values for the true mean variable with varying probability.

Frequentist - P(data|true mean)
The mean variable is an unknown but fixed, "true" value.
Our data sampled is random, but the true value is fixed across all hypothetical samples.
There is a distribution of possible samples given the true fixed value.
"""


#-----------------------------------------------------------------------------------------------------


## Confusion Matrix(TP, FP, TN, FN), Precision, Recall, Accuracy, Misclassification Rate, Precision-Recall
# Confusion Matrix
"""

"""
# Precision - TP / (TP + FP)
"""
When the model predicts yes, how often is it correct
"""
# Recall - TP / (TP + FN)
"""
How often the model predicts yes, and it is actually yes
"""
# Accuracy - (TP + TN) / n
"""
How often is the classifier model correct
"""
# Misclassification Rate (Error Rate) - (FP + FN) / n
"""
How often is the classifier model wrong
"""
# Precision-Recall
"""

"""


#-----------------------------------------------------------------------------------------------------


## Baseline Accuracy/Model, R^2, ROC Curve, Cross Validation, Support Vectory Machine - Kernel Trick
# Baseline Accuracy/Model
"""

"""
# R^2
"""

"""
# ROC Curve
"""

"""
# Cross Validation
"""

"""
# Support Vector Machine - Kernel Trick
"""

"""


#-----------------------------------------------------------------------------------------------------


## Ordinary Least Squares Regression, Residual Sum of Squares, Total Sum of Squares
# Ordinary Least Squares Regression (OLS)
"""

"""
# Residual Sum of Squares
"""

"""
# Total Sum of Squares
"""

"""


#-----------------------------------------------------------------------------------------------------


## Normalization (Min-Max Scaling) vs Standardization (z-score), Feature Scaling, Loss Function
# Normalizataion
"""

"""
# Standardization
"""

"""
# Feature Scaling
"""

"""
# Loss Function
"""
The loss function is what is being optimized by the process of regression. 
Think of the term "loss function" sort of like the greater the value, 
the more information about your target variable that is "lost" by your model.
"""


#-----------------------------------------------------------------------------------------------------


## APIs, OAuth
# APIs
"""

"""
# OAuth
"""
A secure authorization protocol that deals with the authorizaiton of third party application
to access the user data without exposing their password. For example, many websites have 
facebook/google/etc... login alternatives. If you choose to log into a site using your facebook
account, facebook will ask for a token, or OAuth, that re-directs you to a confirmation page that
this other "app" or "website" is allowed to use your facebook information. Once confirmed, it re-directs
you back to the "app" or "website" and authorizes you with facebook's OAuth/token and then
you may log into that "app" or "website".
"""


#-----------------------------------------------------------------------------------------------------


## SQL - Normalized and Denormalized data, Cartesian Product
# Normalized
"""
Normalized structures have a single table per entity, and use many foreign keys or link tables 
to connect the entities.
"""
# Denormalized
"""
Denormalized have fewer tables and may (for example) place all of the tweets and the
information on users in one table.
"""
# What are the pros and cons?
"""
Normalized tables save the storage space by separating the information. However, if we need to access
information between 2 tables, we will need to the join the tables, which is a much slower process, since
we have to write SQL code to access the information.
Denormalized tables duplicate a lot of information. Without writing any SQL code, you can access all
all information 1 single table.
For example, in our combined tweets/users table, we may store the address of each user, which is normalized. 
Now instead of storing this once per user, we are storing this once per tweet, which is denormalized.
"""
# Cartesian Product
"""
Also referred to as a cross-join, returns all the rows in all the tables listed in the query. 
Each row in the first table is paired with all the rows in the second table. This happens when 
there is no relationship defined between the two tables.
"""


#-----------------------------------------------------------------------------------------------------
## Bagging, Boosting
# Bagging
"""

"""
# Boosting
"""

"""
# Extremely Randomized Trees
"""

"""


#-----------------------------------------------------------------------------------------------------


## NLP (Natural Language Processing)
# Tokenization/Vectorization
"""

"""
# Hashing Vectorizer
"""
Used for Big Data, not necessarily used for smaller/medium datasets
"""
## Preprocessing Techniques
# NLP Bag of Words
"""
Bag of word approaches like the one outlined before completely ignores the structure of a sentence.
Ex.
  There's wood floating in the sea
  Mike's in a sea of trouble with the move
"""
# Segmentation
"""
Segmentation is a technique to identify sentences within a body of text. 
Language is not a continuous uninterrupted stream of words: punctuation serves as a guide to group 
together words that convey meaning when contiguous.
"""
# Normalization
"""
When slightly different versions of a word exist.
For example: LinkedIn sees 6000+ variations of the title "Software Engineer" and 8000+ variations of the word "IBM".
"""
# Stemming
"""
It would be wrong to consider the words "MR." and "mr" to be different features, thus we need a technique 
to normalize words to a common root. This technique is called Stemming.
Other Examples
  Science, Scientist => Scien
  Swimming, Swimmer, Swim => Swim
"""
# TFIDF
"""
More interesting than stop-words is the tf-idf score. This tells us which words are most discriminating between 
documents. Words that occur a lot in one document but doesn't occur in many documents will tell you something 
special about the document.

  - This weight is a statistical measure used to evaluate how important a word is to a document in 
    a collection (aka corpus)
  - The importance increases proportionally to the number of times a word appears in the document 
    but is offset by the frequency of the word in the corpus.
"""
# Porter vs Snowball(Porter2) vs Lancaster
"""
At the very basics of it, the major difference between the porter and lancaster stemming algorithms is 
that the lancaster stemmer is significantly more aggressive than the porter stemmer. The three major 
stemming algorithms in use today are Porter, Snowball(Porter2), and Lancaster (Paice-Husk), with the 
aggressiveness continuum basically following along those same lines. Porter is the least aggressive 
algorithm, with the specifics of each algorithm actually being fairly lengthy and technical. Here is a 
break down for you though:

Porter: Most commonly used stemmer without a doubt, also one of the most gentle stemmers. One of the 
few stemmers that actually has Java support which is a plus, though it is also the most computationally 
intensive of the algorithms(Granted not by a very significant margin). It is also the oldest stemming 
algorithm by a large margin.

Porter2: Nearly universally regarded as an improvement over porter, and for good reason. Porter himself 
in fact admits that it is better than his original algorithm. Slightly faster computation time than 
porter, with a fairly large community around it.

Lancaster: Very aggressive stemming algorithm, sometimes to a fault. With porter and snowball, the stemmed 
representations are usually fairly intuitive to a reader, not so with Lancaster, as many shorter words will 
become totally obfuscated. The fastest algorithm here, and will reduce your working set of words hugely, but 
if you want more distinction, not the tool you would want.

Honestly, I feel that Snowball is usually the way to go. There are certain circumstances in which Lancaster 
will hugely trim down your working set, which can be very useful, however the marginal speed increase over 
snowball in my opinion is not worth the lack of precision. Porter has the most implementations though and 
so is usually the default go-to algorithm, but if you can, use snowball.
"""
# LDA
"""
Topics generated from an LDA model are actually a cluster of word probabilities, not clearly defined labels.
Simplifying word vectors like this, should give you a sense about the intuition of how words vectors relate to topics.
Kind of like KNN but we are deciding, up front, on a preset number of topics.
"""


#=====================================================================================================





## To Add
#=====================================================================================================



































