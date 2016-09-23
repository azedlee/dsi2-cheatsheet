### Machine Learning
#=====================================================================================================
## Modeling Types
                ########################################################     
                #       Continuous        |        Categorical         #
                ########################################################
                #    Linear Regression    |    k-Nearest Neighbors     #
# Supervised    #                         |    Logistic Regression     #
                #                         |   Support Vector Machine   #
                #                         |       Decision Tree        #
                #                         |      Ensemble Methods      #
                #                         |        Naive Bayes         #
                #======================================================#
                #          PCA            |          K-Means           #
                #          LDA            |       Hierarchical         #
# Unsupervised  #                         |          DBSCAN            #
                #                         |     Sentiment Analysis     #
                #                         |          Optics            #
                ########################################################


                #########################################################     
                #       Continuous         |        Categorical         #
                #########################################################
# Supervised    #       Regression         |       Classification       #
                #=======================================================#
# Unsupervised  # Dimensionality Reduction |         Clustering         #
                #########################################################


#-----------------------------------------------------------------------------------------------------


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


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


"""
REGRESSION (2.1 Deeper into Loss Functions for graphing)
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
CLASSIFICATION
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


"""
CLUSTERING
"""
## K-Means
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

# Prints all the word and the number of counts
[(word, count) for word, count in vectorizer.vocabulary_.items()]

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


## Hierarchical Clustering
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist

# Convert our data to a matrix (aka array) to pass to the clustering algorithm - the matrix 
# makes it easier for our algorithm to compute distance
X = lang.as_matrix(columns=None)

# Ward's Method which seeks to minimize the variance when forming clusters
Z = linkage(X, 'ward')

# Calculate the cophenetic correlation coefficient to see how well our algorithm has measured 
# the distances between the points
c, coph_dists = cophenet(Z, pdist(X))

# Plot a Dendogram
def plot_dendogram(df):
    
    # Data prep
    X = df.as_matrix(columns=None)
    Z = linkage(X, 'ward')
    
    # plotting
    plt.title('Dendrogram')
    plt.xlabel('Index Numbers')
    plt.ylabel('Distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  
        leaf_font_size=8.,
    )
    plt.show()
    
    
plot_dendogram(lang)

# Classify our clusters with max_dist by assigning cluster IDs
max_dist = 200 # pairwise distance
clusters = fcluster(Z, max_dist, criterion='distance')


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


"""
DIMENSIONALITY REDUCTION
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


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


## Sentiment Analysis


## DBSCAN


## pymc3
import pymc3 as pymc3

with pm.Model() as model:

	mean_belief = pm.Normal('mean_belief', mu=20, sd=5)
	std_belief = pm.Uniform('std_belief', lower=0.001, upper=10000)

	data = pm.Normal('data', mu=mean_belief, sd=std_belief, observed=df)

with model:
	trace = pm.sample(10000, njobs=4)

plt.figure(figsize=(7,7))
pm.traceplot(trace)
plt.tight_layout()


#=====================================================================================================










### Cross Validation & Bootstrapping
#=====================================================================================================
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


## Bootstrapping (Confidence Interval)
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
#=====================================================================================================










## cPickle & Pipeline
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


