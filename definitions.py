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
# Hierarchical Clustering
"""

"""
# Sentiment Analysis
"""

"""
# DBSCAN (Density-based spatial clustering of applications with noise)
"""
Similar to K-Nearest Neighbors, DBSCAN picks neighbors that are close by based on the user's
parameters of eps and min_sample. EPS is the max euclidean distance between points and
min_sample is the minimum number of neighbors for a point to be defined as a core sample.
This is the preferred method of clustering because outliers are not forced into clusters,
rather labelled as outliers.
"""


#-----------------------------------------------------------------------------------------------------


## Bias vs Variance
# What is it?
"""
Bias - I've accepted that my model will not perfectly model the dataset.
Variance - I will try to account for all the data points in my current "sample/training" set.
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
# Z-Test
"""
Based on z-score, is the probability higher or lower than the null hypothesis p-value.
Two-tailed can be shifted to left or right, testing whether the hypothesis is not fair.
"""
# T-tests
"""

"""
# Bayesian vs Frequentist
"""
Bayesian - P(true mean|data)
I have collected fixed data which I use to update my inference of the probability, 
which is called my posterior distribution.
The data informs us about the distribution, and as we receive more data, our view of 
the distribution can be updated, further confirming or denying our previous beliefs
(but never in certainty).

Frequentist - P(data|true mean)
The "true" distribution is fixed (and not known).
Our data sampled is random, but the true value is fixed across all hypothetical samples.
There is a distribution of possible samples given the true fixed value.

Frequentist believes that there is a true distribution somewhere out there and the data is 
just a possible sample of it. Bayesian is the opposite. Bayesian believes that the data informs
us about the true distribution and the more data we can get, the more we can update the
distribution. However, we will never know the true distribution.
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
this other app or website is allowed to use your facebook information. Once confirmed, it re-directs
you back to the app or website and authorizes you with facebooks OAuth/token and then
you may log into that app or website.
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
  Theres wood floating in the sea
  Mikes in a sea of trouble with the move
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


#-----------------------------------------------------------------------------------------------------


# Bayes
"""

"""
# pymc3
"""

"""
# Markov Chain Monte Carlo (MCMC)
"""

"""
# Bayes Regression
"""

"""


#-----------------------------------------------------------------------------------------------------

# Big Data
"""
3 Vs: Volume (large amounts of data), Variety (different types of structured, unstructured and multi-structured data),
Velocity (needs to be analyzed quickly).
Daves 4th V: Value (assess the value of the data, understanding the underpinnings of cost vs benefit)

Parallelism: The foundation of Big Data computing - the idea of using multiple computers to compute and solve a problem.
			This allows many resources to be used in parallel. Another way to explain parallelism is divide and conquer,
			which is to break down a task into independent subtasks.

Map Reduce: Invented and publicized by Google in 2004. Map Reduce is a 2-phase Divide and Conquer. The first mapper phase
			splits the data into chunks and the same computation is performed on each chunk. The reducer phase aggregates
			the data back together to produce a final result.
			
			Map-reduce uses a functional programming paradigm. The data processing primitives are mappers and reducers.
			mappers - filter & transform data
			reducers - aggregate results

			INPUT DATA -> SPLIT -> MAP -> COMBINE -> SHUFFLE & SORT -> REDUCE -> OUTPUT DATA
			
			SPLIT 			- split into chunks of data
			MAP 			- tokenization
			COMBINE 		- local group by after map phase
			SHUFFLE & SORT  - organize the combined data evenly and sort them
			REDUCE 			- aggregate the results/group by

			**Once shuffle is 70% done, it will start reducer phase**
"""
# Hadoop
"""
Hadoop is a data processing framework and a distributed file system (HDFS - Hadoop Distributed File System). Hadoop stores
very large amounts of data in clusters/buckets, but they are not databases. You cannot write queries to get specific
information from the data. Thats why it needs a data processing framework called MapReduce to run analysis on it.
"""
# Hive
"""
Hive is a data warehouse infrastructure built on top of Hadoop for data summarization, query, and analysis. 
Hive gives an SQL-like interface to query data stored in Hadoop. Although the commands are not exactly the same as SQL
databases, they are very similar. Knowing SQL commands can get you quickly acquianted with Hive.
"""
# Spark
"""
Does SQL, MapReduce, Machine Learning, Graphing through Hadoop.
The 2 pillars on which Spark is based are RDDs (Resilient Distributed Datasets) and DAGs (Directed Acyclic Graphs), which
are dataframes and decision tree style MapReduce.
1. DAG (MapReduce) - Automatically counts the number to clusters for you. Map() -> Sort/Shuffle -> Reduce()/Aggregation
   Spark also uses a decision tree style MapReduce, which does parallel computation.
   Spark uses RAM instead of harddrives, which makes the job 10-100x faster.

2. Features that are developed and released in Spark, are first available through the Scala implementation of Spark libraries.
   **Learn Scala aside from Python and R as a 3rd language**

3. Spark provides 2 forms of shared variables
   - Broadcast variables: they reference read-only data that needs to be available on all nodes
   - Accumulators: they can be used to program reductions in an imperative style

4. Spark provides two types of operations:
   - Transformations: these are lazy operations that only return a result upon collect
   - Actions: these are non-lazy operations that immediately return a result

   Using lazy operations, we can build a computation graph that only gets executed when we collect the result.
   This allows Spark to optimize the requested calculation by optimizing the underlying DAG of operations.

"""


#=====================================================================================================



### Interview Questions
#=====================================================================================================



# http://www.kdnuggets.com/2016/02/21-data-science-interview-questions-answers.html



#=====================================================================================================