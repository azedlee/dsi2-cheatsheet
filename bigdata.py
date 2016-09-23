# Used through a terminal
cat <input-file> | python mapper.py | sort -k1,1 | python reducer.py | sort -k1,1 > results.txt
# cat reads the file
# | chains to the next action, > pipes to file
# sort by descending, first column

# mapper.py
import sys

# get text from standard input
for line in sys.stdin:
    line = line.strip()
    words = line.split()
    for word in words:
        print '%s\t%s' % (word, 1)

# reducer.py
from operator import itemgetter
import sys

current_word = None
current_count = 0
word = None

# input comes from STDIN
for line in sys.stdin:
    line = line.strip()
    word, count = line.split('\t', 1)
    
    # try to count, if error continue
    try:
        count = int(count)
    except ValueError:
        continue

    # this IF-switch only works because Hadoop sorts map output
    # by key (here: word) before it is passed to the reducer
    if current_word == word:
        current_count += count
    else:
        if current_word:
            print '%s\t%s' % (current_word, current_count)
        current_count = count
        current_word = word

# do not forget to output the last word if needed!
if current_word == word:
    print '%s\t%s' % (current_word, current_count)

#---------------------------------------------------------------------------------------------------------------


# Use VirtualBox
# Terminal
vagrant up
vagrant ssh # start virtual machine
bigdata_start.sh # start Hive, Spark, etc..
pyspark # start spark
exit # exit out of virtual machine
vagrant half # stop virtual machine
http://127.0.0.1:18888/ # access to jupyter notebook in virtual machine

http://10.211.55.101:8088/cluster # opens hadoop
hadoop fs -ls / # same as cmd line on local machine


#---------------------------------------------------------------------------------------------------------------


## Spark
vagrant up
vagrant ssh
spark_local_start.sh

# Spark into a DataFrame
from pyspark.sql.types import *

# trips.first().split(',')
fields = [StructField(field_name, StringType(), True) for field_name in trips.first().split(',')]
schema = StructType(fields)
tripsRDD = sqlContext.createDataFrame(trips.map(lambda line: line.split(",")), schema)

# Filter now!
tripsRDD.filter(tripsRDD['End Terminal'] == 70).collect()


#---------------------------------------------------------------------------------------------------------------

#######################################################
### Spark - Feature Preparation (RDDs, Spark SQL)	###
###		   	Model Training (MLlib)					###
###		   	Model Evaluation (MLlib)				###
###		   	Production Use (model.predict())		###
#######################################################

## Feature Preparation ##
df = spark.read.csv("/../hello.csv", header=True, mode="DROPMALFORMED")


## rdd.printSchema() ##
df.select("Store Number").describe().show()


## show df as raw data ##
df.take(5)


## show df similar to pandas, but not as good ##
df.select(df.columns).show(5)
df.select("Date", "Store Number", "Category Name", "Bottles Sold", "Sale (Dollars)").show(5)


## Changing the dtype of every column ##
from pyspark.sql.types import StringType, IntegerType, DoubleType
from pyspark.sql.functions import udf, regexp_replace

# stripDollarSigns = udf(lambda s: s.replace("$", ""), DoubleType())

df = df \
.withColumn("Store Number",          df["Store Number"].cast("integer")) \
.withColumn("Sale (Dollars)",        regexp_replace("Sale (Dollars)", "\\$", "").cast("double")) \
.withColumn("Zip Code",              df["Zip Code"].cast("integer")) \
.withColumn("County Number",         df["County Number"].cast("integer")) \
.withColumn("Vendor Number",         df["Vendor Number"].cast("integer")) \
.withColumn("Item Number",           df["Item Number"].cast("integer")) \
.withColumn("Bottle Volume (ml)",    df["Bottle Volume (ml)"].cast("integer")) \
.withColumn("State Bottle Cost",     regexp_replace("State Bottle Cost", "\\$", "")) \
.withColumn("State Bottle Retail",   regexp_replace("State Bottle Retail", "\\$", "")) \
.withColumn("Bottles Sold",          df["Bottles Sold"].cast("integer")) \
.withColumn("Volume Sold (Liters)",  df["Volume Sold (Liters)"].cast("double")) \
.withColumn("Volume Sold (Gallons)", df["Volume Sold (Gallons)"].cast("double"))

df.printSchema()
df.show(5)


## Similar to Pandas .describe() ##
df.select(df.columns).describe().show()
df.select(["Zip Code", "Bottle Volume (ml)", "Bottles Sold", "Sale (Dollars)", "Volume Sold (Liters)"]).describe().show()


## Linear Regression Modeling ##
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel

features = ["Bottles Sold", "Sale (Dollars)", "Bottle Volume (ml)"]
response = "Volume Sold (Liters)"

X = df.rdd.map( 
    lambda row: LabeledPoint(row[response], [row[feature] for feature in features])
)


## Train-Test split ##
# Split the data into training and test sets (30% held out for testing)
trainingData, testData = X.randomSplit([0.7, 0.3])


## Train on LinearRegression ##
linearModel = LinearRegressionWithSGD.train(trainingData, iterations=100, step=0.000001)


## Examining Coefficients ##
zip(features, linearModel.weights.array)


## Regression Methods ##
from pyspark.mllib.evaluation import RegressionMetrics

prediObserRDD = testData.map(lambda row: (float(linearModel.predict(row.features)), row.label)).cache()
metrics = RegressionMetrics(prediObserRDD)

print """
                R2:  %.6f
Explained Variance:  %.6f
               MSE:  %.6f
              RMSE:  %.6f
""" % (metrics.r2, metrics.explainedVariance, metrics.meanSquaredError, metrics.rootMeanSquaredError)


#########################################################################
### Another way to load up the clean data is to pre-create the schema ###
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType

schema = StructType([
    StructField("PassengerId", IntegerType()),
    StructField("Survived",    IntegerType()),
    StructField("Pclass",      IntegerType()),
    StructField("Name",        StringType()),
    StructField("Sex",         StringType()),
    StructField("Age",         DoubleType()),
    StructField("SibSp",       IntegerType()),
    StructField("Parch",       IntegerType()),
    StructField("Fare",        DoubleType()),
    StructField("Embarked",    StringType()) 
])

df = spark.read.csv("../../../hello.csv", header=True, mode="DROPMALFORMED", schema=schema)

# Print schema, and then show the first 5 records in printed format
df.printSchema()
df.show(5)
#########################################################################


## Logistic Regression Modeling ##
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel

logisticModel = LogisticRegressionWithLBFGS.train(trainingData)


## Examining Coefficients ##
zip(features, logisticModel.weights.array)


## Regression Metrics ##
prediObserRDD = testData.map(lambda row: (float(logisticModel.predict(row.features)), row.label)).cache()
metrics = RegressionMetrics(prediObserRDD)

print """
                R2:  %.6f
Explained Variance:  %.6f
               MSE:  %.6f
              RMSE:  %.6f
""" % (metrics.r2, metrics.explainedVariance, metrics.meanSquaredError, metrics.rootMeanSquaredError)


## Classification Metrics ##
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Overall accuracy
def testError(lap):
    return lap.filter(lambda (v, p): v != p).count() / float(testData.count())
    
accuracy = testError(prediObserRDD)

print "Test Accuracy = %s" % accuracy

# Instantiate metrics object
metrics = BinaryClassificationMetrics(prediObserRDD)

# Area under precision-recall curve
print "Area under PR = %s" % metrics.areaUnderPR

# Area under ROC curve
print "Area under ROC = %s" % metrics.areaUnderROC


## Multi-class Metrics (Multinomial Response) ##
from pyspark.mllib.evaluation import MulticlassMetrics

metrics = MulticlassMetrics(prediObserRDD)

precision = metrics.precision()
recall = metrics.recall()
f1Score = metrics.fMeasure()

print "Summary Stats" 
print "--------------------"
print "Accuracy  = %s" % metrics.accuracy
print "Precision = %s" % precision 
print "Recall    = %s" % recall 
print "F1 Score  = %s" % f1Score


## Random Forests with PySpark (Not fully implemented) ##
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())

print('Test Error = ' + str(testErr))
print('Learned classification forest model:')
print(model.toDebugString())


##########################################################
##########################################################
##########################################################
##########################################################
###################                   ####################
###################   df.toPandas()   ####################
###################                   ####################
##########################################################
##########################################################
##########################################################
##########################################################

# Other good features to know
# Pipelines
# ParamGridSearch
# Model Loading/Saving


#---------------------------------------------------------------------------------------------------------------


## AWS (Create EC2 Instance)

# Create an EC2 instance (pick region!)
# If free, please use the Ubuntu server
# Create a key pair or you cannot connect to your server

# Secure your server by "Create Security Group"
# Inbound, create Custom TCP Rule, TCP, 8888, Anywhere

# Secure way to connect to AWS ubuntu server with ssh file (must create security group)
ssh -i ~/.ssh/ssh_file.pem -L 18888:127.0.0.1:8888 ubuntu@'Public DNS'

# Change permission of your ssh file
chmod 600 ~/.ssh/ssh_file.pem

# Setting up AWS server on ubuntu
1  sudo apt-get install python-2.7
2  sudo apt-get install anaconda
3  sudo apt-get install pip
4  sudo apt-get update
5  sudo apt-get install python-pip
6  history
7  sudo pip install jupyter pandas
8  pip install --upgrade pip
9  sudo pip install --upgrade pip
10  sudo pip install jupyter
11  sudo pip install pandas
12  history

jupyter notebook --ip='*'

# q to quit, then y for yes in next prompt, *DO NOT CTRL-C* 
# then type the ip below into browser
localhost:18888/tree

# Create Image - creates a personal instance for future use if you need to create a new instance/server


#---------------------------------------------------------------------------------------------------------------