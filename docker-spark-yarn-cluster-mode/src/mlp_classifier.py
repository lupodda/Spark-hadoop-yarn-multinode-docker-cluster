#!/usr/bin/env python
# coding: utf-8

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from sklearn.metrics import classification_report, confusion_matrix


sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

df = spark.read.csv("hdfs://testbed-master:9000/log2.csv", header = True, inferSchema = True)
df.printSchema()
label_indexer = StringIndexer().setInputCol("Action").setOutputCol("label")
label_indexer_model = label_indexer.fit(df)
li_df = label_indexer_model.transform(df)

assembler = VectorAssembler(inputCols=['Source Port', 'Destination Port', 'NAT Source Port', 'NAT Destination Port', 'Bytes', 'B>output = assembler.transform(li_df)

model_df = output.select('features', 'label')

print('classes distribution:')
model_df.groupBy('label').count().show()


df_train, df_test = model_df.randomSplit([0.7,0.3], seed = 42)


layers = [11, 7, 5, 4]

# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=42)

# train the model
model = trainer.fit(df_train)

# select example rows to display.
predictions = model.transform(df_test)


y_true = predictions.select(['label']).collect()
y_pred = predictions.select(['prediction']).collect()

print('classification report:')
print(classification_report(y_true, y_pred))
print('confusion matrix')
print(confusion_matrix(y_true, y_pred))
