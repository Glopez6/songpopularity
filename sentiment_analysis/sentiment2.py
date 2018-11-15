import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv("train_set.csv")
df_train, df_test = np.array_split(df, 2)

train_data = []
train_data_labels = []

for row in df_train.itertuples():
	train_data.append(row.lyrics)
	train_data_labels.append(row.mood)

test_data = []
test_data_labels = []

for row in df_test.itertuples():
	test_data.append(row.lyrics)
	test_data_labels.append(row.mood)


text_classifier = Pipeline([('vect', CountVectorizer()),
							('tfidf', TfidfTransformer()),
							('clf', MultinomialNB()),
							])

text_classifier.fit(train_data, train_data_labels)

predicted = text_classifier.predict(test_data)
print np.mean(predicted == test_data_labels)