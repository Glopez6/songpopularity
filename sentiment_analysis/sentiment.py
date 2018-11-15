import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("train_set.csv")
log_model = LogisticRegression()

data = []
data_labels = []

for row in df.itertuples():
	lyrics = row.lyrics
	#print lyrics
	mood = row.mood
	#print mood
	data.append(lyrics)
	data_labels.append(mood)

vectorizer = CountVectorizer(
	analyzer = 'word',
	lowercase = False,
)

features = vectorizer.fit_transform(
	data
)

features_nd = features.toarray()

X_train, X_test, y_train, y_test = train_test_split(
	features_nd,
	data_labels,
	train_size=0.80,
	random_state=1234,
)

log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print(accuracy_score(y_test, y_pred))
