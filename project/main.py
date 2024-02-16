import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline #comment this line out if not using Jupyter notebook

dataset = pd.read_csv(r"C:\Users\ismyk\OneDrive\Рабочий стол\project\emails.csv")
dataset.head(10)

print("Data Visualization")

print(dataset.shape)

print(dataset.spam.value_counts())

sns.countplot(x='spam', data=dataset)

# print("Data Preprocessing")


messages = dataset["text"].tolist()
output_labels = dataset["spam"].values

import re

processed_messages = []

for message in messages:
    message = re.sub(r'\W', ' ', message)
    message = re.sub(r'\s+[a-zA-Z]\s+', ' ', message)
    message = re.sub(r'\^[a-zA-Z]\s+', ' ', message)
    message = re.sub(r'\s+', ' ', message, flags=re.I)
    message = re.sub(r'^b\s+', '', message)
    processed_messages.append(message)

#print("Converting text to numbers")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(processed_messages, output_labels, test_size=0.2, random_state=0)

# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.75, stop_words=stopwords.words('english'))
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

print("Training and Evaluating the Spam Classifier Model")


from sklearn.ensemble import RandomForestClassifier
spam_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
spam_classifier.fit(X_train, y_train)

y_pred = spam_classifier .predict(X_test)

from sklearn.metrics import  accuracy_score
print(accuracy_score(y_test, y_pred))
#print(sns.countplot(x='spam', data=dataset))