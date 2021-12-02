import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer

#import dataset
df = pd.read_csv("Restaurant_Reviews.csv")
print(df)

lm = WordNetLemmatizer()

#Data Preprocessing
def text_transformation(df_col):
    corpus = []
    for item in df_col:
        new_item = re.sub('[^a-zA-Z]',' ',str(item))
        new_item = new_item.lower()
        new_item = new_item.split()
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_item))
    return corpus

corpus = text_transformation(df['Review'])
#print(corpus)


#Creating a word cloud
word_cloud = ""
for row in corpus:
    for word in row:
        word_cloud+=" ".join(word)
wordcloud = WordCloud(width = 1000, height = 500,background_color ='white',min_font_size = 10).generate(word_cloud)
plt.figure(figsize = (6, 6))
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer  
cv = CountVectorizer(ngram_range=(1,2)) 
#print(cv)
shpcv = cv.fit_transform(corpus)
#print(shpcv.shape)

# X contains corpus
X = cv.fit_transform(corpus).toarray()
#print(X)  
# y contains class labels
y = df.iloc[:, 1].values
#print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

#Applying SVM 
from sklearn.svm import SVC
SVC_classifier = SVC(kernel = 'linear')
SVC_classifier.fit(X_train, y_train)
y_pred_SVC = SVC_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy score: ", accuracy_score(y_test,y_pred_SVC))

def expression_check(pred):
    if pred == 0:
        print("Negative Review")
    else:
        print("Positive Review")

def sentiment_predictor(s):
    s = text_transformation(s)
    ts = cv.transform(s).toarray()
    pred = SVC_classifier.predict(ts)
    expression_check(pred)

while(True):
    r = input("Enter review: ");
    s = [r]
    sentiment_predictor(s)
    c = input("Press c to continue or q to exit: ");
    if c == 'q':
        break
