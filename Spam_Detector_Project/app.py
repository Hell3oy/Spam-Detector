# Spam Detector 

# Importing Libarires
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

@app.route('/')
def home() :
	return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict() :
	csv_data = pd.read_csv("spam.csv", encoding = "latin-1") # Reading Data
	csv_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True) # Removing Useless Coloums
    
	# Features and Labels
	csv_data['label'] = csv_data['v1'].map({'ham': 0, 'spam': 1})
	features = csv_data['v2']
	labels = csv_data['label']
	
	# Extract Feature With CountVectorizer
	cv = CountVectorizer()
	features = cv.fit_transform(features) # Fit the Data
    
	# Splitting Data With sklearn.model_selection 
	from sklearn.model_selection import train_test_split
	features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.10, random_state=42)
	
	# Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(features_train, labels_train)
	clf.score(features_test, labels_test) # Model Score

	if request.method == 'POST' :
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html', prediction = my_prediction)

if __name__ == '__main__' :
	app.run(debug = True)