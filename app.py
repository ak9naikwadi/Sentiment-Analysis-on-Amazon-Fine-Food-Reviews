
from flask import Flask, render_template,request,url_for
from flask_bootstrap import Bootstrap 

from textblob import TextBlob,Word 
import random 
import time

app = Flask(__name__)
Bootstrap(app)

import joblib
NLP_review_model = open('NLP_review_model.pkl','rb')
nlp = joblib.load(NLP_review_model)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/analyse',methods=['POST'])
def analyse():
	start = time.time()
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		doc = nlp(rawtext)
		conf = doc.cats['POSITIVE']
		if conf > 0.5:
			blob_sentiment = "POSITIVE"
		else:
			blob_sentiment = "NEGATIVE"
		blob_subjectivity = conf
		end = time.time()
		final_time = end-start

	return render_template('index.html',received_text = rawtext,blob_sentiment=blob_sentiment,blob_subjectivity=blob_subjectivity,final_time=final_time)


if __name__ == '__main__':
	app.run(debug=True)