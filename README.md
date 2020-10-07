# __Sentiment Analysis on Amazon Fine Food Reviews__

A web application built using Flask to predict the sentiment of the food reviews provided by customers and classifying them into `Positive` or `Negative`.

#### Technologies used: Flask, Python, HTML, CSS.
<img src="https://img.shields.io/badge/flask%20-%23000.svg?&style=for-the-badge&logo=flask&logoColor=white"/> <img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/html5%20-%23E34F26.svg?&style=for-the-badge&logo=html5&logoColor=white"/> <img src="https://img.shields.io/badge/css3%20-%231572B6.svg?&style=for-the-badge&logo=css3&logoColor=white"/>

## _Table of Contents_
+ [Dataset](#dataset)
+ [Installation](#installation)
+ [Working](#working)
+ [Results](#results)
<br>

## Dataset
In this project, we used [Amazon Fine Food Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews) dataset. This dataset consists of reviews given by customers spanning over a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories.

![dataset distribution](/output/score.png)

## Installation
Anaconda users:

```python
conda install -c anaconda numpy
conda install -c anaconda pandas
conda install -c conda-forge spacy
conda install -c conda-forge scattertext
conda install -c conda-forge sense2vec
conda install -c anaconda flask
conda install -c conda-forge flask-bootstrap
conda install -c conda-forge textblob
```

Python users:

```python
pip install numpy
pip install pandas
pip install spacy
pip install scattertext
pip install sense2vec
pip install Flask
pip install Flask-Bootstrap
pip install textblob
```

## Working
In this we are going to tackle an interesting natural language processing problem i.e. sentiment or text classification. We will explore textual data using the amazing spaCy library and build a text classification model.

We will extract linguistic features like `Tokenization`, `Part-of-Speech(POS) tagging`, `Dependency parsing`, `Lemmatization`, `Named entity recognition (NER)`, `Sentence Boundary Detection` for building language models later.

Word vectors and similarity -> sense2vec <br>
Text classification model -> SpaCy TextCategorizer

<b>We will treat rating 4 and 5 as positive and rest as negative reviews.</b>

![](/output/score_boolean.png)

------
### 1. Tokenization 
First step in any NLP pipeline is tokenizing text i.e. breaking down paragraphs into sentences and then sentences into words, punctuations and so on. We will load English language model to tokenize our English text. Every language is different and have different rules. `SpaCy` offers 8 different language models.

------
### 2. Part-of-Speech tagging
After tokenization we can parse and tag variety of parts of speech to paragraph text. `SpaCy` uses statistical models in background to predict which tag will go for each word(s) based on the context.

------
### 3. Lemmatization
It is the process of extracting uninflected/base form of the word. Lemma can be like For eg. 
+ Adjectives: best, better → good 
+ Adverbs: worse, worst → badly 
+ Nouns: ducks, children → duck, child 
+ Verbs: standing,stood → stand

------
### 4. Named Entity Recognition (NER)
Named entity is real world object like Person, Organization, etc
  
      PERSON	      People, including fictional.
      NORP	        Nationalities or religious or political groups.
      FAC	          Buildings, airports, highways, bridges, etc.
      ORG	          Companies, agencies, institutions, etc.
      GPE	          Countries, cities, states.
      LOC	          Non-GPE locations, mountain ranges, bodies of water.
      PRODUCT	      Objects, vehicles, foods, etc. (Not services.)
      EVENT	        Named hurricanes, battles, wars, sports events, etc.
      WORK_OF_ART	  Titles of books, songs, etc.
      LAW	          Named documents made into laws.
      LANGUAGE	    Any named language.
      DATE	        Absolute or relative dates or periods.
      TIME	        Times smaller than a day.
      PERCENT	      Percentage, including "%".
      MONEY	        Monetary values, including unit.
      QUANTITY	    Measurements, as of weight or distance.
      ORDINAL	      "first", "second", etc.
      CARDINAL	    Numerals that do not fall under another type

------
### 5. sense2vec
The idea is get something better than `word2vec` model. `sense2vec` is super simple. If the problem is that duck as in waterfowl and duck as in crouch are different concepts, the straight-forward solution is to just have two entries, duckN and duckV. Trask et al (2015) published a nice set of experiments showing that the idea worked well.

------

<br>

## Results
Positive output prediction with score:

![img](/output/positive.png)

Negative output prediction with score:

![img](/output/negative.png)






