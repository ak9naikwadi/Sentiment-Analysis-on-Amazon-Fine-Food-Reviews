import pandas as pd
import numpy as np
import spacy
from spacy import displacy
from spacy.util import minibatch, compounding
import explacy
import scattertext as st
import sense2vec
from sense2vec import Sense2VecComponent
import time

food_reviews_df = pd.read_csv('C:/Users/aksha/Desktop/AmazonFFR/amazon-fine-food-reviews/Reviews.csv')
food_reviews_df = food_reviews_df[['Text','Score']].dropna()
food_reviews_df.Score[food_reviews_df.Score<=3]=0
food_reviews_df.Score[food_reviews_df.Score>=4]=1

train_pos_df = food_reviews_df[food_reviews_df.Score==1][:50000]
train_neg_df = food_reviews_df[food_reviews_df.Score==0][:50000]
train_df = train_pos_df.append(train_neg_df)

nlp = spacy.load("en_core_web_md",disable_pipes=["tagger","ner"])
train_df['parsed'] = train_df.Text[49500:50500].apply(nlp)
train_df['tuples'] = train_df.apply(lambda row: (row['Text'],row['Score']), axis=1)
train = train_df['tuples'].tolist()

def load_data(limit=0, split=0.8):
    train_data = train
    np.random.shuffle(train_data)
    train_data = train_data[-limit:]
    texts, labels = zip(*train_data)
    cats = [{'POSITIVE': bool(y)} for y in labels]
    split = int(len(train_data) * split)
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])

def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}

#("Number of texts to train from","t" , int)
n_texts=30000
#You can increase texts count if you have more computational power.
#("Number of training iterations", "n", int))
n_iter=10

nlp = spacy.load('en_core_web_sm')  # create english Language class

if 'textcat' not in nlp.pipe_names:
    textcat = nlp.create_pipe('textcat')
    nlp.add_pipe(textcat, last=True)
# otherwise, get it, so we can add labels to it
else:
    textcat = nlp.get_pipe('textcat')

# add label to text classifier
textcat.add_label('POSITIVE')

# load the dataset
print("Loading food reviews data...")
(train_texts, train_cats), (dev_texts, dev_cats) = load_data(limit=n_texts)
print("Using {} examples ({} training, {} evaluation)".format(n_texts, len(train_texts), len(dev_texts)))
train_data = list(zip(train_texts,[{'cats': cats} for cats in train_cats]))


#training
# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
with nlp.disable_pipes(*other_pipes):  # only train textcat
    optimizer = nlp.begin_training()
    print("Training the model...")
    print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
    for i in range(n_iter):
        t1 = time.time()
        losses = {}
        # batch up the examples using spaCy's minibatch
        batches = minibatch(train_data, size=compounding(4., 32., 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.2,losses=losses)
        with textcat.model.use_params(optimizer.averages):
            # evaluate on the dev data split off in load_data()
            scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
        t2 = time.time()
        time_taken = t2 - t1
        print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4:.3f}'.format(losses['textcat'], scores['textcat_p'],scores['textcat_r'], scores['textcat_f'],time_taken))

import joblib
joblib.dump(nlp, 'NLP_review_model.pkl')