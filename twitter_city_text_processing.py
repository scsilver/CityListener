from TwitterSearch import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pdb
import neglist
import poslist

stemmer = SnowballStemmer("english")
sw = stopwords.words("english")
tweet_list = []
neg_trainer = neglist.NegList()
pos_trainer = poslist.PosList()

count = 0
while True:
    tso = TwitterSearchOrder()  # create a TwitterSearchOrder object
    # let's define all words we would like to have a look for
    tso.set_keywords(['ISIS'])
    tso.set_language('en')  # we want to see German tweets only
    # and don't give us all those entity information
    tso.set_include_entities(False)

    # it's about time to create a TwitterSearch object with our secret tokens
    ts = TwitterSearch(
        consumer_key='BDZszRdd15DRH1mEu8zTO2KjR',
        consumer_secret='fQ2feMYYTQG4p0OcE9z5qTOHFbGsVOdgSvc1Mt4nTjh7s5OFU7',
        access_token='239691362-TpQRih49vRDOMF4bkCvNkYwWdWr5QopaTjWkJWNn',
        access_token_secret='iXpA0c7UNKkZBewSwUJQDrJOuqx8CGVKNQLMFwN1h2r1Y'
    )

    # this is where the fun actually starts :)
    for tweet in ts.search_tweets_iterable(tso):
        count = count + 1
        tweet_list.append(tweet['text'].encode('ascii', 'ignore').lower())
        if count == 100:
            break
    break
print tweet_list
# split to invidual words and remove stop words
tweet_stop = [word for sentence in tweet_list for word in sentence.split(
    " ") if word not in sw]
neg_stop_trainer = [word for sentence in neg_trainer for word in sentence.split(
    " ") if word not in sw]
pos_stop_trainer = [word for sentence in pos_trainer for word in sentence.split(
    " ") if word not in sw]

# stem words
tweet_list_stop_stem = [stemmer.stem(
    word).encode('ascii') for word in tweet_stop]
neg_stop_stem_trainer = [stemmer.stem(word).encode(
    'ascii') for word in neg_stop_trainer]
pos_stop_stem_trainer = [stemmer.stem(word).encode(
    'ascii') for word in pos_stop_trainer]

vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit(neg_stop_stem_trainer)
bag_of_words = vectorizer.transform(neg_stop_stem_trainer)


bag_of_words = vectorizer.fit(tweet_list_stop_stem)
bag_of_words = vectorizer.transform(tweet_list_stop_stem)
print bag_of_words
print tweet_list_stop_stem
tfidf = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
l = ["Have a nice day", "Have a nice night"]
x = tfidf.fit_transform(tweet_list_stop_stem)
t_idf = tfidf.idf_
t_dictionary = dict(zip(tfidf.get_feature_names(), t_idf))

x = tfidf.fit_transform(neg_stop_stem_trainer)
n_idf = tfidf.idf_
n_dictionary = dict(zip(tfidf.get_feature_names(), n_idf))

x = tfidf.fit_transform(pos_stop_stem_trainer)
p_idf = tfidf.idf_
p_dictionary = dict(zip(tfidf.get_feature_names(), p_idf))

#selector = SelectPercentile(f_classif, percentile=10)
# selectory.fit(x,labels train)
n_shared_items = set(t_dictionary.keys()) & set(n_dictionary.keys())
p_shared_items = set(t_dictionary.keys()) & set(p_dictionary.keys())
delta = len(p_shared_items) - len(n_shared_items)

print delta
print "Positive Shared"
print len(p_shared_items)
print p_shared_items
print "Negative Shared"
print len(n_shared_items)
print n_shared_items
