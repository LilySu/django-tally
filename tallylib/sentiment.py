# tallylib/sentiment.py

import re
import nltk
import json
import numpy as np
import pandas as pd
from nltk.corpus import wordnet 
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

from tallylib.sql import getLatestReviews


def yelpReviewSentiment(business_id):
	data = getLatestReviews(business_id, limit=200)
	if len(data)==0:
		return {}
	df = pd.DataFrame(data, columns=['date', 'text', 'stars'])
	del data
	def tokenizer(doc):
	 return [token for token in simple_preprocess(doc) 
			 if token not in STOPWORDS]


	def related_to_food(doc):
		word_similarity_list = []
		for review_word in doc:
			syns = wordnet.synsets(review_word) 
			if len(syns) > 0:
				w1 = wordnet.synset(syns[0].name()) # n here denotes the tag noun
				w2 = wordnet.synset('food.n.01') 
				word_similarity_score = w1.wup_similarity(w2)
				if word_similarity_score !=None and word_similarity_score > 0.5:
					word_similarity_list.append(review_word)
		return word_similarity_list


	def related_to_service(doc):
		word_similarity_list = []
		for review_word in doc:
			syns = wordnet.synsets(review_word) 
			if len(syns) > 0:
				w1 = wordnet.synset(syns[0].name()) # n here denotes the tag noun
				w2 = wordnet.synset('service.n.01') 
				word_similarity_score = w1.wup_similarity(w2)
				if word_similarity_score !=None and word_similarity_score > 0.5:
					word_similarity_list.append(review_word)
		return word_similarity_list


	def related_to_speed(doc):
		word_similarity_list = []
		for review_word in doc:
			syns = wordnet.synsets(review_word) 
			if len(syns) > 0:
				w1 = wordnet.synset(syns[0].name()) # n here denotes the tag noun
				w2 = wordnet.synset('speed.n.01') 
				word_similarity_score = w1.wup_similarity(w2)
				if word_similarity_score !=None and word_similarity_score > 0.5:
					word_similarity_list.append(review_word)
		return word_similarity_list


	def related_to_price(doc):
		word_similarity_list = []
		for review_word in doc:
			syns = wordnet.synsets(review_word) 
			if len(syns) > 0:
				w1 = wordnet.synset(syns[0].name()) # n here denotes the tag noun
				w2 = wordnet.synset('price.n.01') 
				word_similarity_score = w1.wup_similarity(w2)
				if word_similarity_score !=None and word_similarity_score > 0.5:
					word_similarity_list.append(review_word)
		return word_similarity_list


	def related_to_ambience(doc):
		word_similarity_list = []
		for review_word in doc:
			syns = wordnet.synsets(review_word) 
			if len(syns) > 0:
				w1 = wordnet.synset(syns[0].name()) # n here denotes the tag noun
				w2 = wordnet.synset('ambience.n.01') 
				word_similarity_score = w1.wup_similarity(w2)
				if word_similarity_score !=None and word_similarity_score > 0.5:
					word_similarity_list.append(review_word)
		return word_similarity_list


	def related_to_experience(doc):
		word_similarity_list = []
		for review_word in doc:
			syns = wordnet.synsets(review_word) 
			if len(syns) > 0:
				w1 = wordnet.synset(syns[0].name()) # n here denotes the tag noun
				w2 = wordnet.synset('experience.n.01') 
				word_similarity_score = w1.wup_similarity(w2)
				if word_similarity_score !=None and word_similarity_score > 0.5:
					word_similarity_list.append(review_word)
		return word_similarity_list

	def extract_subject_related_words():
		df['text'] = df['text'].apply(lambda x:" ".join(re.findall("[a-zA-Z]+", x)))
		df['cleaned'] = df['text'].apply(tokenizer)
		df['words_related_to_food'] = df['cleaned'].apply(related_to_food)
		df['words_related_to_service'] = df['cleaned'].apply(related_to_service)
		df['words_related_to_speed'] = df['cleaned'].apply(related_to_speed)
		df['words_related_to_price'] = df['cleaned'].apply(related_to_price)
		df['words_related_to_ambience'] = df['cleaned'].apply(related_to_ambience)
		df['words_related_to_experience'] = df['cleaned'].apply(related_to_experience)


	extract_subject_related_words()
	food_review_list = df[df['words_related_to_food'].map(len) > 1]['text'].tolist()
	service_review_list = df[df['words_related_to_service'].map(len) > 1]['text'].tolist()
	speed_review_list = df[df['words_related_to_speed'].map(len) > 1]['text'].tolist()
	price_review_list = df[df['words_related_to_price'].map(len) > 1]['text'].tolist()
	ambience_review_list = df[df['words_related_to_ambience'].map(len) > 1]['text'].tolist()
	experience_review_list = df[df['words_related_to_experience'].map(len) > 1]['text'].tolist()

	del df

	def sentiment_score(sentence):
		# Create a SentimentIntensityAnalyzer object. 
		sid_obj = SentimentIntensityAnalyzer()

		# polarity_scores method of SentimentIntensityAnalyzer oject gives a sentiment dictionary. which contains pos, neg, neu, and compound scores. 
		sentiment_dict = sid_obj.polarity_scores(sentence)

		return sentiment_dict

	def get_sentiment(review_list):
		all_sentiments = []
		compounds = []

		if len(review_list) > 0:
			for review in review_list:
				score = sentiment_score(review)
				all_sentiments.append(score)

		if len(all_sentiments) > 0:
			for sentiment_dict in all_sentiments:
				compound = sentiment_dict['compound']
				compounds.append(compound)

		if len(compounds) > 0:
			avg_sentiment = sum(compounds) / len(compounds)
		
		else:
			avg_sentiment = None

		return avg_sentiment


	def get_scores():
		if len(food_review_list) > 0:
			food_sentiment_score = round((get_sentiment(food_review_list))*150)
		else: 
			food_sentiment_score = 75
		if len(service_review_list) > 0:
			service_sentiment_score = round((get_sentiment(service_review_list))*150)
		else: 
			service_sentiment_score = 75
		if len(speed_review_list) > 0:
			speed_sentiment_score = round((get_sentiment(speed_review_list))*150)
		else: 
			speed_sentiment_score = 75
		if len(price_review_list) > 0:
			price_sentiment_score = round((get_sentiment(price_review_list))*150)
		else: 
			price_sentiment_score = 75
		if len(ambience_review_list) > 0:
			ambience_sentiment_score = round((get_sentiment(ambience_review_list))*150)
		else: 
			ambience_sentiment_score = 75
		if len(experience_review_list) > 0:
			experience_sentiment_score = round((get_sentiment(experience_review_list))*150)
		else: 
			experience_sentiment_score = 75
		return food_sentiment_score, service_sentiment_score, speed_sentiment_score, 
		price_sentiment_score, ambience_sentiment_score, experience_sentiment_score

	food_sentiment_score, service_sentiment_score, speed_sentiment_score, price_sentiment_score, ambience_sentiment_score, experience_sentiment_score = get_scores()
	
	del [food_review_list, service_review_list, speed_review_list, price_review_list, ambience_review_list, experience_review_list]
	result = json.dumps([
        { 'subject': 'Food', 'data1': food_sentiment_score, 'data2': 0, 'maxValue': 150 },
        { 'subject': 'Service', 'data1': service_sentiment_score, 'data2': 0, 'maxValue': 150 },
        { 'subject': 'Speed', 'data1': speed_sentiment_score, 'data2': 0, 'maxValue': 150 },
        { 'subject': 'Price', 'data1': price_sentiment_score, 'data2': 0, 'maxValue': 150 },
        { 'subject': 'Ambience', 'data1': ambience_sentiment_score, 'data2': 0, 'maxValue': 150},
        { 'subject': 'Experience', 'data1': experience_sentiment_score, 'data2': 0, 'maxValue': 150}
	])

	del [food_sentiment_score, service_sentiment_score, speed_sentiment_score, price_sentiment_score, ambience_sentiment_score, experience_sentiment_score]

	return result
