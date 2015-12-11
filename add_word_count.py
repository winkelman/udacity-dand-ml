#!/usr/bin/python
# -*- coding: utf-8 -*-


## HERE WE ADD IMPORTANT WORD COUNTS AS FEATURES FOR ALL PEOPLE

## PASS IN DICTIONARY WITH THE 'WORDS' FEATURE CONTAINING A STRING OF ALL WORDS
## AND A LIST OF IMPORTANT WORDS, RETURN DICTIONARY WITH NEW FEATURE COUNTS
def add_count(data_dict, id_words):
	
	for person in data_dict:
		text = data_dict[person]['words']
		for word in id_words:
			count = text.count(word)
			data_dict[person][word] = count
			
	return data_dict