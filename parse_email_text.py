#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string
import re

digit = re.compile('\d+') ## to exclude id words contaning numbers

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top,
	stem all words ignoring stop-words
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        """

    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")

    if len(content) > 1:
	    
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        ### removing stop-words and words containing digits, then stemming each word
        words_list = text_string.split()
        stemmer = SnowballStemmer("english")
        stop_words = stopwords.words("english")
        stemmed_words_list = [stemmer.stem(word) for word in words_list if ((word not in stop_words) and not bool(digit.search(word)))]
        words = ' '.join(stemmed_words_list)
        
        ### removing problematic words found later in the DT classifier (words of importance in predicting POIs)
        remove = [u'catalytica', u'kennethpst', u'fernandez', u'elliot', u'ppas']
        for bad_word in remove:
            words = words.replace(bad_word, "")

    return words
