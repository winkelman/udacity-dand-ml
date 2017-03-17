#!/usr/bin/python
# -*- coding: utf-8 -*-

import string
import re

from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords # not needed with pos?

lemma = WordNetLemmatizer()
stop_words = stopwords.words("english")

replace_string = '''
_______________________________________________________
This message may contain confidential and/or legally privileged
information.
If it has been sent to you in error, please reply immediately to advise the
sender of the error and then destroy this message, any copies of this
message and any printout of this message.  If you are not the intended
recipient of the message, any unauthorized dissemination, distribution or
copying of the material in this message, and any attachments to the
message,
is strictly forbidden.
'''

meta_content = re.compile(r'\@|cc:|X-FileName|-----|=====')
# should speed this up, not all nltk libs necessary

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        """

    f.seek(0)  ### go back to beginning of file (annoying)
    email_text = f.read()
    # last index of metadata, nothing pretty
    start = email_text.find("X-FileName:")
    # remove replace string and split on newlines
    lines = email_text[start: ].replace(replace_string, '') \
                                .splitlines()
    # filter lines with metadata content and remove punctuation
    lines = [line.translate(None, string.punctuation) for line in lines if not meta_content.search(line)]
    clean_text = []
    for line in lines:
        # whole words and lower case
        for word in re.findall(r'\b[a-z]+\b', line.lower()):
            # not a stop word
            if word not in stop_words:
                # lemmatize word
                clean_text.append(lemma.lemmatize(word))
    # nouns only
    nouns = [word for word, pos in pos_tag(clean_text) if pos == 'NN']
    # return as string
    return ' '.join(nouns)