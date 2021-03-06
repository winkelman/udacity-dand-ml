#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pickle
import time

from audit_emails import has_email_data
from parse_out_email_text import parseOutText

paths_dir = './emails_by_address'
file_list = os.listdir(paths_dir)


def add_email_paths(enron_data):
    """add email path info for to and from all persons."""
    persons_with_data = has_email_data(enron_data, print_out=False)

    for person in persons_with_data:
        enron_data[person]['paths'] = []
        email_address = enron_data[person]['email_address']
        # from this person
        file_name = "from_" + email_address + ".txt"
        with open(paths_dir + '/' + file_name) as f_in:
            for file_path in f_in:
                # remove first part of path, irrelevant
                idx = file_path.find(r'/')
                # remove newline character at end of path
                file_path = file_path[idx: -1]
                enron_data[person]['paths'].append(file_path)
        # to this person
        file_name = "to_" + email_address + ".txt"
        with open(paths_dir + '/' + file_name) as f_in:
            for file_path in f_in:
                # remove first part of path, irrelevant
                idx = file_path.find(r'/')
                # remove newline character at end of path
                file_path = file_path[idx: -1]
                enron_data[person]['paths'].append(file_path)
        # email address no longer needed
        enron_data[person].pop('email_address')

    return enron_data


def create_corpora(enron_data):
    """create email corpus for each person with email data.
    input dictionary should be output of 'add_email_paths' function.
    corpus key-value is list of all email strings for that person.
    this can take a while."""
    paths_not_found = 0
    for person in enron_data:
        paths = enron_data[person].get('paths', None)
        enron_data[person]['corpus'] = []
        if paths:
            for path in paths:
                # try block, some paths don't exist
                try:
                    # 'maildir' up one level
                    with open('..' + path) as f_in:
                        email_text = parseOutText(f_in)
                        enron_data[person]['corpus'].append(email_text)
                except IOError:
                    paths_not_found += 1
            # paths no longer needed
            enron_data[person].pop('paths')
    print "{} paths not found".format(paths_not_found)
    return enron_data


def build_email_dataset(enron_data):
    # time to build corpus
    start_time = time.time()
    corpus_data = create_corpora(add_email_paths(enron_data))
    elapsed_time = time.time() - start_time
    print "time to build corpus: {} minutes".format(round(elapsed_time/60.0, 2))
    # export file
    with open("emails_dataset.pkl", "w") as f_out:
        pickle.dump(corpus_data, f_out)


if __name__ == '__main__':
    pass
    '''
    with open("final_project_dataset.pkl", "r") as f_in:
        enron_data = pickle.load(f_in)
    build_emails_dataset(enron_data)
    '''