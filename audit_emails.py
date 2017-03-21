#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pickle

# email messages in 'maildir'.
# each person has a directory file in 'emails_by_address'.
# directory file is named with format "from_<email_address>.txt" or "to_<email_address>.txt"
# file contains full path to all emails for that person in 'maildir'.
paths_dir = './emails_by_address'
# list of file names
file_list = os.listdir(paths_dir)
# null value in data is "NaN"
null_val = "NaN"


def has_email_data(enron_data, print_out=True):
	"""check if people with email addresses have email data"""
	email_and_data = []
	for person in enron_data:
		email_address = enron_data[person]['email_address']
		# if have email address
		if email_address != null_val:
			supposed_from_file = "from_" + email_address + ".txt"
			supposed_to_file = "to_" + email_address + ".txt"
			# if to or from messages present
			if supposed_from_file in file_list or supposed_to_file in file_list:
				email_and_data.append(person)
			# print audit
			elif print_out:
				from_msg = enron_data[person]['from_messages']
				to_msg = enron_data[person]['to_messages']
				# make sure to/from message features match up
				if from_msg != null_val and to_msg != null_val:
					print "{} has non-null email feature but no email data".format(person)
				else:
					print "{} has email address but no data consistent with features".format(person)
	return email_and_data


def validate_from_email_counts(enron_data):
	"""compare count of 'from_messages' to emails in corpus
	for people with data.
	for now only from messages will be used to build email feature."""
	has_data = has_email_data(enron_data, print_out=False)
	miscounts = []
	for person in enron_data:
		if person in has_data:
			supposed_email_count = enron_data[person]['from_messages']
			# some existing emails with "NaN" string for count
			if supposed_email_count == null_val:
				supposed_email_count = 0
			email_address = enron_data[person]['email_address']
			text_file = "from_" + email_address + ".txt"
			corpus_email_count = 0
			with open(paths_dir + '/' + text_file) as f_in:
				# sum each line in text file
				corpus_email_count = sum(1 for line in f_in)
			# compare counts
			if supposed_email_count != corpus_email_count:
				miscounts.append(person)
				print "{} has count of\n{} emails but\n{} emails in corpus\n".format(person, supposed_email_count, corpus_email_count)
	return


if __name__ == '__main__':
	pass
	'''
	with open("final_project_dataset.pkl", "r") as f_in:
    	enron_data = pickle.load(f_in)
	has_email_data(enron_data)
	validate_from_email_counts(enron_data)
	'''