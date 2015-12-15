#!/usr/bin/python
# -*- coding: utf-8 -*-



## HERE WE ADD A STRING FEATURE ('WORDS') TO DATA_DICT FOR EACH PERSON.
## IF WE HAVE EMAIL DATA FOR THAT PERSON, THE STRING IS A CONCATENATION
## OF ALL PROCESSED (STEMMED, ETC.) EMAIL TEXT FOR THAT PERSON.
## IF WE HAVE NO EMAIL DATA THEN THE STRING IS EMPTY ''.

import os
import pickle

## BEFORE WE PROCESS EMAIL TEXT, LET'S AUDIT EMAIL ADDRESSES TO SEE WHO HAS DATA.

def audit_emails():
	
	enron_data = pickle.load(open("final_project_dataset.pkl", "r"))
	
	## list of file names in the emails_by_address directory
	file_list = os.listdir('./emails_by_address')	
	
	## email address status for all people
	poi_no_email = []
	non_poi_no_email = []
	poi_email = []
	non_poi_email = []
	
	for person in enron_data:
		poi_val = enron_data[person]['poi']
		email = enron_data[person]['email_address']
		
		if email == "NaN":
			if poi_val:
				poi_no_email.append(person)
			else:
				non_poi_no_email.append(person)
				
		else:
			if poi_val:
				poi_email.append(person)
			else:
				non_poi_email.append(person)
				
	print "POI without email address: ", poi_no_email
	print "Non POI without email address: ", non_poi_no_email
	#print "POI with email address: ", poi_email
	#print "Non POI with email address: ", non_poi_email
	print "Number of people without email address: ", len(poi_no_email + non_poi_no_email)
	
	## all POIs have email addresses but 35 others without an email address
	
	
	

	## email data status for people with email addresses
	have_email = poi_email + non_poi_email
	have_poi_data = []
	no_poi_data = []
	have_data = []
	no_data = []
	
	for person in have_email:
		email = enron_data[person]['email_address']
		file_name = "from_" + email + ".txt"
		
		if file_name in file_list:
			if person in poi_email:
				have_poi_data.append(person)
			else:
				have_data.append(person)
		else:
			if person in poi_email:
				no_poi_data.append(person)
			else:
				no_data.append(person)
	
	#print "POI with email data: ", have_poi_data
	print "POI with NO email data: ", no_poi_data
	#print "All others with email data: ", have_data
	print "All others without email data: ", no_data
	print "Number of POIs without email data: ", len(no_poi_data)
	print "Number of non POIs with email address but no email data: ", len(no_data)
	#print enron_data['PAI LOU L'] ## not a POI?
	#print enron_data['BERBERIAN DAVID'] ## this name occurs as an important word feature in identifying POIs
	
	## 4 POIs without email data and 21 non-POIs with email addresses but no email data
	
	
	
	
	## let's see if the 'to_messages' feature matches up with what we found and can give us a clue as to who has email data
	poi_have_message = []
	poi_no_message = []
	have_message = []
	no_message = []
	for person in enron_data:
		poi_val = enron_data[person]['poi']
		msg = enron_data[person]['to_messages']
		
		if poi_val:
			if msg != "NaN":
				poi_have_message.append(person)
			else:
				poi_no_message.append(person)
		else:
			if msg != "NaN":
				have_message.append(person)
			else:
				no_message.append(person)
		
	print (poi_have_message == have_poi_data) and (have_message == have_data)
	
	## the 'to_messages' feature can tell us whether or not we have email data!






## ADD 'WORD' FEATURE TO DATA_DICT.
## WE PASS IN DATA_DICT HERE, ADD THE FEATURE TO IT,
## THEN RETURN THE DICTIONARY WITH NEW FEATURE.

## FOR THOSE PEOPLE WITH NO EMAIL ADDRESS AND/OR DATA,
## WE IMPUTE WITH AN EMPTY STRING ""

from parse_email_text import parseOutText


def add_words(data_dict, n=30):
    
        if n < 2: ## make sure that n is at least 2 so we don't get an error
                n = 2
	
	os.chdir('./emails_by_address') # change back directory at end of function

	for person in data_dict:
		
		email_address = data_dict[person]['email_address']
		from_emails = data_dict[person]['to_messages'] ## checking for email data
		
		if (email_address != "NaN") and (from_emails != "NaN"):

			word_data = []  ## store each email as a string element in a list
			word_string = ''  ## concatenate each email element into one long string after processing all emails, do this for each person
			path = "from_" + email_address + ".txt"  ## path of the file in the 'emails_by_address' folder
			#print person, path  ## double check that all paths are correct
                        
                        try: # GitHub: try and except block here as we have limited people/data in 'emails_by_address'
			
                                with open(path, 'rb') as file_dir:  ## path is actually a text file containing multiple email text file paths for each person
                                        
                                        email_limit_counter = 0 ## this allows us to control how many emails to process per person
                                        
                                        for email_path in file_dir:
                                                
                                                email_limit_counter += 1
                                                
                                                start_idx = email_path.find('/') ## email paths are not correct as is
                                                # note the path below is different for the GitHub project folder; '../../' for local or '../' for GitHub
                                                actual_email_path = '../' + email_path[start_idx + 1: -2] ## -2 because we have a period on the end of the file name
                                                
                                                if email_limit_counter < n: ## the limit of how many emails to process per person
                                                        
                                                        #print actual_email_path ## double check that the email path is correct
                                                        
                                                        try: # GitHub: try and except block here as we have limited people/data in 'maildir'
                                                            
                                                                with open(actual_email_path, 'rb') as email:
                                                                        
                                                                        email_text = parseOutText(email) ## applying word stemmer and removing words with numbers, returns a string of the entire email
                                                                        word_data.append(email_text)
                                                                
                                                        except IOError:
                                                                print actual_email_path, " not found for: ", person
                                                                
                        except IOError:
                                print path, " does not exist in 'emails_by_address' for: ", person
						
		
                        if word_data:  ## check if we have word data at all, paths and/or emails could have been deleted somehow in maildir...
                                word_string = ' '.join(word_data)
                                data_dict[person]['words'] = word_string
                        else:
                                data_dict[person]['words'] = ''
			

		else: ## for people with no email address or no email data
			data_dict[person]['words'] = ''

        os.chdir('..')

	return data_dict
			
			
			
			
if __name__ == "__main__":
	audit_emails()
