# Program to scrape app reviews from Play Store

# Import libraries
import requests # For rendering webpages
from bs4 import BeautifulSoup # Crawl webpage information
import pickle
import os
import pandas as pd

# Global variables
# URLs of transcripts in scope
urls = ['http://scrapsfromtheloft.com/2017/05/06/louis-ck-oh-my-god-full-transcript/',
        'http://scrapsfromtheloft.com/2017/04/11/dave-chappelle-age-spin-2017-full-transcript/',
        'http://scrapsfromtheloft.com/2018/03/15/ricky-gervais-humanity-transcript/',
        'http://scrapsfromtheloft.com/2017/08/07/bo-burnham-2013-full-transcript/',
        'http://scrapsfromtheloft.com/2017/05/24/bill-burr-im-sorry-feel-way-2014-full-transcript/',
        'http://scrapsfromtheloft.com/2017/04/21/jim-jefferies-bare-2014-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/02/john-mulaney-comeback-kid-2015-full-transcript/',
        'http://scrapsfromtheloft.com/2017/10/21/hasan-minhaj-homecoming-king-2017-full-transcript/',
        'http://scrapsfromtheloft.com/2017/09/19/ali-wong-baby-cobra-2016-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/03/anthony-jeselnik-thoughts-prayers-2015-full-transcript/',
        'http://scrapsfromtheloft.com/2018/03/03/mike-birbiglia-my-girlfriends-boyfriend-2013-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/19/joe-rogan-triggered-2016-full-transcript/']

# Comedian names
comedians = ['louis', 'dave', 'ricky', 'bo', 'bill', 'jim', 'john', 'hasan', 'ali', 'anthony', 'mike', 'joe']

# Web scraper (function accepts a URL string and returns the desired text transcript)
def url_to_transcript(url):
    
    '''Returns transcript data specifically from scrapsfromtheloft.com.'''
    page = requests.get(url).text
    soup = BeautifulSoup(page, "lxml")
    text = [p.text for p in soup.find(class_="post-content").find_all('p')]
    print(url)
    return text
	
# Get transcripts and store it in a list
transcripts = [url_to_transcript(u) for u in urls]

print ("Transcripts:\n", transcripts)

# Store transcripts
# Create directory if it doesn't exists
!mkdir transcripts # OR os.mkdir (os.getcwd() + "/transcripts")

# Store transcripts as files (pickle transcripts)
for i, c in enumerate(comedians): # Enumerate function creates and maps a counter/index to each element in comedians
    
    # Create file object
    with open("transcripts/" + c + ".txt", "wb") as file:
        
        # Dump transcript to file
        pickle.dump(transcripts[i], file)
        
    # Close file object
    file.close ()