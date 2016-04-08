# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 11:16:18 2016

@author: pravinth
"""

import urllib2

def downloadFile(url, filepath):
    attempts = 0
    
    while attempts < 3:
        try:
            response = urllib2.urlopen("http://example.com", timeout = 5)
            content = response.read()
            f = open( "local/index.html", 'w' )
            f.write( content )
            f.close()
            break
        except urllib2.URLError as e:
            attempts += 1
            print type(e)
            print e
            

if __name__ == '__main__':
    downloadFile('https://www.kaggle.com/c/digit-recognizer/download/train.csv',
                 'train.csv')
    downloadFile('https://www.kaggle.com/c/digit-recognizer/download/test.csv',
                 'test.csv')
                 
                 
