"""test.py

Should be deleted later
"""

from lxml import etree
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.feature_extraction.text import CountVectorizer
import time


xmlfile = 'Data/wikicomp-2014_ennl.xml'
start_tag = None

data_list = []

start_time = time.time()

for event, element in etree.iterparse(xmlfile, events=('start', 'end'), recover=True):
    if event == 'start' and start_tag is None:
        start_tag = element.tag
    if event == 'end' and element.tag == "articlePair" and (element.attrib.get('id') != None):
        if int(element.attrib.get('id')) > 1000:
            break
        categories = []
        for article in element.findall('article'):
            categories += article.find('categories').get('name').split("|")
        for article in element.findall('article'):
            content = ""
            for content_child in article.find('content'):
                if(content_child.text != None):
                    content += content_child.text
            a = {"language": article.attrib.get('lang'), "name": article.attrib.get(
                'name'), "categories": categories, "content": content}
            data_list.append(a)

df = pd.DataFrame(data_list)
print(df)

duration_time = time.time() - start_time
print(duration_time)