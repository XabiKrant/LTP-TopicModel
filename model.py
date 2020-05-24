from lxml import etree
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.feature_extraction.text import CountVectorizer


xmlfile = 'Data/wikicomp-2014_ennl.xml'
start_tag = None

data_list = []

for event, element in etree.iterparse(xmlfile, events=('start', 'end'), recover=True):
    if event == 'start' and start_tag is None:
        start_tag = element.tag
    if event == 'end' and element.tag == "articlePair" and (element.attrib.get('id') != None):
        if int(element.attrib.get('id')) <= 1000:
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

corpus = df['content'].tolist()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
clustering = MeanShift().fit(X.toarray())

purity_dict = [{}] * (max(clustering.labels_) + 1)

for index, row in df.iterrows():
    label_dict = purity_dict[clustering.labels_[index]]
    for category in row["categories"]:
        if category in label_dict:
            label_dict[category] += 1
        else:
            label_dict[category] = 1
    purity_dict[clustering.labels_[index - 1]] = label_dict

dict_sum = 0 
for label_dict in purity_dict:
	dict_sum += label_dict[max(label_dict)]

purity = dict_sum/df.size
print("Purity score ", purity)