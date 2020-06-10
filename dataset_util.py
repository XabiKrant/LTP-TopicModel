"""dataset_util.py

"""

from lxml import etree
import pandas as pd
import time
from stop_words import get_stop_words


def process_dataset(xmlfile, n_documents):
    """This function calls the parse_data function defined below.
    Then it splits the resulting dataframe into an English and Dutch dataframe.

    args:
        xmlfile -- A string containing the path to the xml-file
                   This is used as input for parse_data
        n_documents -- The maximal number of documents we want
                    This is also used as input for parse_data

    returns:
        df_english -- The English part of the dataframe
        df_dutch -- The Dutch part of the dataframe
    """
    df = parse_data(xmlfile, max_id=n_documents)
    df_sample = df[:n_documents]
    df_english = df_sample[df_sample["language"] == "en"]
    df_dutch = df_sample[df_sample["language"] == "nl"]
    return df_english, df_dutch


def parse_data(xmlfile, max_id=10000):
    """This function reads the data from the specified xml file
    and outputs a DataFrame with the data.
    
    args:
        xmlfile -- A string containing the path to the xml-file
        max_id  -- The limit for when we stop parsing.
                   If max_id == None, we parse through the entire dataset.

    returns:
        df -- A pandas DataFrame object. The columns are
              language, name, categories and content
    """
    start_time = time.time()
    print("Parsing dataset...")

    start_tag = None
    data_list = []
    for event, element in etree.iterparse(xmlfile, events=('start', 'end'), recover=True):
        if event == 'start' and start_tag is None:
            start_tag = element.tag
        if event == 'end' and element.tag == "articlePair" and (element.attrib.get('id') != None):
            if max_id is not None and int(element.attrib.get('id')) > max_id:
                break
            categories = []
            for article in element.findall('article'):
                category = article.find('categories').get('name').split("|")
                while '' in category:
                    category.remove('')
                categories += category
            for article in element.findall('article'):
                content = ""
                for content_child in article.find('content'):
                    if(content_child.text != None):
                        content += content_child.text
                a = {"language": article.attrib.get('lang'), "name": article.attrib.get(
                    'name'), "categories": categories, "content": content}
                data_list.append(a)

    df = pd.DataFrame(data_list)

    # Get all the rows with singular categories and the empty categories out
    drop_indices = []
    for i in range(1, len(df), 2):
        if df["categories"][i-1-len(drop_indices)] != df["categories"][i-len(drop_indices)]:
            drop_indices.append(i-1-len(drop_indices))
    df = df.drop(drop_indices)
    df = df.reset_index()
    drop_indices = [i for i, _ in df.iterrows() if len(df.iloc[i]["categories"]) == 0]
    df = df.drop(drop_indices)

    parse_time = time.time() - start_time
    print(f"Parsing dataset: {xmlfile} took {parse_time} seconds.")
    return df

def preprocess_stopwords(language):
    """This function wraps the get_stop_words function imported
    from the stop_words library. This function also removes all apostrophes, 
    since the tokens from the corpus won't have these either.
    If we do not do this, we get a warning.

    args:
        language -- The language the stopwords come from
    returns:
        stop_words -- The list of stop preprocessed stop words
    """
    stop_words = get_stop_words(language)
    for index, word in enumerate(stop_words):
        stop_words[index] = word.replace("'", "")
    return stop_words