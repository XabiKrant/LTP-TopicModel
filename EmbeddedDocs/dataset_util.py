"""dataset_util.py

"""

from lxml import etree
import pandas as pd
import time
from stop_words import get_stop_words


def process_dataset(xmlfile, n_documents, topNcats):
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
    df = take_top_N_categories(df, topNcats)
    df_sample = df[:n_documents]
    df_english = df_sample[df_sample["language"] == "en"]
    df_dutch = df_sample[df_sample["language"] == "nl"]
    df_english = df_english.reset_index(drop=True)
    df_dutch = df_dutch.reset_index(drop=True)
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
    df = df.reset_index(drop=True)
    drop_indices = [i for i, _ in df.iterrows() if len(df.iloc[i]["categories"]) == 0]
    df = df.drop(drop_indices)

    parse_time = time.time() - start_time
    print(f"Parsing dataset: {xmlfile} took {parse_time:.3f} seconds.")
    return df

def take_top_N_categories(df, N=300):
    """This function removes rows from the dataframe that
    do not have a category that is in the top 500 categories.
    Then this function also removes the categories from the
    remaining rows such that only categories from the top 500 remain.
    """
    counts = {}
    for i, row in df.iterrows():
        cats = row["categories"]
        # counts[cat] += 1 for cat in cats if cat in counts else counts[cat] = 0
        for cat in cats:
            if cat in counts:
                counts[cat] += 1
            else:
                counts[cat] = 0

    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top_500_cats = [cat for cat, count in sorted_counts[:N]]
    
    drop_indices = []
    for i, row in df.iterrows():
        row_stays = False
        cats = row["categories"]

        for j, cat in enumerate(cats.copy()):
            if cat in top_500_cats:
                row_stays = True
            else:
                cats.remove(cat)

        if not row_stays:
            drop_indices.append(i)
        else:
            df.at[i, "categories"] = cats

    df = df.drop(drop_indices)
    df = df.reset_index(drop=True)
    return df


def construct_stopwords():
    """This function wraps the get_stop_words function imported
    from the stop_words library. This function also removes all apostrophes, 
    since the tokens from the corpus won't have these either.
    If we do not do this, we get a warning.

    returns:
    stop_words -- The list of stop preprocessed stop words
    """

    # Remove HTML, CSS, etc from Wikipedia articles
    stopwords = get_stop_words('en')
    stopwords.extend(get_stop_words('nl'))
    stopcode = ["div", "class", "id", "href", "align", "left", "right", "body", "title", "head",
                "h1", "h2", "h3", "h4", "h5", "h6", "hr", "br", "html", "img", "jpg", "li", "ol", "ul",
                "p", "attribute", "tag"]
    stopwords.extend(stopcode)
    for i in range(2020):
        # Remove all numerals as well, mostly years
        stopwords.append(str(i))
        stopwords.append(str(i).zfill(2))

    for index, word in enumerate(stopwords):
        stopwords[index] = word.replace("'", "")
    return stopwords