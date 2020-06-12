# LTP-TopicModel

## Dependencies
This project requires Python 3.6 or higher.
We make use of the following libraries:  scikit-learn, stop_words, lmxl and pandas, nltk, translate, fasttext and gensim. Install these libraries with the following command:
```bash
pip3 install scikit-learn stop_words lxml pandas nltk translate fasttext gensim
```

## Dataset
The dataset (EN-NL) can be downloaded from https://www.dropbox.com/s/vrf6y3e3b3x3624/wikicomp-2014_ennl.xml.bz2?dl=0  
It is imperative that this data is put in a directory called Data. The location for this dataset is hardcoded in the code.  

The fasttext multilanguage embeddings are downloaded in the code itself. However this is a heavy operation and it can be that your computer will run out of memory. The embeddings can also be downloaded from https://fasttext.cc/docs/en/crawl-vectors.html, after this the dimension should be reduced to 50. The new bins can either be put in the Data directory, or the bin names can be changed. 

## Running
When running this program, either the LDA version or the Neural approach version, please keep in mind that parsing the dataset will take some time, usually around a  minute. On our machines the LDA takes around 8 minutes, mainly because the translation takes some time. The multilingual Doc2Vec takes a lot longer, approximately one and a half hour. 

To run the LDA implementation, use the following command:
```bash
python3 lda.py
```

To run the multilingual Doc2Vec implementation, use the following command:
```bash
python3 EmbeddedDocs/unsupervised.py
```