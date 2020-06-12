# LTP-TopicModel

## Dependencies
This project requires Python 3.6 or higher.
We make use of the following libraries:  scikit-learn, stop_words, lmxl and pandas, nltk, translate, fasttext and gensim. Install these libraries with the following command:
```bash
pip3 install scikit-learn stop_words lxml pandas nltk translate fasttext gensim
```

## Dataset
The dataset (EN-NL, 5.7GB) can be downloaded from https://www.dropbox.com/s/vrf6y3e3b3x3624/wikicomp-2014_ennl.xml.bz2?dl=0  
If this URL does not work, please try to download the English-Dutch dataset from https://linguatools.org/tools/corpora/wikipedia-comparable-corpora/  
It is imperative that this dataset is put in a directory called Data. The location for this dataset is hardcoded in the code.  

The fasttext multilanguage embeddings are downloaded in the code itself. However this is a heavy operation and it can be that your computer will run out of memory. The embeddings can also be downloaded from https://fasttext.cc/docs/en/crawl-vectors.html, after this the dimension should be reduced to 50. The new bins can either be put in the Data directory, or the bin names can be changed. 

## Running
When running this program, either the LDA version or the Neural approach version, please keep in mind that parsing the dataset will take some time, usually around a  minute. On our machines the complete LDA takes around 20 minutes, mainly because the translation takes quite some time. The multilingual Doc2Vec takes a lot longer, approximately one and a half hour. 

To run the LDA implementation with embeddings (NOTE: The downloads of these embeddings are large and will take a considerable amount of time), run the following command:
```bash
python3 lda.py --embeddings
```

### Multilingual Doc2Vec approach
To run the multilingual Doc2Vec implementation, use the following command:
```bash
python3 EmbeddedDocs/unsupervised.py
```
Use --device gpu and --cuda if you would like to use your GPU.

### Feature based neural approach
To train and test the feature based neural approach and doc2vec models, use the following command, with --cuda if you would like to use your GPU:
```bash
python3 network.py --train [--cuda]
```
The command above will also save the network and doc2vec models, so if you want to just test them, you can do so with:
```bash
python3 network.py [--cuda]
```