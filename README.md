# LTP-TopicModel

## Dependencies
This project requires Python 3.6 or higher.
We make use of the following libraries:  scikit-learn, stop_words, lmxl and pandas, nltk and gensim. Install these libraries with the following command:
```bash
pip3 install scikit-learn stop_words lxml pandas nltk gensim fasttext
```

## Dataset
The dataset (EN-NL, 5.7GB) can be downloaded from https://www.dropbox.com/s/vrf6y3e3b3x3624/wikicomp-2014_ennl.xml.bz2?dl=0  
If this URL does not work, please try to download the English-Dutch dataset from https://linguatools.org/tools/corpora/wikipedia-comparable-corpora/  
It is imperative that this dataset is put in a directory called Data. The location for this dataset is hardcoded in the code.  

## Running
When running this program, either the LDA version or the Neural approach version, please keep in mind that parsing the dataset will take some time. On our machines, this usually takes around 70-80 seconds.

To run the LDA implementation, use the following command:
```bash
python3 lda.py
```

To run the LDA implementation with embeddings (NOTE: The downloads of these embeddings are large and will take a considerable amount of time), run the following command:
```bash
python3 lda.py --embeddings
```

To train and test the neural approach, use the following command, with --cuda if you would like to use your GPU:
```bash
python3 network.py --train [--cuda]
```
The command above will also save the network, so if you want to just test that network, you can do so with:
```bash
python3 network.py [--cuda]
```