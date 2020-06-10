# LTP-TopicModel

## Dependencies

Python 3.6 or higher.  

We make use of the following libraries: scikit-learn, stop_words, lmxl and pandas, nltk and gensim. Install these libraries with the following command:
```bash
pip3 install scikit-learn stop_words lxml pandas nltk gensim
```
When running this program, either the LDA version or the Neural approach version, please keep in mind that parsing the dataset will take some time. On our machines, this will usually take around 70-80 seconds.  

To run the LDA implementation, use the following command:
```bash
python3 lda.py
```