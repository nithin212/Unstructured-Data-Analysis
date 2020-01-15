'''Text Cleaning techniques:
1. Normalizing text: case normalization
2. Tokenize: taking the smallest part of the text
    word_tokenize(),wordpunct_tokenize(),tweettokenizer,regexp_tokenize
3. Removing stop words and punctuations
    stop words are connectors which add no value to my analysis
4. Stemming and lememtization -takes the words to its root
'''

import nltk
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import SnowballStemmer
from wordcloud import WordCloud, STOPWORDS

data=pd.read_csv("Put the path of your file which has Lenovo K8 data")
def clean_text(sent):
    terms1=word_tokenize(sent.lower())
    stop_nltk=stopwords.words("english")
    stop_updated=stop_nltk+["...","..","!!"]+["phone","mobile","lenovo","k8","note","amazon"]
    reviews_upated1=[term for term in terms1 \
                if term not in stop_updated
                and term not in list(punctuation) and len(term)>2]
    list_stem=[stemmer_s.stem(word) for word in reviews_upated1]
    res=" ".join(list_stem)
    return(res)
stemmer_s=SnowballStemmer("english")
data["Clean_review"]=data.review.apply(clean_text) # Adding the cleaned text as a column to the dataframe

all_joinedterms=" ".join(data.Clean_review.values) #combining all the lines of newly created column and creating a single string

word_cloud=WordCloud(max_font_size=150,background_color="white").generate(all_joinedterms) #Creating wordcloud
plt.imshow(word_cloud) #Code to siplay the wordcloud

all_joinedterms1=word_tokenize(all_joinedterms.lower()) #tokenizing the words
fdist=FreqDist(all_joinedterms1) #Finding the frequencies of all the words 

plt.figure(figsize=(10,10))
fdist.plot(20,cumulative=False) # creating a plot(line chart) for 20 words
plt.show() #to show/display the plot of freqeuncies




