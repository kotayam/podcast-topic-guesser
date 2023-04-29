import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from wordcloud import WordCloud
import gensim
import gensim.corpora as corpora

class LdaBuilder:
    
    def __init__(self, texts):
        self.texts = texts
    
    def clean_text(text):
        """
        Cleans the description text. 
        Removes numbers, punctuation, stopwords, converts to lower case, 
        remove stopwords and whitespace.

        Args:
            text (String): description to be cleaned

        Returns:
            String: cleaned description
        """
        # remove numbers
        clean = re.sub(r"\d", "", text)

        # remove punctuation
        translator = str.maketrans("", "", string.punctuation)
        clean = clean.translate(translator)

        # lower case
        clean = clean.lower()

        # remove stop words
        my_dict = {"podcast", "show", "stories", "talk", 
                "share", "weekly", "take", "hosted", 
                "thing", "conversation", "listen", "host", 
                "topic"}
        stop_words = set(stopwords.words('english')).union(my_dict)
        words = clean.split()
        filtered_words = [word for word in words if word.casefold() not in stop_words]
        clean = " ".join(filtered_words)

        # remove whitespace
        clean = " ".join(clean.split())

        return clean

    def create_wordcloud(texts, max=30):
        """
        Creates a wordcloud from a given list of texts.

        Args:
            textdf(list): list of text
            max(int): max number of word for wordcloud. 20 is default
        
        Returns:
            nothing
        """
        # combine text
        alltext = " ".join(texts)

        # create a wordcloud object
        wordcloud = WordCloud(max_words=max, background_color="white", 
                            contour_width=3, contour_color="steelblue")

        # generate a wordcloud
        wordcloud.generate(alltext)

        # visualize the wordcloud
        img = wordcloud.to_image()

        # save to file
        img.save("wordcloud.jpg")

    def compute_tdf(texts):
        """
        Computes the term document frequecny for the given texts

        Args:
            texts(list): list of text

        Returns:
            list: list of corpus
        """
        # tokenize text
        def sent_to_words(sentences):
            for sentence in sentences:
                yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

        data_words = list(sent_to_words(texts))

        # create dictionary
        id2word = corpora.Dictionary(data_words)

        # create corpus
        texts = data_words

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        return corpus

    def build_model(self, corpus, id2word, num_topics):
        """
        Builds LDA model

        Args:
            corpus(corpus): corpus
            id2word(dictionary): dictionary
            num_topics(int): number of topics to be generated

        Returns:
            ldamodel: the LDA Model
        """
        clean_text = [clean_text(text) in text in self.texts]
        lda_model = gensim.models.LdaMulticore(corpus=corpus, 
                                        id2word=id2word,
                                        num_topics=num_topics)
        
        return lda_model

if __name__ == "__main__":
    # Read data
    podcasts = pd.read_csv("poddf.csv")
    print(podcasts.head())

    # clean description
    podcasts["clean_desc"] = podcasts["Description"].apply(clean_text)
    print(podcasts.head(10))

    # create wordcloud
    create_wordcloud(list(podcasts["clean_desc"]))

    # compute tdf (term document frequency)
    tdf = compute_tdf(list(podcasts["clean_desc"]))
    print(tdf[:1])

    # create lda model
    lda_model = build_model()
