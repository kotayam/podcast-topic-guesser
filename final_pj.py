import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from wordcloud import WordCloud
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

class LdaBuilder:
    
    def __init__(self, texts):
        self.texts = texts
    
    def __str__(self):
        s = ""
        for idx, topic in self.lda_model.print_topics(num_words=10):
            s += "Topic: {} \nWords: {} \n".format(idx, topic)
        return s
    
    def clean_text(self):
        """
        
        """
        return [LdaBuilder.clean_text_helper(text) for text in self.texts]

    def clean_text_helper(text):
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
                "topic", "us", "get", "things", "radio"}
        stop_words = set(stopwords.words('english')).union(my_dict)
        words = clean.split()
        filtered_words = [word for word in words if word.casefold() not in stop_words]
        clean = " ".join(filtered_words)

        # remove whitespace
        clean = " ".join(clean.split())

        return clean
    
    def create_wordcloud(clean_texts, max=50):
        """
        Creates a wordcloud from a given list of texts.

        Args:
            texts(list): list of text
            max(int): max number of word for wordcloud. 20 is default
        
        Returns:
            nothing
        """
        # combine text
        alltext = " ".join(clean_texts)

        # create a wordcloud object
        wordcloud = WordCloud(max_words=max, background_color="white", 
                            contour_width=3, contour_color="steelblue", min_word_length=2)
        # generate a wordcloud
        wordcloud.generate(alltext)

        # visualize the wordcloud
        img = wordcloud.to_image()

        # save to file
        img.save("wordcloud.jpg")

    def compute_tdf(self, clean_texts):
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

        data_words = list(sent_to_words(clean_texts))
        self.data_words = data_words

        # create dictionary
        id2word = corpora.Dictionary(data_words)
        self.id2word = id2word

        # create corpus
        texts = data_words

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        self.corpus = corpus

    def build_model(self, num_topics=10):
        """
        Builds LDA model

        Args:
            corpus(corpus): corpus
            id2word(dictionary): dictionary
            num_topics(int): number of topics to be generated

        Returns:
            ldamodel: the LDA Model
        """
        # clean text
        clean_texts = self.clean_text()

        # create wordcloud
        LdaBuilder.create_wordcloud(clean_texts)

        # compute tdf
        self.compute_tdf(clean_texts)

        # build/train model
        lda_model = gensim.models.LdaModel(corpus=self.corpus, 
                                        id2word=self.id2word,
                                        num_topics=num_topics,
                                        random_state=0,
                                        per_word_topics=True,
                                        chunksize=500,
                                        passes=10,
                                        alpha="auto")
        
        self.lda_model = lda_model
        return lda_model
    
    def coherence_score(self):
        """
        """
        coherence_model_lda = CoherenceModel(model=self.lda_model, texts=self.data_words, dictionary=self.id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        return coherence_lda


if __name__ == "__main__":
    # Read data
    podcasts = pd.read_csv("poddf.csv")
    print(podcasts.head())

    # create LdaBuilder
    lda_model = LdaBuilder(list(podcasts["Description"]))

    # build model
    lda_model.build_model(10)

    # print topics
    print(lda_model)

    # compute coherence score
    print(lda_model.coherence_score())
    
    
