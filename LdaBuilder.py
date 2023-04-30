import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from wordcloud import WordCloud
import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

class LdaBuilder:
    def __init__(self, texts):
        """
        Construct an instance of the LdaBuilder class.

        Attributes:
            - self.texts: list of string containing descriptions from the dataset
            - self.data_words: list of tokeninzed text
            - self.id2word: dictionary of words
            - self.corpus: list containing the term document frequency for every word
            - self.lda_model: the trained LDA model
        
        Args:
            texts (list): list of string containing podcast descriptions
        """
        self.texts = texts
        self.data_words = None
        self.id2word = None
        self.corpus = None
        self.lda_model = None
    
    def __str__(self):
        """
        Defines the string representation of a LdaBuilder instance.

        Prints the topics and its keywords produced by the model.

        Args:
            None
        
        Returns:
            str: a LdaBuilder's string representation
        """
        s = ""
        for idx, topic in self.lda_model.print_topics(num_words=15):
            s += "Topic: {} \nWords: {} \n".format(idx, topic)
        return s
    
    def clean_text(self):
        """
        cleans the entire text (descriptions) from the data.
        Calls helper function.

        Args:
            None
        
        Returns:
            list: a list of cleaned texts
        """
        clean_texts = [LdaBuilder.clean_text_helper(text) for text in self.texts]
        return clean_texts

    def clean_text_helper(text):
        """
        Helper function for clean_text. 
        Cleans a single description text. 
        Removes numbers, punctuation, stopwords, converts to lower case, 
        remove stopwords and whitespace.

        Args:
            text (str): description to be cleaned

        Returns:
            str: cleaned description
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
                "topic", "us", "get", "things", "radio", 
                "de", "eastern", "utc", "monday", "pm"}
        #, "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "am"
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
        Saves an image to file "wordcloud.jpg"

        Args:
            clean_texts (list): list of cleaned texts
            max (int): max number of word for wordcloud. (default: 50)
        
        Returns:
            None
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
        Computes the term document frequecny for the given texts.
        Updates the class attributes accordingly.

        Args:
            clean_texts (list): list of cleaned texts

        Returns:
            None
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

        # save the dictionary
        id2word.save("lda_dictionary")

        # create corpus
        texts = data_words

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        self.corpus = corpus

    def build_model(self, num_topics=10):
        """
        Builds and trains the LDA model

        Args:
            num_topics (int): number of topics to be generated. (default: 10)

        Returns:
            lda_model: the trained LDA Model
        """
        # clean text
        clean_texts = self.clean_text()

        # create wordcloud
        LdaBuilder.create_wordcloud(clean_texts)

        # compute tdf
        self.compute_tdf(clean_texts)

        # build/train model
        # set seed to 0 to reproduce same result
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
        Computes the coherence score for the model.

        Args:
            None

        Returns:
            double: the coherence score
        """
        coherence_model_lda = CoherenceModel(model=self.lda_model, texts=self.data_words, dictionary=self.id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        return coherence_lda
    
    def save_model(self):
        """
        Saves the trained model to file "lda_model"

        Args:
            None
        
        Returns:
            None
        """
        self.lda_model.save("lda_model")

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

    # save model
    lda_model.save_model()