import string
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from gensim.models.ldamodel import LdaModel
import gensim.corpora as corpora

# dictionary for topics
topics = {0: "Business",
          1: "Personal Journal",
          2: "Legal",
          3: "Religion/Spirituality",
          4: "Fitness/Self-Help",
          5: "Health",
          6: "World News",
          7: "Technology",
          8: "Sports",
          9: "Shopping/Hobbies"}

def clean_query(query):
    """
    Clean the query text.
    """
    # remove numbers
    clean = re.sub(r"\d", "", query)

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

def predict_topic(query, lda_model, id2word):
    """
    Predict the top 3 topic prediction for the cleaned query.
    """
    query_data = query
    bow_query_data = id2word.doc2bow(query_data.lower().split())
    query_topic = lda_model[bow_query_data]

    sorted_topics = sorted(query_topic[0], key=lambda x: x[1], reverse=True)

    top3 = "Top 5 Topics:"
    for i in range(0,5):
        name = topics[sorted_topics[i][0]]
        prob = sorted_topics[i][1]
        top3 += "\n{}: Topic: {}, Probability: {}".format(i+1, name, prob)
    
    return top3

if __name__ == "__main__":
    # load trained lda model
    topic_guesser = LdaModel.load("lda_model")

    # load dictionary
    id2word = corpora.Dictionary.load("lda_dictionary")

    # load query
    query = ""
    with open("query.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip() + " "
            query += line

    # clean query
    clean_q = clean_query(query)
    print(clean_q)

    # predict topics
    result = predict_topic(clean_q, topic_guesser, id2word)
    print(result)