import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


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
    stop_words = set(stopwords.words('english'))
    words = clean.split()
    filtered_words = [word for word in words if word.casefold() not in stop_words]
    clean = " ".join(filtered_words)

    # remove whitespace
    clean = " ".join(clean.split())

    return clean

if __name__ == "__main__":
    # Read data
    podcasts = pd.read_csv("poddf.csv")
    print(podcasts.head())

    # clean description
    podcasts["clean_desc"] = podcasts["Description"].apply(clean_text)
    print(podcasts.head(10))