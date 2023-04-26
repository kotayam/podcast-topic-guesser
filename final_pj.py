import pandas as pd
import string
import re

def clean_text(text):
    """
    Cleans the description text. 
    Removes numbers, punctuation, and converts to lower case.

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