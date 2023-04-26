The new project I'm thinking of is a web app for predicting genres for a Podcast based on its channel description. A user (podcast maker) can upload a text file containing a description of their podcast channel, and receive a prediction result for what genre/topic their podcast will be categorized in.  

1. Creating the prediction model:

I will use topic modeling (LDA analysis) for my prediction model. I will train the model using data from Kaggle: https://www.kaggle.com/datasets/roman6335/13000-itunes-podcasts-april-2018

For the topics, I will try to assign topic names from this list:  https://chartable.com/charts/spotify/us

I will use pandas to load the data and re to clean the data. I will use gensim for topic modeling.

2. Creating the web app:

Similar to HW6, I will create a locally hosted web application where users can upload text files containing their channel description and view prediction results. 

I will use flask and jinja to do this.

If there is time, I would like to try deploying the website.