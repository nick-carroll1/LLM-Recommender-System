# LLM Recommender System
This repository uses ChatGPT to create a movie recommender system and compares the recommender's performance across four prompt templates.

## Project Overview

This project aims to build a movie recommendation system by leveraging the OpenAI API, integrating Large Language Models (LLM) to enhance the recommendation accuracy through innovative prompt engineering. We specifically focus on the application of four distinct prompt templates to evaluate the performance of movie recommendations. By conducting a comparative analysis of these templates, we aim to identify the most effective approach for generating personalized movie recommendations.

## Methodology
![image](https://github.com/nick-carroll1/LLM-Recommender-System/assets/110933007/73024e3a-4c72-450c-8e4f-daade755fa08)

### Data
In this project, we use the MovieLens 100K dataset. Collected by the GroupLens Research Project at the University of Minnesota[1]. MoiveLens 100K is a widely-used benchmark dataset in the field of recommender systems. Released in April 1998, it comprises 100,000 movie ratings, ranging from 1 to 5, provided by 943 users on 1,682 movies. Each user has rated at least 20 movies, ensuring a reasonable level of engagement and data density for analysis.

The dataset can be found here: https://grouplens.org/datasets/movielens/

The original dataset was provided in MATLAB format. We converted it using the [prepare_data.ipynb] notebook and stored the processed data in ```processed_movie100k.csv``` for subsequent use. We also implemented a 70/30 train-test split to facilitate model training and evaluation. For prompt engineering, we enriched the dataset by appending a one-sentence summary from Wikipedia to each movie to provide additional context. These movie summaries, along with their corresponding titles, are compiled in ```movie_wiki.csv```.

### Prompt Engineering

We experimented four different types of prompts to evaluate their effectiveness. Here are the templates:

1. **A collaborative filter type prompt**

```
I am user {User ID}. 
The most recent ten movies I have seen are: {List of ten recent movies}. 
My top rated movies are: {List of ten top rated movies}. 
The users who are most like me are {10 user id's of similar users}. 
The top movies for each of these users are: {Similar User ID: List of ten top rated movies}. 
Please recommend ten movies for me to watch that I have not seen. 
Please use brackets around the movies you recommend and separate the titles by new lines so I can easily parse them.
(Format Example: Here are the 10 movies recommended for you: [Midnight Cowboy (1969){new_line}Lost in Translation (2003){new_line}etc.]) 
Answer:
```

2. **A prompt that provides a candidate set and the genres of a user's top rated movies**

```
Candidate Set (candidate movies): {List of candidate movies}. 
The movies I have rated highly (watched movies): {List of ten top rated movies}. 
Their genres are: {List of genres from the ten top rated movies}. 
Can you recommend 10 movies from the Candidate Set similar to but not in the selected movies I've watched?. 
Please use brackets around the movies you recommend and separate the titles by new lines so I can easily parse them. 
(Format Example: Here are the 10 movies recommended for you: [Midnight Cowboy (1969)\nLost in Translation (2003)\netc.]) 
Answer:
```

3. **A two-step prompt, which is a slightly modified version of the prompt provided in the paper[2]**

```
Candidate Set (candidate movies): {List of Candidate movies}. 
The movies I have rated highly (watched movies): {List of ten top rated movies}. 
Their genres are: {List of genres from the ten top rated movies}. 
Step 1: What features are most important to me when selecting movies (Summarize my preferences briefly)? Answer:
```

```
Candidate Set (candidate movies): {List of Candidate movies}. 
The movies I have rated highly (watched movies): {List of ten top rated movies}. 
Their genres are: {List of genres from the ten top rated movies}. 
Step 1: What features are most important to me when selecting movies (Summarize my preferences briefly)? Answer: {Response from Step 1}. 
Step 2: Can you recommend 10 movies from the Candidate Set similar to but not in the selected movies I've watched? 
Please use brackets around the movies you recommend and separate the titles by new lines so I can easily parse them. 
(Format Example: Here are the 10 movies recommended for you: [Midnight Cowboy (1969)\nLost in Translation (2003)\netc.]) 
Answer:"
```

4. **A prompt that includes a 1 sentence summary of the Wikipedia page for the movie**

```
Candidate Set (candidate movies): {List of Candidate movies}. 
The movies I have rated highly (watched movies): {List of ten top rated movies}. 
Summary of the movies I have watched: {Each Movie: 1 sentence summary of the wikipedia page for the movie} 
Can you recommend 10 movies from the Candidate Set similar to but not in the selected movies I've watched?. 
Please use brackets around the movies you recommend and separate the titles by new lines so I can easily parse them. 
(Format Example: Here are the 10 movies recommended for you: [Midnight Cowboy (1969)\nLost in Translation (2003)\netc.]) 
Answer:
```

All four prompts were compared against a baseline model, which recommended the top ten most popular movies from the training set to every user.

*Note: candidate sets are selected from top rated movies by similar users; similar users are selected by cosine similarity of a vector of a user's ratings*

## Results

The comparative performance for the prompts is shown in the below table:

|      Hit Rate | Collab Prompt | Genre Prompt | Two Step Prompt | Wiki Prompt  | Baseline    |
| ------------- | ------------- | ------------ | --------------- | ------------ | ----------- |
|         mean  | 2.500000    % | 82.000000  % | 72.500000     % | 76.000000  % | 65.000000 % |
|         std   | 5.501196    % | 17.947291  % | 31.601965     % | 26.437613  % | 20.900768 % |
|         min   | 0.000000    % | 30.000000  % | 0.000000      % | 0.000000   % | 20.000000 % |
|         max   | 20.000000   % | 100.000000 % | 100.000000    % | 100.000000 % | 100.00000 % |


*Note: hit rate is the proportion of recommended movies that the user watched.*

## How to Reproduce

1. Clone the repository: ```git clone https://github.com/nick-carroll1/LLM-Recommender-System.git```
2. Install the libraries with right version: ```pip install -r requirements.txt```
3. Get an Open AI API Key from the [website](https://platform.openai.com/api-keys) and include it in a python file called ```key.py```. The file only needs one line stating: ```OPENAI_API_KEY = "YOUR_KEY_HERE"```. This will allow all of the other files to import your API key.
4. To run all four prompts and compare performance click "Run All" on the ```comparing_prompts.ipynb``` Jupyter Notebook.


## References

[1] F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

[2] Wang, L., & Lim, E.-P. (2023). Zero-Shot Next-Item Recommendation using Large Pretrained Language Models. arXiv [Cs.IR]. Retrieved from http://arxiv.org/abs/2304.03153


Note: candidate sets are selected from top rated movies by similar users; similar users are selected by cosine similarity of a vector of a user's ratings; hit rate is the proportion of recommended movies that the user watched.
