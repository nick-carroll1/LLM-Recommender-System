# LLM Recommender System

This repository uses Large Language Model (LLM) to create a movie recommender system and compares the recommender's performance across four prompt templates.

<img width="1000" alt="image" src="https://github.com/nick-carroll1/LLM-Recommender-System/assets/110933007/cfa47949-bade-4891-8af2-e3d109e24f26">

## Project Overview

This project aims to build a movie recommendation system by leveraging the OpenAI API, integrating Large Language Models (LLM) to enhance the recommendation accuracy through innovative prompt engineering. We specifically focus on the application of four distinct prompt templates to evaluate the performance of movie recommendations. By conducting a comparative analysis of these templates, we aim to identify the most effective approach for generating personalized movie recommendations.

## Methodology
![image](https://github.com/nick-carroll1/LLM-Recommender-System/assets/110933007/73024e3a-4c72-450c-8e4f-daade755fa08)

### Data
In this project, we use the MovieLens 100K dataset. Collected by the GroupLens Research Project at the University of Minnesota[1]. MoiveLens 100K is a widely-used benchmark dataset in the field of recommender systems. Released in April 1998, it comprises 100,000 movie ratings, ranging from 1 to 5, provided by 943 users on 1,682 movies. Each user has rated at least 20 movies, ensuring a reasonable level of engagement and data density for analysis. The dataset can be found here: https://grouplens.org/datasets/movielens/

The original dataset was provided in MATLAB format. We converted it using the [```src/prepare_data.ipynb```](https://github.com/nick-carroll1/LLM-Recommender-System/blob/main/src/prepare_data.ipynb) notebook and stored the processed data in ```processed_movie100k.csv``` for subsequent use. We also implemented a 70/30 [train-test split](https://github.com/nick-carroll1/LLM-Recommender-System/blob/main/src/test_train_split.ipynb) to facilitate model training and evaluation. For prompt engineering, we enriched the dataset by appending a one-sentence summary from Wikipedia to each movie to provide additional context. These movie summaries, along with their corresponding titles, are compiled in ```movie_wiki.csv```.

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

**Candidate Set**: Since there are 1683 movies in the MovieLens 100k dataset, it's impratical to feed them all to the prompt. Therefore, we generate a candidate set of potential recommendations by identifying users with similar tastes and selecting movies that they have rated highly. Specifically, candidate sets are selected from top rated movies by similar users, who are selected by cosine similarity of a vector of a user's ratings. Based on Wang & Lim's research[2], which suggests the best number of similar users and movies to consider, we use ratings from $10$ users who are most like the person we're recommending movies to, and from these, we choose $20$ movies to recommend as our candidate set.

**Baseline Model**: The baseline model recommended the top 10 movies that have been most frequently watched in the training dataset to every user, assuming their popularity would broadly appeal to a general audience. All four prompt recommmendations were compared to this baseline model.



## Results

Our evaluation metric is the hit ratio at 10 (HR@10). This measures the proportion of the 10 movies recommended by the LLM that users have actually watched. 

To enhance the robustness of our analysis, we accounted for the non-deterministic nature of LLM recommendations by running the notebook three times and recording the average result from each trial.

Here are the comparative performance for the prompts:

|      HR@10    | Collab Prompt | Genre Prompt | Two Step Prompt | Wiki Prompt  | Baseline    |
| ------------- | ------------- | ------------ | --------------- | ------------ | ----------- |
|         mean  | 4.0357      % | 76.5172    % | 68.9755       % | 71.9439    % | 67.7444   % |
|         std   | 7.0931      % | 21.6064    % | 29.8969       % | 25.1883    % | 23.2775   % |


The collaborative filtering prompt lags behind with a mean HR@10 of approximately 4.04%, suggesting it may not be capturing user preferences as effectively as other methods. This is largely because candidate set is not provided in this case. Without this candidate set, the collaborative filtering approach is possibly too broad or misaligned with the user's specific preferences, resulting in recommendations that the users are less likely to have watched.

In contrast, the three other prompts show good performance with a candidate set provided, all exceeding the popularity-based baseline. The genre-based prompt excells with a robust 76.52% mean HR@10, underscoring the importance of genre preferences in driving user satisfaction with recommendations. Similarly, the Wikipedia-summarized prompt also performs well, with a mean HR@10 of 71.94%, indicating that incorporating content summaries from Wikipedia contributes positively to the recommendation quality. The two-step prompt achieves a mean HR@10 of 68.98%, which, while effective, suggests that additional steps in the recommendation process do not necessarily translate to a proportional increase in performance. This is further illustrated by instances where the first step failed to define user preferences clearly, giving answers like "It is difficult to summarize your preferences based on the given information". More details and context should be provided to enhance the recommendation accuracy.

Interestingly, the baseline model, despite its simplicity, achieved a mean HR@10 of roughly 67.74%, reinforcing the idea that general popularity is a strong indicator of potential interest.


## How to Reproduce

1. Clone the repository: ```git clone https://github.com/nick-carroll1/LLM-Recommender-System.git```
2. Install the libraries with right version: ```pip install -r requirements.txt```
3. Get an Open AI API Key from the official [website](https://platform.openai.com/api-keys) and include it in a python file called ```key.py```. The file only needs one line stating: ```OPENAI_API_KEY = "YOUR_KEY_HERE"```. This will allow all of the other files to import your API key.
4. To run all four prompts and compare performance click "Run All" on the [```src/comparing_prompts.ipynb```](https://github.com/nick-carroll1/LLM-Recommender-System/blob/main/src/comparing_prompts.ipynb) Jupyter Notebook.


## References

[1] F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

[2] Wang, L., & Lim, E.-P. (2023). Zero-Shot Next-Item Recommendation using Large Pretrained Language Models. arXiv [Cs.IR]. Retrieved from http://arxiv.org/abs/2304.03153
