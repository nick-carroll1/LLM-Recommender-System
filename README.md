# LLM-Recommender-System
This uses pre-trained LLM's for recommender systems

Yuanjing's update on Dec 3:

```prepare_data.ipynb```: merge *user* dataset and *movie* dataset to *processed_movie100k.csv*

```UF_prompt_engineering.ipynb```:  
1. Implement User-Filtering principle in the paper: candidate set is $n_s$ top popular movies from $m$ similar users (measured by cosine similarity)
2. 2-step Prompt Engineering:
  <img width="850" alt="image" src="https://github.com/nick-carroll1/LLM-Recommender-System/assets/110933007/6d544af4-77e4-4b11-a3b6-abbc9b59a322">

3. Tested first 100 users with $m=10$, $n_s=20$. Average hit rate is 0.509
