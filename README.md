# Movie-Recommendation

### Goal

The aim of the this project is to build a system that recommends top 10 movies that are similar to movie watched by a user and are highy rated by other users as well.


### Dataset

The dataset used here is Movie Lens dataset and follwing is brief summary of dataset:

This dataset (ml-latest-small) describes 5-star rating and free-text tagging activity from [MovieLens](http://movielens.org), a movie recommendation service. It contains 100836 ratings and 3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996 and September 24, 2018. This dataset was generated on September 26, 2018.


### Technique

I have employed use of Word2Vec technique to train the model. Instead of using words and creating word embeddings, I have used Movies and created movie embeddings vector and trained it over. So, for two movies to be similar to each other, their movie embedding vector will essentially be similar. 

Let number of movies be M, and Xij be the number of users that liked both movies i and j. So, I have created a matrix that has Xij for each movie corresponding to other movie. 

My aim to get embedding vectors v1,...,vi,...,vj,...,vM for all movies such that we minimize the cost 


![Image](https://github.com/yuvraj16/Movie-Recommendation/blob/master/Loss%20Function.png)


Here 1[i̸=j] is a function that is 0 when i = j and 1 otherwise.


And lastly I have optimized the cost using Optimizer in pytorch.
