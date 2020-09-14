# Recommendation Systems on Yelp Data
This project contains the review data from the original Yelp review dataset with some filters, such as the condition: “state” == “CA”. 80% of the data is used for training, 10% of the data for testing, and 10% of the data as the blind dataset.

Task 1: Min-Hash + LSH

Implemented the Min-Hash and Locality Sensitive Hashing algorithms with Jaccard similarity1 to find similar business pairs ifrom the Trained Data.
Results: Accuracy >= 0.8. The execution time is less than 200 seconds.

Task 2: Content-based Recommendation System

Built a content-based recommendation system by generating profiles from review texts for users and businesses in the train review set. Then used the system/model to predict if a user prefers to review a given business, i.e., computing the cosine similarity between the user and item profile vectors.
Results : Accuracy more than 80% on test and blind datasets, with execution time less than 300 seconds on test and blind dataset.

Task 3: Collaborative Filtering Recommendation System

Built a collaborative filtering recommendation systems with train reviews and use the models to predict the ratings for a pair of user and business. 
Case1: Item-based CF recommendation system 
During Training process, Build a model by computing the Pearson correlation for the business pairs that have at least 3 co-rated users, containing the valid pairs that have positive Pearson similarity. During the predicting process, the model to predict the rating for a given pair of user and business, using most N business neighbors that are most similar to the target business for prediction.
Results: RMSE value 0.84 on test data and 0.85 on blind data, with execution time less than 100 seconds on test and blind dataset.

case2: User-based CF recommendation system with Min-Hash LSH
During Training process, if the number of potential user pairs is too large to compute in memory, you could combine the Min-Hash and LSH algorithms in your user-based CF recommendation system. Identified user pairs who are similar using their co-rated businesses without considering their rating scores. This process reduces the number of user pairs needed to compare for the final Pearson correlation score. And then computed the Pearson correlation for the user pair candidates that have Jaccard similarity >= 0.01 and at least 3 co-rated businesses. During the predicting process, the model to predict the rating for a given pair of user and business, using most N business neighbors that are most similar to the target business for prediction.
Results: RMSE value 0.92 on test data and 0.94 on blind data, with execution time less than 100 seconds on test and blind dataset.
