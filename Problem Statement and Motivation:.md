

### Problem Statement and Motivation:

The aim of the project was to create a recommendation system for restaurants using regression. In this case, we aimed to create a system that would predict which restaurants a given user would rate highly on Yelp.



### Introduction and Description of Data

Choices become more difficult to make when alternative options are more visible and easier to access. As consumers are inundated with an ever-increasing array of choices, systems that organize information to deliver recommendations have become increasingly valuable and widespread. Many of today’s well-known technology companies profit not from generating information themselves, but by making preexisting information accessible, comprehendible, and freely available for mass consumption. One such company is Yelp, a crowd-sourced local business review aggregator. A Yelp review is an integer rating of a business (ranging from the lowly single star to the glamourous five-star review) provided by a user, typically accompanied by some text explanation for the rating given. Yelp gathers these reviews for each business and provides users with ratings data for businesses. 

The most popular use of Yelp is to help users pick restaurants to eat at. For these users, Yelp is only as useful as it is helpful in organizing restaurants options; consequently, Yelp has a vital interest in helping users find restaurants they will enjoy. Yelp’s profits depend heavily upon advertising, advertising revenue depends upon the size and attention of the user-base, and the size and attention of the user-base depends upon helping consumers find businesses they will enjoy. Naturally, a recommendation system that can effectively predict which restaurants a given user will most enjoy would be of great use to Yelp.

Yelp released a dataset this September that contains information about local businesses about local businesses in 12 metropolitan areas across 4 countries and millions of reviews. The dataset is provided in five separate JSON files of varying sizes. Each JSON file contained information about the following topics: businesses, check­ins, reviews, tips, users and photos. For our purposes, the useful information was contained in the ‘businesses’, ‘reviews’, and ‘users’ files which contained detailed information about business attributes, full reviews, and details about user rating tendencies.

Yelp made obtaining and cleaning the dataset relatively easy. The difficulties that arise from working with this dataset result from its massive size and corresponding demands on memory and computational speed. There were total 4736897 reviews that came from 1183362 users that rated at least one of the 156639 restaurants. 

Through our preliminary EDA, we noticed some patterns that frequently occur when working with review data. Long tailed distributions. The frequency of user review counts tends to delay almost exponetially: there are lots of users who have given very few review. The review count for restaurants followed a similar, though less extreme pattern: most restaurants have less than 80 reviews, but there are some extremely popular restaurants with thousands of reviews. 



### Literature Review

#### Neighborhood Model (Non-parametric)

The basic idea of collabortive filtering is that "similar users rate similar item similarly". In this paradigm, to predict what a given user will rate a given restaurant one must determine similarity between users, restaurants, or both. We decided that two restaurants are similar if the set of users who rated both restaurants tended to rate them similarly, i.e. two restaurant are similiar if the set of common users have strongly correlated reviews values for the restaurants. Formally, we used Pearson's r as our measure of similarity. The stronger the correlation between a pair of restaurant ratings, the more similar we considered the restaurants to be.

While correlation provides an easy to interpret way of comparing similarity between items, the nature of the dataset demanded regularization so that we would properly penalize those restaurant similarities whose high correlations were merely an artifact of the limited size of their common reviewers. This was an attempt to say that, all else held equal, we deem high correlations between restaurants more trustworthy when the correlation was obtained from a greater number of common reviewers.  
$$
 
$$

$$
s_{m_i, m_j} = \frac{number(common\ reviewers)+pearson(m_i, m_j),}{number(common\ reviewers) + reg}
$$

- where increasing the value of tuning parameter $reg$ discounts the similarity between restaurants with smaller shared users more. 

In theory, one should be able to determine similarities between both users and items, but with our dataset, we found this unuseful due to the fact that most users provide relatively few reviews. As a consequence, the vast majority of users have no other users who have rated enough of the same restaurants to determine by correlation whether or not the users are similar. Furthermore, the size of the userbase is extremely large, so calculating the similarity of each user to each other user, even within a subset of a city, was computationally infeasible. 

The neighborhood approach has the advantage of being highly interpretable but suffers from several downsides. For us, the most noticable problem with finding near neighbors of items is that there tends to be very few points on which to calculate similarity because the matrix of user item review values is so sparsely populated.  Because these calcuations depend on users who have reviewed both restaurants we were only able to directly calculate similarity between restaurants with high numbers of reviews within the same city. 

#### Latent Factors Approach (Model Based)

Any real-world matrix of user item ratings tends to be sparse as a consequence of most users having reviewed a very tiny portion of the possible items. 

The key idea of any recommendation system is that the rows and columns of the review matrix are correlated: there are similar items and there are similar users. Because rows and columns are not independent, a product of non-sparse matrices of lower dimension can do a reasonably good job approximating the missing entries of the original matrix. Matrix factorization methods attempt to utilize the dependencies among both the rows and the columns of the rating matrix to estimate the entire rating matrix at one time.$^1$ 
$$
R\approx UV^T
$$
Where: 

$R$ is the rating matrix. 

U is the User affinity matrix (specifies how much each user expresses preference for each latent factor)

V is the Item affinity matrix (Specifies how much each item adheres to each latent factor).

So, each rating in the rating matring $R$ depends upon how much a user seems to express preference for latent factors and how much each item matches those preferences. 

There are several ways to achieve this factorization,  Alternating Least Squares is a popular method that affords parallelization . $ ^2$ Currently, matrix factorization methods are considered to be "state-of-the-art" for  recommendation systems.



### Modeling Approach and Project Trajectory



##### Baseline Model

The baseline predictor generates predicitions for each user item pairing in the following way. Each user  $u$ and each item $m$ has an associated bias: deviation from average. In this context, the bias of a user is the difference between the user's average rating and the average of the average user rating average; the bias of an item is the difference between a restaurant's average rating and the average rating of all restaurants. So, a very kind reviewer would have a positive bias while and a very poorly reviewed restaurant would have a negative bias. The baseline model predicts that for a given user item pair $(u, m)$ the review will be the sum of the bias terms added to the global average, $\mu$ of restaurant ratings.
$$
\hat{Y}_{um} = \mu +\hat\theta_u +\hat\gamma_m
$$
We found that this simple baseline model performs fairly well, with a training $R^2​$ score of __ and a test $R^2​$ of ____. 

##### Neighborhood approach

After developing the baseline model, we took a neighborhood approach improve our predictions. We focused on finding similarities between items rather than users. The size of the datset presented some challenges. Calculating the pairwise similiarity between two items would be prohibitively expensive so we decided to focus on restaurants in Pittsburgh. We quickly realized that finding trustworthy neighbors of restaurants with a small number of ratings was unlikely as there would be limited common users to calculate similiarity with. We assumed that restaurants with the highest number of reviews would be the restaurants with the greatest likelihood of having similar enough near neighbors to improve predictions. 

We considered using an analogous approach to find similar users. With a more complete ratings matrix, its possible to expect a rating $\hat{r}_{um}$ to be close to the ratings for the set of users most similar to $u$ on those items most similar to $ m$, however, the sparisity of the ratings matrix made this approach impractical for the vast majority of cases. 

The challenge of handling sparity in the review matrix complicated our efforts to find useful neighbors, even within the subset of popular restaurants in the Pittsburgh area. A traditional KNN regression model predicts that something will be the average of its k closest neighbors. In our case, we considered closeness to be the reciprocal of the regularized normalized Pearsons r. The problem with this approach is that with a sparse matrix, it frequenly occurs that the set of restaurant pairs that happen to have common support often are not similar enough to improve upon the baseline model. In fact, it frequently occurred that some of the closest neighbors would have negative correlations with the item of interest. Predicting that something will be rated close to something which tends to be rated far away seemed like a poor strategy. 

To avoid including neighbors less useful than predicting the average, we set a threshold for neighborhood inclusion with a fairly low standard: if the ratings of the item are negatively correlated with the ratings of our item of interest then it won't be considered for the neighborhood. We then ran a 3-fold cross validation to select the number of best number of neighbors and regularization term.

Our method for predicting the score for nearest

// Results

### Results, Conclusions, and Future Work

- Results
  - Sad!
- Conclusions
  - Treatment of main effects vitally important
- Future work
  - matrix factorization
  - potential in-app features for yelp

##### Sources

1. Recommendation Systems: The Textbook
2. https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf "Matrix Factorization Techniques for Recommender Systems"