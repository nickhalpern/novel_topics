# Novel Topics

## Objectives and data sources

I am interested in modeling the topics that novelists write about, how the frequency of topics changes over time, and whether these topics are reflective of the reality of the period.

My data source is Project Gutenberg, which has over 40,000 English language books.  Many of these are not novels, so to filter down the universe I restricted the list to books whose authors appear on wikipedia's lists of British and American authors.

## How many topics?

I chose Non-Negative Matrix Factorization as my clustering method.  To determine the proper number of clusters, I calculated the reconstruction error for different cluster numbers.  There is not an "elbow," i.e. error seems to decline very slowly.

![NMF Reconstruction Error](https://github.com/nickhalpern/novel_topics/blob/master/images/error.png)

Instead of relying just on reconstruction error, I looked at the new topics that are created, and how novels transition as the number of clusters is increased.

The boxes in red show how many books in the earlier cluster transition as clusters are added (for example, how many move from one category where k = 4 to the analogous category where k = 5).  As clusters are added, a previous cluster is broken into two clusters and the k-1 clusters are relatively unchanged.  This is a very satisfying result, because it suggests that books are truly being grouped with similiar books.

![Evolution of topics with increasing k](https://github.com/nickhalpern/novel_topics/blob/master/images/num_topics_word_clouds.png)

