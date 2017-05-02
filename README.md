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

I determined that five topics was sufficiently granular to divide the universe of novels without making overly fine distinctions.  The chart below shows the vocabulary of each novel embedded into 2 dimensions with TSNE.

![Novel topics](https://github.com/nickhalpern/novel_topics/blob/master/images/topics_and_example_novels.png)

## How do novel topics change over time?

I found that the frequency of topic varied significantly over the period studied.  In particular, the drawingroom category - books about society, marriage/romance, and politics, was very popular in the early 19th century and was gradually joined by books about city life and the American west.

![Frequency Over Time](https://github.com/nickhalpern/novel_topics/blob/master/images/cluster_freq_over_time.png)

The example of books about the American West is a compelling example that novel topics don't reflect the reality of the current period.  As people settled the West, encouraged by the Homestead Act in 1862, there was an increasing number of texts about cowboys and shootouts.  After the US declared the end of the frontier in the census of 1890, novels about the frontier continued to increase!  (It is also interesting to think about all the Western movies in the 1950s and 60s.)

![Wild West US](https://github.com/nickhalpern/novel_topics/blob/master/images/cluster_freq_over_time_us_ww.png)
