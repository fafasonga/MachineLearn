YOUR TASK:
Implement a Naive Bayes classifier which can handle mixed input data types

YOUR DATA:
Credit approval dataset (you can read about it here - http://archive.ics.uci.edu/ml/datasets/Credit+Approval)
Contains both categorical and numerical features.
The data has been cleaned from missing inputs. No data pre-processing is required from your side.

INSTRUCTIONS:
You have a template file and four methods which you need to fill, main of them are fit() and predict().
At the end of the template you will find several calls to your Naive Bayes algorithm:
    - with categorical features only
    - with all (mixed) features

We suggest the following order of implementation:
1) First make sure that your algorithm can handle categorical features.
Calculate conditional probability of each attribute category based on training data and use it for predictions.
Compare accuracy you are getting with a benchmark.

2) Then add Laplace add-one smoothing (https://en.wikipedia.org/wiki/Additive_smoothing)
to handle the case when we have an unobserved category in the test data.
Compare the accuracy with and without Laplace smoothing.

3) Now add support for numerical features. Conditional probability of numerical features should be calculated based on
Probability Density Function assuming normal distribution. Use auxiliary estimate_mean_and_stdev() and calc_probability()
methods for these purposes.

4) Finally, do some analysis. Did your accuracy improve after you considered numerical features as well?
Think why it is so. What can you do to improve it? Try it.
Write down your thoughts in the comments at the end of the template
