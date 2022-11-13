
## Meet the Team

Liv: 4th Year CS Major with People & Intelligence Threads <br>
Eli: 4th Year CS Major with People & Intelligence Threads <br>
Dan: 4th Year CS Major with People & Intelligence Threads <br>
Liz:    3rd Year CS Major with Device & Intelligence Threads <br>
Ric:   2nd Year CS Major with Media & Intelligence Threads <br>

## Project Proposal Video: https://youtu.be/r-2KxHIcPFs
### Gantt Chart Drive Link: https://docs.google.com/spreadsheets/d/1wRbEzjqjg5izmsQLVldgokHcVH765NBH/edit?usp=sharing&ouid=106124061592007882318&rtpof=true&sd=true
## Project Proposal


**Introduction & Background**

Our team will be utilizing machine learning for the purpose of sentiment analysis on social media posts. Specifically, we will be using a database of over 300,000 reddit posts, which have been labeled for suicidal content, in order to train a machine learning model to recognize suicidal social media posts. Our dataset can be found on Kaggle here: https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch 

Some existing research has delved into the use of Naive Bayes Machine Learning algorithm for sentiment analysis on Reddit posts to analyze them for suicidal content. One study found that the Naive Bayes model performed better than an SVM model and an ensemble model at binary-classification tasks such as distinguishing between posts flagged as either risk of suicide or no risk of suicide (Ruiz et al., 2019). Another found that the Naive Bayes model was outperformed by a Support Vector Machine model, a Random Forests model, and a Long Short Term Memory Convolutional Neural Network when classifying suicidal Reddit posts (Tadessee al., 2019). A third study found that Naive Bayes had the best Macro F1 score (a type of accuracy measurement) out of 11 different Machine Learning algorithms when performing this task (Kumar et al., 2021).

**Problem Definition**

The pre-existing research which uses Naive Bayes to classify suicide ideation makes it clear that while it can be successful in certain respects, there are still areas to be optimized such as increasing the range of application of detection models. Since increasing the quality and quantity of labeled data sets is resource intensive, it is much more effective to focus on applying transfer learning techniques to increase the flexibility of detecting suicidal posts.

**Data Collection**

The dataset we utilized for our project utilized approximately 350,000 Reddit posts on the “Suicide Watch” and “depression” subreddits. All posts made to “Suicide Watch” from its creation on December 16, 2008 until January 2, 2021 were collected, while the posts collected which were made to “depression” came from January 1, 2009 until January 2, 2021. These posts were collected using Pushshift API.

**Methods**


*Pre-Processing* 

During the text pre-processing stage, the reddit post data will be tokenized, stemmed and filtered for stop words. Specifically, we included the following pre-processing methods before conducting any further analysis: Remove Punctuation, Convert Contractions/Abbreviations/Acronyms into Words, Remove Emojis and URLs, Tokenization, Stemming, Stop Words (removal), and Lowercase Text (convert to). 

The stemming method combines like terms together, which is a form of dimension reduction. For example, words such as “running” and “run” have similar meanings, but just in different forms. Stemming would group these words together into the same token. By reducing the variety of words that have the similar meaning, the quality of the data is significantly improved upon since the weight of the tokens will become more representative of the meaning of the words rather than just raw word frequency. 

To classify a post to be suicidal or non-suicidal, the multinomial Naive Bayes classifier from the scikit-learn package (MultinomialNB) will be used. Log prior, and log likelihood will be used to ensure greater computing efficiency and laplace smoothing will also be implemented to reduce the weight of words that appear zero times conditionally. After creating a Naive Bayes model based on the labeled data, the model will be applied to other subreddits that are similar to the source of the data through transfer learning. The posterior probability will be optimized for the new unlabeled data sets through the Expectation-Maximization algorithm. 

*Implementation of Naive Bayes and TD-IDF*

Once the data from reddit was preprocessed and tokenized, the data frame was split into suicidal and non-suicidal comments using the provided labels. A frequency table was generated for each to assist in calculating the Naive Bayes probability; however after examining a few of the frequent words within each frequency table, some words like “much” and “because” appear extremely frequently. If the Naive Bayes were to be run over the data, those words would be given too much weight due to its frequent nature and lackluster correlation with the labels. Thus the term frequency–inverse document frequency (TD-IDF) was used instead of the actual frequency of the terms. 

![Equations](equation.png) 


By multiplying the term frequency with the inverse document frequency, if the term ( occurs frequently within the given document (d) and there is low document frequency in the overall dataset the TF-IDF weight will be high. This helps give more weight to words that have a significant impact on the document’s classification but does not appear as frequently. 
After the TF IDF score was calculated, the data was split into 80% training data and 20% testing data. The training data was used to train the Naive Bayes classifier and the classifier was then run on the testing data and compared to the ground truth labels. 

**Results & Discussion**

In order to determine the success of our project, our primary metric is percent accuracy in predicting if posts contain suicidal content. One recent study examining the accuracy of different sentiment analysis classifiers reported a 85.48% accuracy for the Naive Bayes approach, which is our starting point (Samal et al., 2017). Accuracy is the most important metric for us to collect because both false positives and false negatives could have drastic effects for the health and safety of those involved in the posts. Our approach was able to achieve 88% accuracy, which was measured by comparing the predicted labels to the ground truth labels from the testing dataset. However, there are still many more areas that we could increase the accuracy of the Naive Bayes method, such as trying alternative preprocessing methods, and different frequency weightings 

*Flaws in Dataset and Preprocessing*

There are several flaws with this dataset which limit the efficacy of our model. Due to the nature of social media posts, this dataset is rife with the use of slang, emojis, URLs, acronyms, gibberish, and other confounding factors which makes the data difficult to process. While we were able to account for many of these aspects with our data preprocessing, we were likely unable to entirely account for all these issues, meaning our cleaned data which the model was trained with may have still been somewhat flawed and thus could have yielded inaccuracies. Furthermore, there were certain flaws with the collection of data which resulted in some potentially meaningful words being effectively lost. We believe that when collecting data, the creators of the dataset did not put spaces between the last word of a post’s title and the first word of the post’s content, resulting in words being fused in several posts. Therefore, we may be losing some highly significant words, like “suicidal,” because a post’s title ended with the word and had more text within the body of the Reddit post. In most of these cases our data preprocessing is likely recognizing these fused words as gibberish and removing them, which means we could be losing significant data. 

Although there is a possibility of losing significant words due to the spacing error from the data set, the overall impact is minimal because the merged word is unlikely to repeat multiple times in other comments. The other issues with the data should only serve as noise and not have a significant impact on accuracy.


*Ways to Optimize Naive Bayes*

Because Naive Bayes is a relatively simple approach, we are going to utilize a few different methods to optimize it. One potential way of optimizing the approach is to remove the correlated features. In naive bayes, the highly correlated features are counted twice. This double counting leads to overestimating the importance of those features. Another way of optimizing naive bayes is to eliminate the zero observations problem. If the model comes across a feature that wasn’t in the training set, it gets a probability of 0 which ends up turning other values into 0 when multiplying. Finally, we will use log probabilities to avoid working with very small numbers that are difficult to store precisely.  These three optimizations should help improve the accuracy of the model.

**Bibliography**

Kumar, A., Trueman, T. E., & Abinesh, A. K. (2021). Suicidal risk identification in social media. Procedia Computer Science, 189, 368–373. https://doi.org/10.1016/j.procs.2021.05.106

Ruiz, V., Shi, L., Quan, W., Ryan, N., Biernesser, C., Brent, D., & Tsui, R. (2019). Predicting Suicide Risk Level from Reddit Posts on Multiple Forums. Proceedings of the Sixth Workshop on Computational Linguistics and Clinical Psychology. https://doi.org/10.18653/v1/w19-3020

Samal, B., Behera, A. K., & Panda, M. (2017). Performance analysis of supervised machine learning techniques for sentiment analysis. 2017 Third International Conference on Sensing, Signal Processing and Security (ICSSS). https://doi.org/10.1109/ssps.2017.8071579

Tadesse, M. M., Lin, H., Xu, B., & Yang, L. (2019). Detection of Suicide Ideation in Social Media Forums Using Deep Learning. Algorithms, 13(1), 7. https://doi.org/10.3390/a13010007




### Updated Contribution Chart

| Name | Contributions |
|:-----|:--------------|
| Elijah Kessler | For the midterm: Completed and coded 2 pre-processing tasks (Remove Punctuation, Remove Acronyms) and processed the dataset from original csv into Python dictionary for pre-processing. Wrote the "Flaws in Dataset" section of the midterm report. For the proposal: Wrote the introduction, literature review, and assisted with the problem definition. Created the GitHub page and added everyone's contributions to it. |
| Olivia Mauger | For the midterm: Completed and coded 3 pre-processing tasks (Remove Emojis, URLs, and to Lowercase), edited and helped with discussion on pre-processing errors & For the proposal: Helped with creating GitHub page and styled it using Markdown, recorded video and uploaded to YouTube, wrote Potential Results & Discussion section with Liz    |
| Daniel Ling |Researched Naive Bayes and pre-processing for sentiment analysis, Met TA to check on project idea, Researched Naives Bayes Transfer Learning, Wrote the Methods Section, wrote problem statment with Eli|
| Liz Huller | For the midterm: Completed and coded 3 pre-processing tasks (Stemming, Stop words, tokenization). Wrote the ways of optimizing naive bays section of the midterm report. For the proposal: Researched Naive Bayes and other methods for sentiment analysis, worked with Olivia on Potential Results & Discussion section, helped plan video, and added dates/assigned names to the Gantt Chart |
| Eric Zhang (no longer in class) | Found the labeled data set for suicidal reddit posts. Researched the Naive Bayes Model and contributed to the methods section. Created visuals for the video presentation.  

