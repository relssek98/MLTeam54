
## Meet the Team

Olivia: 4th Year CS Major with People & Intelligence Threads
Elijah: 4th Year CS Major with People & Intelligence Threads

## Project Proposal

### Introduction & Background
Our team will be utilizing machine learning for the purpose of sentiment analysis on social media posts. Specifically, we will be using a database of over 200,000 reddit posts, which have been labeled for suicidal content, in order to train a machine learning model to recognize suicidal social media posts.
 Some existing research has delved into the use of Naive Bayes Machine Learning algorithm for sentiment analysis on Reddit posts to analyze them for suicidal content. One study found that the Naive Bayes model performed better than an SVM model and an ensemble model at binary-classification tasks such as distinguishing between posts flagged as either risk of suicide or no risk of suicide (Ruiz et al., 2019). Another found that the Naive Bayes model was outperformed by a Support Vector Machine model, a Random Forests model, and a Long Short Term Memory Convolutional Neural Network when classifying suicidal Reddit posts (Tadesseet al., 2019). A third study found that Naive Bayes had the best Macro F1 score (a type of accuracy measurement) out of 11 different Machine Learning algorithms when performing this task (Kumar et al., 2021).

### Problem Definition
The  pre-existing research which uses Naive Bayes to classify suicide ideation makes it clear that while it can be successful in certain respects, there are still areas to be optimized such as increasing the range of application of detection models. Since increasing the quality and quantity of labeled data sets is resource intensive, it is much more effective to focus on applying transfer learning techniques to increase the flexibility of detecting suicidal posts.

### Methods
During the text pre-processing stage, the reddit post data will be tokenized, stemmed and filtered for stop words. 
To classify a post to be suicidal or non-suicidal, the multinomial Naive Bayes classifer from the scikit-learn package (MultinomialNB) will be used. Log prior, and log likelihood will be used to ensure greater computing efficiency and laplace smoothing will also be implemented to reduce the weight of words that appear zero times conditionally.
After creating a Naive Bayes model based on the labeled data, the model will be applied to other subreddits that are similar to the source of the data through transfer learning. The posterior probability will be optimized for the new unlabeled data sets through the Expectation-Maximization algorithm.

### Potential Results & Discussion
In order to determine the success of our project, our primary metric is percent accuracy in predicting if posts contain suicidal content. One recent study examining the accuracy of different sentiment analysis classifiers reported a 85.48% accuracy for the Naive Bayes approach, which is our starting point (Samal et al., 2017). Therefore, we will consider accuracy above 80% to be a reliable final result. This may shift if we decide to implement a different approach. Accuracy is the most important metric for us to collect because both false positives and false negatives could have drastic effects for the health and safety of those involved in the posts.

### References
