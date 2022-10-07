
## Meet the Team

Olivia: 4th Year CS Major with People & Intelligence Threads <br>
Elijah: 4th Year CS Major with People & Intelligence Threads <br>
Daniel: 4th Year CS Major with People & Intelligence Threads <br>
Liz:    3rd Year CS Major with Device & Intelligence Threads <br>
Eric:   2nd Year CS Major with Media & Intelligence Threads <br>

## Project Proposal Video: https://youtu.be/r-2KxHIcPFs
## Project Proposal

### Introduction & Background
Our team will be utilizing machine learning for the purpose of sentiment analysis on social media posts. Specifically, we will be using a database of over 200,000 reddit posts, which have been labeled for suicidal content, in order to train a machine learning model to recognize suicidal social media posts.
 Some existing research has delved into the use of Naive Bayes Machine Learning algorithm for sentiment analysis on Reddit posts to analyze them for suicidal content. One study found that the Naive Bayes model performed better than an SVM model and an ensemble model at binary-classification tasks such as distinguishing between posts flagged as either risk of suicide or no risk of suicide (Ruiz et al., 2019). Another found that the Naive Bayes model was outperformed by a Support Vector Machine model, a Random Forests model, and a Long Short Term Memory Convolutional Neural Network when classifying suicidal Reddit posts (Tadessee al., 2019). A third study found that Naive Bayes had the best Macro F1 score (a type of accuracy measurement) out of 11 different Machine Learning algorithms when performing this task (Kumar et al., 2021).

### Problem Definition
The  pre-existing research which uses Naive Bayes to classify suicide ideation makes it clear that while it can be successful in certain respects, there are still areas to be optimized such as increasing the range of application of detection models. Since increasing the quality and quantity of labeled data sets is resource intensive, it is much more effective to focus on applying transfer learning techniques to increase the flexibility of detecting suicidal posts.

### Methods
During the text pre-processing stage, the reddit post data will be tokenized, stemmed and filtered for stop words. 
To classify a post to be suicidal or non-suicidal, the multinomial Naive Bayes classifer from the scikit-learn package (MultinomialNB) will be used. Log prior, and log likelihood will be used to ensure greater computing efficiency and laplace smoothing will also be implemented to reduce the weight of words that appear zero times conditionally.
After creating a Naive Bayes model based on the labeled data, the model will be applied to other subreddits that are similar to the source of the data through transfer learning. The posterior probability will be optimized for the new unlabeled data sets through the Expectation-Maximization algorithm.
![alt text](Naive Bayes Equation Visual.png)
### Potential Results & Discussion
In order to determine the success of our project, our primary metric is percent accuracy in predicting if posts contain suicidal content. One recent study examining the accuracy of different sentiment analysis classifiers reported a 85.48% accuracy for the Naive Bayes approach, which is our starting point (Samal et al., 2017). Therefore, we will consider accuracy above 80% to be a reliable final result. This may shift if we decide to implement a different approach. Accuracy is the most important metric for us to collect because both false positives and false negatives could have drastic effects for the health and safety of those involved in the posts.

### References

Kumar, A., Trueman, T. E., & Abinesh, A. K. (2021). Suicidal risk identification in social media. Procedia Computer Science, 189, 368–373.  
 https://doi.org/10.1016/j.procs.2021.05.106

Ruiz, V., Shi, L., Quan, W., Ryan, N., Biernesser, C., Brent, D., & Tsui, R. (2019). Predicting Suicide Risk Level from Reddit Posts on Multiple Forums. 
 Proceedings of the Sixth Workshop on Computational Linguistics and Clinical Psychology. https://doi.org/10.18653/v1/w19-3020

Samal, B., Behera, A. K., & Panda, M. (2017). Performance analysis of supervised machine learning techniques for sentiment analysis. 2017 Third 
 International Conference on Sensing, Signal Processing and Security (ICSSS). https://doi.org/10.1109/ssps.2017.8071579

Tadesse, M. M., Lin, H., Xu, B., & Yang, L. (2019). Detection of Suicide Ideation in Social Media Forums Using Deep Learning. Algorithms, 13(1), 7. 
 https://doi.org/10.3390/a13010007


### Contribution Chart
| Name | Contributions |
|:-----:|:--------------:|
| Elijah Kessler | Wrote the introduction, literature review, and assisted with the problem definition. Created the GitHub page and added everyone's contributions to it |
| Olivia Mauger |     |
| Daniel Ling |       |
| Liz Huller |        |
| Eric Zhang |        |

