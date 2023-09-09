# Sentiment Analysis of US Airline Tweets

![](./images/airplane.jpeg)

## Introduction
This project focuses on sentiment analysis of tweets related to US airlines. The dataset used for this analysis is the [Twitter US Airline Sentiment dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) from Kaggle. The goal is to classify tweets as either positive or negative sentiment.

## Libraries Used
The project utilizes various Python libraries including:
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- NLTK
- Gensim

## EDA and Preprocessing
Exploratory Data Analysis (EDA) is conducted to understand the dataset. This involves checking for null values, obtaining descriptive statistics, and visualizing the distribution of sentiments. Text preprocessing techniques such as removing special characters, tokenizing, removing stopwords, and lemmatization are applied.

#### Distribution of Sentiment

![](./images/distribution_sentiment.png)

As we can see from the chart there is a much greater frequency of negative reviews comparred to postive reviews.

#### Distribution of Review Lengths by Sentiment

![](./images/distr_review_lengths.png)

According to this histogram the curve for the Negative review class is skewed to the left where the majority of negative reviews tend to be in the longer range at 80-100 words in length. The postive review class is practically uniformaly distrubuted where the length of each positive review has the same probability of being submitted.


#### Word Cloud of most frequent words

![](./images/word_cloud.png)

## Classical ML Approach
### Data Preparation
The text data is vectorized using TF-IDF vectorization. The dataset is split into training and testing sets.

### Random Forest and XGBoost
A Random Forest Classifier and an XGBoostClassifier are trained and evaluated for sentiment classification. Hyperparameter tuning is performed using GridSearchCV for the XGBoost model.

### Evaluation and Results
The models are evaluated using metrics such as accuracy, ROC curve, and confusion matrix. Visualizations are generated to illustrate model performance.

**Base Model Performance**
* Accuracy - Random Forest: 90.18%
* Accuracy - XGBoostClassifier: 90.33%

**Optimized XGBoost Performance and Evaluation**
* Accuracy - XGBoostClassifier: 90.50% (not much improvement)
* AUC - XGBoostClassifier: 0.9404

**Confusion Matrix**

![](./images/confusion_matrix.png)

**ROC AUC Curve**

![](./images/roc_auc_curve.png)


### XGBoost Model Predictions

#### Positive Texts

- "co flight great http jetblue view"
- "ago airport busiest club month none one thank united weird"
- "americanair followed great thanks"
- "got thanks united"
- "a320 airbus bostonlogan co gate http jetblue jetbluesofly morning pulling sunrise"

#### Negative Texts

- "cancelled charged flight flightled unbelievable usairways"
- "actually agent awful booking flight gate line missed problem reflight rude standing sti united waiting"
- "200 aa additional americanair asleep back called everything fee flightr gr8 hr late standard took worry"
- "2015 airline brother country cross cup flight help lost luggage panamerican please sunday united"
- "flow hard keeping making positive smoothly sure thing usairways work"



## Deep Learning Approach
### Tokenization and Embedding
The text data is tokenized and sequences are created. Word embeddings using Word2Vec are performed.

### LSTM Model
A deep learning model is defined with embedding layers and LSTM units for sentiment analysis. The model is compiled and one-hot encoding is applied to the labels.

### Model Training
The LSTM model is trained and evaluated. Loss, accuracy, and AUC metrics are plotted across epochs. A ModelCheckpoint callback is used to save the best model.

### Evaluation and Results
The best model is loaded and evaluated on the test set. Predictions are made, and positive/negative texts are identified based on these predictions.

**Model Performance**
* Accuracy - Best LSTM: 87.63%
* AUC - Best LSTM: 0.9462

**Loss and Metric Plots (50 epochs)**

**Training vs Testing Loss**

![](./images/best_model_loss.png)

The training and testing losses begin to diverege around 20 or so epochs, this is where the model begins overfitting.

**Training vs Testing Accuracy**

![](./images/best_model_acc.png)

Like with the loss the training and testing accuracies begin to diverege around 20 or so epochs, this is where the model begins overfitting.

**Training vs Testing AUC**

![](./images/best_model_auc.png)

As with both the loss and the accuracy training and testing auc scores begin to diverege around 20 or so epochs, this is where the model begins overfitting. This is also where our checkpoint saves the best model - where the test auc performs the best. 


### LSTM Model Predictions

#### Positive Texts

- "jetblue great flight great view http co yxn00pnoav"
- "united thank one month ago none weird club one busiest airport u"
- "americanair great thanks followed"
- "united got thanks"
- "virginamerica look like broken link asset http co oardjjgrrd"

#### Negative Texts

- "usairways charged flight cancelled flightled unbelievable unheard"
- "united actually gate agent rude standing line waiting reflight booking problem missed flight sti awful"
- "americanair worry called back 4 hr late flightr asleep took additional 200 fee aa standard everything gr8"
- "united brother luggage lost copa airline flight 635 competing sunday 2015 panamerican cross country cup please help"
- "usairways work hard making sure thing flow smoothly keeping positive"

