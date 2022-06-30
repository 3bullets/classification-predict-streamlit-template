"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
import altair as alt
import matplotlib.pyplot as plt
from PIL import Image




# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/vector3.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file



# Load your raw data
raw = pd.read_csv("resources/train.csv")
df = pd.read_csv("df_train.csv")
bf_world = pd.read_csv("bf_world.csv")
bf_news = pd.read_csv("bf_news.csv")
bf_anti = pd.read_csv("bf_anti.csv")
bf_pro = pd.read_csv("bf_pro.csv")
bf_neutral = pd.read_csv("bf_neutral.csv")


bf_world = pd.read_csv("bf_world.csv")
bf_pro = pd.read_csv("bf_pro.csv")
bf_anti = pd.read_csv("bf_anti.csv")
bf_news = pd.read_csv("bf_news.csv")
bf_neutral = pd.read_csv("bf_neutral.csv")

bf_world_list = []
for i in bf_world["hashtags"]:
    bf_world_list.append(i)

bf_news_list = []
for i in bf_news["hashtags"]:
    bf_news_list.append(i)

bf_neutral_list = []
for i in bf_neutral["hashtags"]:
    bf_neutral_list.append(i)

bf_anti_list = []
for i in bf_anti["hashtags"]:
    bf_anti_list.append(i)
    
bf_pro_list = []
for i in bf_pro["hashtags"]:
    bf_pro_list.append(i)

bf_world_count_list = []
for i in bf_world["count"]:
    bf_world_count_list.append(i)

bf_news_count_list = []
for i in bf_news["count"]:
    bf_news_count_list.append(i)

bf_neutral_count_list = []
for i in bf_neutral["count"]:
    bf_neutral_count_list.append(i)

bf_anti_count_list = []
for i in bf_anti["count"]:
    bf_anti_count_list.append(i)
    
bf_pro_count_list = []
for i in bf_pro["count"]:
    bf_pro_count_list.append(i)

bf_world_dict = {}
for i in range(20):
    bf_world_dict[bf_world_list[i]] = bf_world_count_list[i]

bf_pro_dict = {}
for i in range(20):
    bf_pro_dict[bf_pro_list[i]] = bf_pro_count_list[i]

bf_anti_dict = {}
for i in range(20):
    bf_anti_dict[bf_anti_list[i]] = bf_anti_count_list[i]
    
bf_news_dict = {}
for i in range(20):
    bf_news_dict[bf_news_list[i]] = bf_news_count_list[i]
    
bf_neutral_dict = {}
for i in range(20):
    bf_neutral_dict[bf_neutral_list[i]] = bf_neutral_count_list[i]


f = df[df['sentiment'] == 0]
neutral = f["text_length"]
g = df[df['sentiment'] == 1]
pro = g["text_length"]
h = df[df['sentiment'] == -1]
anti = h["text_length"]
i = df[df['sentiment'] == 2]
news = f["text_length"]

sentiment = ["Pro", "News", "Neutral", "Anti"]
sentiment2 = ["Pro", "Anti", "News", "Neutral"]
colors2 = ['#0000FF', '#00FF00', '#FFFF00', '#FF00FF']
num = [1, 2, 3, 4]
sentiment_count = [8530, 3640, 2353, 1296]
sentiment_dict = {"Pro": 8530,"News": 3640, "Neutral": 2353, "Anti": 1296}
sentiment2_dict = {"Pro": pro,"Anti": anti, "News": news, "Neutral": neutral}
list1 = ['climate','change','global', 'warming', 'trump', 'believe', 'amp', 'doesn', 'world', 'real', 'going', 'people',
        'just', 'president', 'epa', 'new', 'don', 'fight', 'like', 'science', 'says', 'die', 'scientists', 'hoax', 'donald',
        'say', 'think', 'thinking', 'husband', 'isn']
list2 = [12659,12643,3792,3534,2298,1158,939,803,730,720,670,618,608,565,550,542,505,474,473,442,430,421,416,394,391,370,
        366,355,317,303]
word_dict = {'climate':12659,'change':12643,'global':3792,'warming':3534,'trump':2298,'believe':1158,'amp':939,'doesn':803,'world':730,
        'real':720,'going':670,'people':618,'just':608,'president':565,'epa':550,'new':542,'don':505,'fight':474,'like':473,'science':442,
        'says':430,'die':421,'scientists':416,'hoax':394,'donald':391,'say':370,'think':366,'thinking':355,'husband':317,'isn':303}


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Team", "Prediction", "Information", "EDA", "Classification Models", "Feature Engineering", "Model Evaluation", "Conclusion"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Model Evaluation":
                if st.checkbox('Show model evaluation'):
                    st.info("Perfomance metrics")
                
                    st.markdown("# Model Evaluation")
                    image = Image.open('resources/perfomance.jpg')
                    st.image(image,width=700)
                    
                    st.markdown("### Evalution of DecisionTreeClassifier")
                    st.markdown("**Confusion matrix for DecisionTreeClassifier**")
                    image = Image.open('resources/confusion.jpg')
                    st.image(image,width=700)
                    st.markdown("**Key Observations**")
                    image = Image.open('resources/metrics.jpg')
                    st.image(image,width=700)
                    st.markdown("**Decision Tree Confusion Matrix**")
                    image = Image.open('resources/heatmap.jpg')
                    st.image(image,width=700)

                    st.markdown("**Key Observations**")
                    st.markdown("A Classification report is used to measure the quality of predictions from a classification algorithm.")
                    st.markdown("The confusion matrix heatmap shows the model's  ability to classify positive samples, each class achieving a recall score of:")
                    st.markdown("+ Anti Climate Change : 0.28")
                    st.markdown("+ Neutral : 0.33")
                    st.markdown("+ Pro : 0.78")
                    st.markdown("+ News : 0.66 We see that  most of the tweets are incorrectly classified as `pro` sentiment classes, with 44% and 48% of `anti` and `neutral` sentiment classes respectively incorrectly classified to belong to `Pro` sentiment class.")
                    
                    image = Image.open('resources/imgs/print accuracy.jpg')
                    st.image(image,width=700)
                    st.markdown("Overal the Decision Tree classifier did a poor job at classifying the sentiments, achieving the accuracy score of 0.6498 and a weighted F1 score of 0.6403")
                    st.markdown("### Evalution of RandomForestClassifier")
                    image = Image.open('resources/imgs/confusion2.jpg')
                    st.image(image,width=700)
                    image = Image.open('resources/imgs/metrics2.jpg')
                    st.image(image,width=700)
                    image = Image.open('resources/imgs/heatmap2.jpg')
                    st.image(image,width=700)
                    st.markdown("**Key Observations**")
                    st.markdown("A Classification report is used to measure the quality of predictions from a classification algorithm.")
                    st.markdown("The confusion matrix heatmap shows the model's  ability to classify positive samples, each class achieving a recall score of:")
                    st.markdown("+ Anti Climate Change : 0.19")
                    st.markdown("+ Neutral : 0.31")
                    st.markdown("+ Pro : 0.92")
                    st.markdown("+ News : 0.62")
                    st.markdown("Random Forest classifier did a bad job at correctly classifying sentiment class `anti` and `neutral` incorrectly classifying them as `pro` sentiment class 66% and 68% of the time respectively.")
                    image = Image.open('resources/imgs/print accuracy2.jpg')
                    st.image(image,width=700)
                    st.markdown("The overall accuracy and weighted f1 score for the Random Forest Classifier is better compared to that of the Decision Tree classifier, making the Random Forest classifier the best model this far")
                    st.markdown("### Evaluation of LinearSVClassifier")

                    image = Image.open('resources/imgs/confusion3.jpg')
                    st.image(image,width=700)
                    image = Image.open('resources/imgs/metrics3.jpg')
                    st.image(image,width=700)
                    image = Image.open('resources/imgs/heatmap3.jpg')
                    st.image(image,width=700)
                    st.markdown("**Key Observations**")
                    st.markdown("A Classification report is used to measure the quality of predictions from a classification algorithm.")
                    st.markdown("The confusion matrix heatmap shows the model's  ability to classify positive samples, each class achieving a recall score of:")
                    st.markdown("+ Anti Climate Change : 0.44")
                    st.markdown("+ Neutral : 0.49")
                    st.markdown("+ Pro : 0.82")
                    st.markdown("+ News : 0.78")
                    st.markdown("We see that that the LinearSVC did a very job at classifying positive samples, the biggest concern here is that 32% and 39% of `anti` and `neutral` sentiment classes respectively were incorrectly classified as `Pro` sentiment class")
                    st.markdown("- We see that the LinearSVC model did a far better job at classifiying `Pro` and `News` sentiment classes compared to `Decision Tree` and `RandomForest` models  with both classes achieving an f1 score of 0.85 and 0.81 respectively")
                    st.markdown("- The LinearSVC model also did a far better job at classifying `Anti` sentiment class comapred to both the Decision tree and the Randrom Forest.")
                    st.markdown("- There was a slight improvement in the classification of `neutral` tweets with the LinearSVC, which is by far overshadowed by the improvements we see in other sentiments classes")
                    st.markdown("- The LinearSVC has done a better job overall in classifying the sentiments, we see that `Anti` and `Neutral` sentiments have almost the same score, same applies with `Pro` and `News` sentiments which is consistent with the distribution of the data between the sentiment classes.")
                    image = Image.open('resources/imgs/print accuracy3.jpg')
                    st.image(image,width=700)
                    st.markdown("The LinearSVC is the best we've seen this far achieving an accuracy score of 0.732 and a weighted F1 score of 0.7273")
                    st.markdown("### Evaluation of  Logistic Regression ")
                    image = Image.open('resources/imgs/confusion4.jpg')
                    st.image(image,width=700)
                    image = Image.open('resources/imgs/metrics4.jpg')
                    st.image(image,width=700)
                    image = Image.open('resources/imgs/heatmap4.jpg')
                    st.image(image,width=700)
                    st.markdown("**Key Observations**")
                    st.markdown("A Classification report is used to measure the quality of predictions from a classification algorithm.")
                    st.markdown("The confusion matrix heatmap shows the model's  ability to classify positive samples, each class achieving a recall score of:")
                    st.markdown("+ Anti Climate Change : 0.44")
                    
                    st.markdown("+ Neutral : 0.47")
                    st.markdown("+ Pro : 0.86")
                    st.markdown("+ News : 0.77")
                    st.markdown("Just like the LinearSVC we see that Logistic Regression does a very good job at classifying positive classes")
                    image = Image.open('resources/imgs/print accuracy4.jpg')
                    st.image(image,width=700)
                    
                    st.markdown("### Evaluation of SGD Classifier")
                    image = Image.open('resources/imgs/confusion5.jpg')
                    st.image(image,width=700)
                    image = Image.open('resources/imgs/metrics5.jpg')
                    st.image(image,width=700)
                    image = Image.open('resources/imgs/heatmap5.jpg')
                    st.image(image,width=700)
                    st.markdown("**Key Observations**")
                    st.markdown("A Classification report is used to measure the quality of predictions from a classification algorithm.")
                    st.markdown("The confusion matrix heatmap shows the model's  ability to classify positive samples, each class achieving a recall score of:")
                    st.markdown("+ Anti Climate Change : 0.47")
                    st.markdown("+ Neutral : 0.40")
                    st.markdown("+ Pro : 0.83")
                    st.markdown("+ News : 0.80")
                    st.markdown("SGD classifier scored the highest in classification of positive classes for `anti` and `neutral` sentiment classes despite incorretly classsifying `anti` and `neutral` sentiment classes as `Pro` sentiment class 35% and 42% of the time respectively")
                    st.markdown("- The SGD classifier is just as good at classifying `Pro` sentiment classs as the LinearSVC both achieving an f1 score of 0.84 however falls short in classifying the rest of the sentiment classes")
                    image = Image.open('resources/imgs/print accuracy5.jpg')
                    st.image(image,width=700)
                    st.markdown("The overall accuracy score of the SGD classifier is 0.7317 and the wighted f1 score of 0.7236")
                    st.markdown("### Support Vector Classfifier")
                    image = Image.open('resources/imgs/confusion6.jpg')
                    st.image(image,width=700)
                    image = Image.open('resources/imgs/metrics6.jpg')
                    st.image(image,width=700)
                    image = Image.open('resources/imgs/heatmap6.jpg')
                    st.image(image,width=700)
                    st.markdown("**Key Observations**")
                    st.markdown("A Classification report is used to measure the quality of predictions from a classification algorithm.")
                    st.markdown("The confusion matrix heatmap shows the model's  ability to classify positive samples, each class achieving a recall score of:")
                    st.markdown("+ Anti Climate Change : 0.22")
                    st.markdown("+ Neutral : 0.33")
                    st.markdown("+ Pro : 0.88")
                    st.markdown("+ News : 0.80")
                    st.markdown("The Support Vector Classifier incorrectly classfied over 50% of `neutral` tweets as `Pro` climate change tweets and 49% of `anti` sentiment class tweets as `Pro` sentiment class tweets.")
                    
                    st.markdown("- Much like the `LinearSVC` we see that the  the `SVC` does a really good job at classifying `Pro` sentiment class with a score of 0.88, followed by the `News` sentiment class with an f1 score of over 0.20.")
                    st.markdown("- Unlike most of the models we've build this far, the Support Vector Classifier struggle more with classifying the `Anti`sentiment class.")    
                    image = Image.open('resources/imgs/print accuracy6.jpg')
                    st.image(image,width=700)
                    st.markdown("### Ridge Classifier")
                    image = Image.open('resources/imgs/confusion7.jpg')
                    st.image(image,width=700)
                    image = Image.open('resources/imgs/metrics7.jpg')
                    st.image(image,width=700)
                    image = Image.open('resources/imgs/heatmap7.jpg')
                    st.image(image,width=700)
                    st.markdown("**Key Observations**")
                    st.markdown("A Classification report is used to measure the quality of predictions from a classification algorithm.")
                    st.markdown("The confusion matrix heatmap shows the model's  ability to classify positive samples, each class achieving a recall score of:")
                    st.markdown("+ Anti Climate Change : 0.42")
                    st.markdown("+ Neutral : 0.48")
                    st.markdown("+ Pro : 0.84")
                    st.markdown("+ News : 0.76")
                    st.markdown("The major concern here is that the Ridge classification classified 40% of of `neutral` tweets as `Pro` climate change tweets")
                    st.markdown("- Much like the `LinearSVC` we see that the  the `Ridge classifier` does a really good job at classifying `Pro` sentiment class with a score of 0.84, followed by the `News` sentiment class with an f1 score of over 0.76.")
                    st.markdown("- Just like the support Vector Classifier, we see that Ridge Classifier does very good job at classifying the `anti` and `neutral` sentiment class")
                    image = Image.open('resources/imgs/print accuracy7.jpg')
                    st.image(image,width=700)
                    st.markdown("## Model Comparision")
                    image = Image.open('resources/imgs/comparing.jpg')
                    st.image(image,width=700)
                    image = Image.open('resources/imgs/classifiers perfomance.jpg')
                    st.image(image,width=700)
                    st.markdown("**Key observations**")
                    st.markdown("From the above bar graph we see comparison of all the 7 models we've attempted thus far based on their `accuracy score` and associated `wighted f1 score`")
                    st.markdown("- We see that our top 3 best performing models are `LinearSVC`,`Logistic Regression` and `Ridge Classification` respectively, theres are the models will use in ensemble methods to try and improve our results")
                    st.markdown("- The `Decision Tree` classifer is the worst  at classifying the tweets with the lowest accuracy and wighted f1 scores of 0.64 and 0.61 respectivey")
                    st.markdown("**LinearSVC is the best performing model out of all 7 models that we've tried thus far with an accuracy score of 0.7538 and a weighted f1 score of 0.7453**")
                    st.markdown("# Ensemble Methods")
                    st.markdown("Ensemble learning in machine learning is the practice of combining multiple models to try and achieve higher overall model performance. In general, ensembles consist of multiple **heterogeneous or homogeneous** models trained on the same dataset. Each of these models is used to make predictions on the same input, then these predictions are aggregated across all models in some way (e.g. by taking the mean) to produce the final output.")
                    st.markdown("## Heterogeneous Ensembel Method")
                    st.markdown("This type of ensemble consists of different types of models, so it can add pretty much any classification model we want, however in our case we're only going to add our top 3 best perfoming models which are, `LinearSVC, Stochastic Gradient Descent, Logistic Regression, `.")
                    st.markdown("The Heterogeneous ensemble method we're going to look at is the `Voting classifier`")
                    st.markdown("### Voting classifer ")
                    st.markdown("Voting involves combining individual model outputs through a kind of '[majority rule]' paradigm. The diagram below shows how the Voting Classifier works")
                    image = Image.open('resources/imgs/ud382N9.png')
                    st.image(image,width=700)
                    image = Image.open('resources/imgs/voting classifier.jpg')
                    st.image(image,width=700)
                    image = Image.open('resources/imgs/metrics voting classifier.jpg')
                    st.image(image,width=700)
                    st.markdown("The voting  classifer did a good job at classifying the sentiment classes which 'Neutral' sentiment class being the poorly classified one")
                    st.markdown("- achieving the f1 score of")
                    st.markdown("- `Pro` sentiment class : 0.81")
                    st.markdown("- `News` sentiment class : 0.73")
                    st.markdown("- `Anti` sentiment class: 0.53")
                    st.markdown("- `Neutral` sentiment class : 0.50")
                    image = Image.open('resources/imgs/print accuracy voting classifier.jpg')
                    st.image(image,width=700)

                    st.markdown("The Voting Classifier achieved the accuracy and weighted f1 score of 0.7332 and 0.727 respectively, which is not much of an improvement from our best performing model that achieved the accuracy of 0.75 and 0.74")

                    st.markdown("# Hyperparameter Tunning")
                    st.markdown("We will look at two methods of hyperparameter tunning, namely `GridSearchCV` and `Parfit`")
                    st.markdown("* Models we will perform hyperparameter tunning on")
                    st.markdown(" * LinearSVC")
                    st.markdown(" * Logistic Regression")
                    st.markdown(" * Ridge Classifier")
                    st.markdown("The caveat of using pipelines to build our models is that we can't easily get the parameters for our models as such to perfom hyperparameter tunning and obtain the best parameters for our models we wont be using the pipelines, this means we will convert raw text data to numeric data independently from building the models.")

                    st.markdown("### Tuning LinearSVC")
                    image = Image.open('resources/imgs/tuning LinearSVC.jpg')
                    st.image(image,width=700)
                    st.markdown("### Tuning Logistic Regression")
                    image = Image.open('resources/imgs/tuning Logistic.jpg')
                    st.image(image,width=700)
                    st.markdown("## Tuning Ridge Classifier")
                    image = Image.open('resources/imgs/tuning Ridge.jpg')
                    st.image(image,width=700)
                    st.markdown("# Final model selection")
                    st.markdown("Comparing all the models we've build so far to choose the best performing one")
                    image = Image.open('resources/imgs/classifier perfomance final selection model.jpg')
                    st.image(image,width=700)
                    st.markdown("We have build a total of 11 models in this notebook out of all the models we've build, We see that the best performing model is the tunned LogisticReg with the best accuracy score of 0.76 and the best weighted f1 score of 0.76 based on the validation dataset, however for the unseen/test dataset the Ridge Classifier achieved the best score on Kaggle.")
                    st.markdown("We will be using the Ridge Regression to make the final prediction")
                    st.markdown("## Final evaluation of our best model")
                    image = Image.open('resources/imgs/metrics final model.jpg')
                    st.image(image,width=700)
                    image = Image.open('resources/imgs/f1 score per sentiment class.jpg')
                    st.image(image,width=700)
                    st.markdown("The above bar graph shows the f1 score for each sentiment class our best model")
                    st.markdown("- The Ridge Classifier is our best performing model, achieving f1 score of 0.83 for `Pro climate change` sentiment class, followed by `News` and `Anti` Climate sentiment classes with f1 scores of 0.76 and 0.58 respectively, which is quite impressive given that all our models perfomed poorly when it comes to classifying `anti climate change` sentiment class")
                    st.markdown("## ROC Curves and AUC")
                    st.markdown("ROC curves show the trade-off between sensitivity/recall known as the ability of a model to detect positive samples and specificity also known as the True Negative rate. classifiers that produce curves that are closer to the top left corner, that is closer to the 100% True positive rate are classifiers that are considerd to have better perfomance")
                
                    
	if selection == "Conclusion":
                if st.checkbox('Show conclusion'):
                    st.info("Summary")
                   
                
                    st.markdown("# Conclusion")
                    st.markdown("In this notebook we have succesfully build 11 machine learning models to classify whether not a person believes in man made climate change based on their novel tweet data, Even though our models struggled with classifying the `anti` man made climate change sentiment class they did a very good job as classifying the `pro` man made climate change sentiment class. Our best model is the Ridge Classification model  achieving an accuracy score and the weighted f1 score of 0.76 and 0.76 respectively based on the validation dataset.")
                    st.markdown("The Ridge classifier model achieved an F1 score of 0.76043 on unseen/test data.")
                    st.markdown("Text classification problems tend to be quite high dimensionality and high dimensionality problems are likely to be linearly separable, So linear classifiers, whether ridge classification or SVM with a linear kernel, are likely to do well. In both cases, the ridge parameter or C for the SVM control the complexity of the classifier and help to avoid over-fitting by separating the patterns of each class by large margins as we can see in our final model comparision, our best models are linear classifiers")
	if selection == "Feature Engineering":
                if st.checkbox('Show feature engineering'):
                    st.info("Engineering process")
                   
                
                    st.markdown("# Feature Engineering")
                    st.markdown("### TFIDF")
                    st.markdown("`TF-IDF` stands for Term Frequency — Inverse Document Frequency and is a statistic that aims to better define how important a word is for a document, while also taking into account the relation to other documents from the same corpus. This is performed by looking at how many times a word appears into a document while also paying attention to how many times the same word appears in other documents in the corpus. `vocabulary_` Is a dictionary that converts each word in the text to feature index in the matrix, each unique token gets a feature index.")
                    st.markdown("### CountVectorizer")
                    st.markdown("The `CountVectorizer` provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary by creating a matrix in which each unique word is represented by a column of the matrix, and each text sample from the document is a row in the matrix.")
                    image = Image.open('resources/Untitled.png')
                    st.image(image,width=700)
                    st.markdown("### Creating our X and y Metrics")
                    image = Image.open('resources/X and y.jpg')
                    st.image(image,width=700)
                    st.markdown("### Splitting data")
                    st.markdown("Separating data into training and validation sets is an important part of evaluating our models. In our case we will randomly split the train data into 90% train and 10% validation. After our model is trained with the train data we then use it to make predictions for the target using the validation set,Because the data in the validation set already contains known values for the target variable this will make it easy for us to asses our model's accuracy.")
                    image = Image.open('resources/split.jpg')
                    st.image(image,width=700)
                    st.markdown("# Pipelines")
                    st.markdown("`Pipeline`  by definition is a tool that sequentially applies a list of transforms and a final estimator. Intermediate steps of pipeline implement fit and transform methods and the final estimator only needs to implement fit. In our case pipelines will help us tranform the train, validation and test data as well as train our models.")
                    st.markdown("Since our models can only process numerical data our first step is to build a pipeline that converts text data into numeric data, In this notebook we will be focusing on two methods of feature engineering, which we will use to convert text data to numeric data, namely `TfidfVectorizer` and the `CountVectorizer`, then we will train our models within these pipelines")
                    st.markdown("We will be building pipelines with features generated using both `tfidfVectorizer` and the `CountVectorizer`")
                    st.markdown("**CREATING PIPELINES**")
                    
                    image = Image.open('resources/pipeline1.jpg')
                    st.image(image,width=700)
                    image = Image.open('resources/pipeline2.jpg')
                    st.image(image,width=700)
                    image = Image.open('resources/pipeline3.jpg')
                    st.image(image,width=700)
                    image = Image.open('resources/pipeline4.jpg')
                    st.image(image,width=700)

                    st.markdown("## Training models")
                    st.markdown("Each model is trained using it's custom pipeline which will take raw text data turn it into numeric data and initial the classifier with default parameters")
                    image = Image.open('resources/training.jpg')
                    st.image(image,width=700)

                    st.markdown("### Avg accuracy score per method")
                    image = Image.open('resources/feature method.jpg')
                    st.image(image,width=700)
                    st.markdown("### Avg accuracy score per method")
                    
                    
                    source = pd.DataFrame({
                        'Accuracy score': ["0.7143", "0.7203"],
                        'Feature engineering method': ["TFIDF", "CountVec"]
                        })
                                

                    bar_chart = alt.Chart(source).mark_bar().encode(
                        x='Accuracy score:Q',
                        y="Feature engineering method:O"
                    )#.properties(height=700)
                    st.altair_chart(bar_chart, use_container_width=True)

                    st.markdown("We see that on average the models build using the CountVectorizer performed the best and for the remainder of this notebook we will generate our features using the CountVectorizer")

                    

                    
	if selection == "Classification Models":
                if st.checkbox('Show models'):
                    st.info("Models")
                   
                
                    st.markdown("* Decision Tree Classifier")
                    st.markdown("* RandomForest Classifier")
                    st.markdown("* LinearSVC(Support Vector Classifier")
                    st.markdown("* Support Vector Classifier")
                    st.markdown("* Logistic Regression")
                    st.markdown("* Stochastic Gradient Descent (SGD)")
                    st.markdown("* Ridge Classiffier")
                    
                    st.markdown("### Decision Tree Classifier")
                    st.markdown("A decision tree is a decision support tool that uses a tree-like graph or model of decisions and their possible consequences. It is one way to display an algorithm that only contains conditional control statements.")
                    st.markdown("Decision trees are extremely intuitive ways to classify objects or predict continuous values: you simply ask a series of questions designed to zero-in on the classification/prediction.Overfitting turns out to be a general property of decision trees: it is very easy to go too deep in the tree, and thus to fit details of the particular data rather than the overall properties of the distributions they are drawn from. This issue can be addressed by using random forests.")
                    image = Image.open('resources/1_bcLAJfWN2GpVQNTVOCrrvw.png')
                    st.image(image,width=700)

                    st.markdown("### Random Forest Classifier")
                    st.markdown("A random forest is a powerful non-parametric algorithm that is an example of an ensemble method built on decision trees, meaning that it relies on aggregating the results of an ensemble of decision trees. The ensemble of trees are randomized and the output is the mode of the classes (classification) or mean prediction (regression) of the individual trees.")
                    image = Image.open('resources/voting_dnjweq.jpg')
                    st.image(image,width=700)

                    st.markdown("## Support Vector Classification(LinearSVC)")
                    st.markdown("SVM or Support Vector Machine is a linear model for classification and regression problems. It can solve linear and non-linear problems and work well for many practical problems. The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the data into classes as seen in the diagram below")
                    image = Image.open('resources/1_dh0lzq0QNCOyRlX1Ot4Vow.jpeg')
                    st.image(image,width=700)

                    st.markdown("To better explain the concept of `SVM` we will look at a case of two classes.")
                    st.markdown("**To find the best line seperating the classes**")
                    st.markdown("The `SVM` algorithm finds the points closest to the line from both the classes.These points are called support vectors, then it compute the distance between the line and the support vectors, This distance is called the margin. Our goal is to maximize the margin.")
                    st.markdown("In a case for more than two classes the goal is to find the the best hyperplane that seperates the classes. The hyperplane for which the margin is maximum is the optimal hyperplane.")

                    st.markdown("We wil be looking at two Support Vector Classifer models namely SVC and LinearSVC, the main differences between these two are as follows:")
                    st.markdown("- By default scaling, LinearSVC minimizes the squared hinge loss while SVC minimizes the regular hinge loss.")
                    st.markdown("- LinearSVC uses the One-vs-All (also known as One-vs-Rest) multiclass reduction while SVCuses the One-vs-One multiclass reduction.")
                    st.markdown("* Stochastic Gradient Descent (SGD)")

                    st.markdown("## Logistic Regression")
                    st.markdown("**Logistic regression** is a statistical method for predicting binary classes. The outcome or target variable is dichotomous in nature. Dichotomous means there are only two possible classes. For example, it can be used for cancer detection problems. It computes the probability of an event occurrence.")
                    st.markdown("Logistic Regression uses the probability of a data point to belonging to a certain class to classify each datapoint to it's best estimated class.")
                    st.markdown("Logistic regression has been rated as the best performing model for linearly separable data especially if it's predicting binary data(Yes & NO or 1 & 0), and performs better when there's no class imbalance.")
                    st.markdown("The figure below is the sigmoid function logistic regression models use to make predictions:")
                    image = Image.open('resources/1_QY3CSyA4BzAU6sEPFwp9ZQ.png')
                    st.image(image,width=700)
                    st.markdown("Advantages")
                    st.markdown("* Convenient probability scores for observations (probability of each outcome is transformed into a classification);")
                    st.markdown("* Not a major issue if there is collinearity among features (much worse with linear regression).")
                    st.markdown("Disadvantages")
                    st.markdown("* Can overfit when data is unbalanced (i.e.: we have far more observations in one class than the other).")
                    st.markdown("* Doesn't handle large number of categorical variables well.")
                    
                    st.markdown("## Stochastic Gradient Descent")
                    st.markdown("**Stochastic Gradient Descent (SGD)** is a simple yet very efficient approach to fitting linear classifiers and regressors under convex loss functions such as (linear) Support Vector Machines and Logistic Regression. Even though SGD has been around in the machine learning community for a long time, it has received a considerable amount of attention just recently in the context of large-scale learning.")
                    st.markdown("In Stochastic Gradient Descent, a few samples are selected randomly instead of the whole data set for each iteration.")
                    st.markdown("The advantages of Stochastic Gradient Descent are:")
                    st.markdown("* Efficiency.")
                    st.markdown("* Ease of implementation (lots of opportunities for code tuning).")
                    st.markdown("The disadvantages of Stochastic Gradient Descent include:")
                    st.markdown("* SGD requires a number of hyperparameters such as the regularization parameter and the number of iterations.")
                    st.markdown("* SGD is sensitive to feature scaling.")
                    
                    st.markdown("## Ridge Classifier")
                    st.markdown("This Ridge classifier first converts the target values into {-1, 1} and then treats the problem as a regression task (multi-output regression in the multiclass case) ")
                    st.markdown("Ridge Regression is a technique for analyzing multiple regression data that suffer from multicollinearity. When multicollinearity occurs, least squares estimates are unbiased, but their variances are large so they may be far from the true value. By adding a degree of bias to the regression estimates, ridge regression reduces the standard errors. It is hoped that the net effect will be to give estimates that are more reliable. Another biased regression technique, principal components regression, is also available in NCSS. Ridge regression is the more popular of the two methods.")
                    
                
               
                
	if selection == "Information":
                if st.checkbox('Show data description'):
                    st.info("Data Description")
                    st.markdown("The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo. The dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were collected. Each tweet is labelled as one of the following classes:")
                
                    st.markdown("**Class Description**")
                    st.markdown("* 2 News: the tweet links to factual news about climate change")
                    st.markdown("* 1 Pro: the tweet supports the belief of man-made climate change")
                    st.markdown("* 0 Neutral: the tweet neither supports nor refutes the belief of man-made climate change")
                    st.markdown("* -1 Anti: the tweet does not believe in man-made climate change")
                    st.markdown("**Variable definitions**")
                    st.markdown("* sentiment: Sentiment of tweet")
                    st.markdown("* message: Tweet body")
                    st.markdown("* tweetid: Twitter unique id")
               
                if st.checkbox("Show raw data"):
                    st.write(raw[['sentiment', 'message']])

                   

		
                
	if selection == "Team":
                st.subheader("2201FT_GM2")
                if st.checkbox('Show team members'):                    
                        image = Image.open('resources/imgs/571fdbd2-7a7c-414d-ab71-750299b3c191.jpg')
                        st.image(image, caption='Rirhandzo Masinga',width=500)
                        image = Image.open('resources/imgs/6c796955-cc18-4a37-a143-60ce7a418f70.jpg')
                        st.image(image, caption='Tebelelo Selowa',width=700)
                        image = Image.open('resources/imgs/qqqqqqqq.jpeg')
                        st.image(image, caption='Maphuti Lehutjo',width=400)
                        image = Image.open('resources/imgs/384800b9-49df-4a5c-87e6-a37f9a2891e6 (2).jpg')
                        st.image(image, caption='Aphiwe Rasisemula (Leader)',width=700)
           
                
		
	if selection == "EDA":
                if st.checkbox("Show Top 20 hashtags from the whole data"):
                        bf_world_list2 = st.multiselect("Which hashtag would you like to see?", bf_world_list)
                        bf_emp_list = []
                        if len(bf_world_list2) == 0:
                                source = pd.DataFrame({
                                    'Hashtag count': bf_world_count_list,
                                    'Hashtags used': bf_world_list
                                 })
                                

                                bar_chart = alt.Chart(source).mark_bar().encode(
                                    x='Hashtag count:Q',
                                    y="Hashtags used:O"
                                )#.properties(height=700)
                                st.altair_chart(bar_chart, use_container_width=True)
                        else:
                                for i in bf_world_list2:
                                        bf_emp_list.append(bf_world_dict[i])
                                
                                st.subheader("Sentiment type vs Count graph")
                                st.write("  Sentiment type vs Sentount Bar graph")
                                source = pd.DataFrame({
                                'Hashtag count': bf_emp_list, 'Hashtags used': bf_world_list2 })
  
                                bar_chart = alt.Chart(source).mark_bar().encode(
                                        x='Hashtag count:Q',
                                    y="Hashtags used:O"
                                )
                                st.altair_chart(bar_chart, use_container_width=True)
                if st.checkbox("Show Top 20 hashtags pro sentimet"):
                        bf_pro_list2 = st.multiselect("Which hashtag for pro sentiment would you like to see?", bf_pro_list)
                        bf_emp_list2 = []
                        if len(bf_pro_list2) == 0:
                                source = pd.DataFrame({
                                    'Hashtag count': bf_pro_count_list,
                                    'Hashtags used': bf_pro_list
                                 })
                                

                                bar_chart = alt.Chart(source).mark_bar().encode(
                                    x='Hashtag count:Q',
                                    y="Hashtags used:O"
                                )#.properties(height=700)
                                st.altair_chart(bar_chart, use_container_width=True)
                        else:
                                for i in bf_pro_list2:
                                        bf_emp_list2.append(bf_world_dict[i])
                                
                                st.subheader("Sentiment type vs Count graph")
                                st.write("  Sentiment type vs Sentount Bar graph")
                                source = pd.DataFrame({
                                'Hashtag count': bf_emp_list2, 'Hashtags used': bf_pro_list2 })
  
                                bar_chart = alt.Chart(source).mark_bar().encode(
                                        x='Hashtag count:Q',
                                    y="Hashtags used:O"
                                )
                                st.altair_chart(bar_chart, use_container_width=True)
                if st.checkbox("Show Top 20 hashtags for anti sentiment"):
                        bf_anti_list2 = st.multiselect("Which hashtag for anti sentiment would you like to see?", bf_anti_list)
                        bf_emp_list3 = []
                        if len(bf_anti_list2) == 0:
                                source = pd.DataFrame({
                                    'Hashtag count': bf_anti_count_list,
                                    'Hashtags used': bf_anti_list
                                 })
                                

                                bar_chart = alt.Chart(source).mark_bar().encode(
                                    x='Hashtag count:Q',
                                    y="Hashtags used:O"
                                )#.properties(height=700)
                                st.altair_chart(bar_chart, use_container_width=True)
                        else:
                                for i in bf_anti_list2:
                                        bf_emp_list3.append(bf_anti_dict[i])
                                
                                st.subheader("Sentiment type vs Count graph")
                                st.write("  Sentiment type vs Sentount Bar graph")
                                source = pd.DataFrame({
                                'Hashtag count': bf_emp_list3, 'Hashtags used': bf_anti_list2 })
  
                                bar_chart = alt.Chart(source).mark_bar().encode(
                                        x='Hashtag count:Q',
                                    y="Hashtags used:O"
                                )
                                st.altair_chart(bar_chart, use_container_width=True)
                if st.checkbox("Show Top 20 hashtags from the news sentiment"):
                        bf_news_list2 = st.multiselect("Which hashtag for news sentiment would you like to see?", bf_news_list)
                        bf_emp_list4 = []
                        if len(bf_news_list2) == 0:
                                source = pd.DataFrame({
                                    'Hashtag count': bf_news_count_list,
                                    'Hashtags used': bf_news_list
                                 })
                                

                                bar_chart = alt.Chart(source).mark_bar().encode(
                                    x='Hashtag count:Q',
                                    y="Hashtags used:O"
                                )#.properties(height=700)
                                st.altair_chart(bar_chart, use_container_width=True)
                        else:
                                for i in bf_news_list2:
                                        bf_emp_list4.append(bf_news_dict[i])
                                
                                st.subheader("Sentiment type vs Count graph")
                                st.write("  Sentiment type vs Sentount Bar graph")
                                source = pd.DataFrame({
                                'Hashtag count': bf_emp_list4, 'Hashtags used': bf_news_list2 })
  
                                bar_chart = alt.Chart(source).mark_bar().encode(
                                        x='Hashtag count:Q',
                                    y="Hashtags used:O"
                                )
                                st.altair_chart(bar_chart, use_container_width=True)
                if st.checkbox("Show Top 20 hashtags from the neutral sentiment"):
                        bf_neutral_list2 = st.multiselect("Which hashtag for neutral sentiment would you like to see?", bf_neutral_list)
                        bf_emp_list5 = []
                        if len(bf_neutral_list2) == 0:
                                source = pd.DataFrame({
                                    'Hashtag count': bf_neutral_count_list,
                                    'Hashtags used': bf_neutral_list
                                 })
                                

                                bar_chart = alt.Chart(source).mark_bar().encode(
                                    x='Hashtag count:Q',
                                    y="Hashtags used:O"
                                )#.properties(height=700)
                                st.altair_chart(bar_chart, use_container_width=True)
                        else:
                                for i in bf_neutral_list2:
                                        bf_emp_list5.append(bf_neutral_dict[i])
                                
                                st.subheader("Sentiment type vs Count grap")
                                st.write("  Sentiment type vs Sentount Bargraph")
                                source = pd.DataFrame({
                                'Hashtag count': bf_emp_list5, 'Hashtags used': bf_neutral_list2 })
  
                                bar_chart = alt.Chart(source).mark_bar().encode(
                                        x='Hashtag count:Q',
                                    y="Hashtags used:O"
                                )
                                st.altair_chart(bar_chart, use_container_width=True)
                if st.checkbox('Show Top 5 Hashtags Summary:'):
                        st.markdown("#### Quick observations:")
                        image = Image.open('resources/Top 5 Hashtags.jpg')
                        st.image(image, caption='Top 5 Hashtags Summary',width=700)
                                        
                        st.markdown("* Overall, #climatechange and #climate are at the top of the charts as expected, they are the words used to identify tweets that identify climate change content.")
                        st.markdown("* #BeforeTheFlood was trending in the year 2016 following the documentray by Actor Leonardo DiCaprio with scientists, activists and world leaders to discuss the dangers of climate change and possible solutions.")
                        st.markdown("* In the same year; 2016, the outgoing president of U.S.A was canvassing for presidency and he had made his stand clear on Climate Change clear to the public describing it as a 'hoax'. That accounts for his name appearing across all sentiment classes.")
                        st.markdown("* In his campaign, he used the slogan #MAGA which stands for 'Make America Great Again'. This appears to have attracted more tweets for tweets in the 'anti' class, making it to the top spot.")
                        st.markdown("* It is for this reason that #iamvotingbecause was at the top for 'pro' class as it was election year in the United States of America.")
                        st.markdown("* We can notice that #cop22 also made it to top 5 in the 'pro' class. COP22 (Conference of the Parties) represents the United Nations Climate Change Conference in 2016.")
                        
               
                if st.checkbox("Show Box plot of message length for each sentiment class"):
                        sentiment2_list = st.multiselect("Which sentiment would you like to see?", sentiment2)
                        www = []
                        mmm = []
                        rrr = []
                        count = 0
                        count2 = 0
                        if len(sentiment2_list) == 0:

                                columns = [pro, anti, news, neutral]

                                fig, ax = plt.subplots()
                                ax.boxplot(columns)
                                box = ax.boxplot(columns, notch=True, patch_artist=True)
                                plt.xticks([1, 2, 3, 4], ["Pro", "Anti", "News", "Neutral"], rotation=10)
                                colors = ['#0000FF', '#00FF00',
                                  '#FFFF00', '#FF00FF']
                                for patch, color in zip(box['boxes'], colors):
                                    patch.set_facecolor(color)
                                plt.ylabel('Lenght of the message')
                                plt.xlabel('Sentiment class')
                                plt.title('Message length for each sentiment class')
                                st.pyplot(fig)
                                #st.plt.show()
                                if st.checkbox('Show pie chart interperetation'):
                                        st.markdown("**Looking at the above pie chart:**")
                                        st.markdown("* We see that pro tweets make up the majority with 53.92%")
                                        st.markdown("* Followed by news 23.01%, neutral 14.87% and lastly 8.19%")
                                
                        else:
                                for i in sentiment2_list:
                                        www.append(sentiment2_dict[i])
                                        count += 1
                                        mmm.append(count)
                                        rrr.append(colors2[count2])
                                        count2 += 1
                                        
                                columns = www

                                fig, ax = plt.subplots()
                                ax.boxplot(columns)
                                box = ax.boxplot(columns, notch=True, patch_artist=True)
                                plt.xticks(mmm, sentiment2_list, rotation=10)
                                colors = rrr
                                for patch, color in zip(box['boxes'], colors):
                                    patch.set_facecolor(color)
                                plt.ylabel('Lenght of the message')
                                plt.xlabel('Sentiment class')
                                plt.title('Message length for each sentiment class')
                                st.pyplot(fig)
                                #st.plt.show()
                                if st.checkbox('Show Box plot interperetation'):
                                        st.markdown("#### Quick observations:")
                                        st.markdown("* There is a strong imbalance for our sentiment classes")
                                        st.markdown("* Sentiment class '1' (Pro) dominates the chart with over 50% contribution, while class '-1' (Anti) lags behind with 8%.")
                                        st.markdown("* The text length is dependent on the character limit of each tweet on Twitter. Character limitused to be 140, but it increased to 280 characters in late 2017.")
                                        st.markdown("* There are evident outliers in all classes, except for 'neutral' sentiment where all the data is taken in.")
                                        st.markdown("* It is evident that the 'pro' class had a lot to say to express their opinion, as shown by more lenghty message in the outliers.")
               
                if st.checkbox('Show Sentiment type vs Count bar graph'): # data is hidden if box is unchecked
                        sentiment_list = st.multiselect("Which sentiment would you like to see?", sentiment)
                        yyy = []
                        if len(sentiment_list) == 0:
                                st.subheader("Sentiment type vs Count bar graph")
                                
                                source = pd.DataFrame({
                                'count': sentiment_count, 'sentiment':  sentiment})
  
                                bar_chart = alt.Chart(source).mark_bar().encode(
                                        y='count:Q',
                                        x='sentiment:N',
                                )
                                st.altair_chart(bar_chart, use_container_width=True)
                                if st.checkbox('Show bar graph interperetation'):
                                        st.markdown("#### Quick observations:")
                                        st.markdown("* Most tweets are pro climate change, which are more than double all the other sentiments")
                                        st.markdown("* Tweets about news on climate change are the second highest")
                                        st.markdown("* 3rd highest tweets are neutral")
                                        st.markdown("* 4th and lowest tweets are anti-climate change")
                                        
                        else:
                                for r in sentiment_list:
                                        yyy.append(sentiment_dict[r])
                                
                                st.subheader("Sentiment type vs Count bar graph")
                                st.write("  Sentiment type vs Sentiment count Bar graph")
                                source = pd.DataFrame({
                                'count': yyy, 'sentiment': sentiment_list })
  
                                bar_chart = alt.Chart(source).mark_bar().encode(
                                        y='count:Q',
                                        x='sentiment:N',
                                )
                                st.altair_chart(bar_chart, use_container_width=True)
                                if st.checkbox('Show bar graph interperetation'):
                                        st.markdown("#### Quick observations:")
                                        st.markdown("* Most tweets are pro climate change, which are more than double all the other sentiments")
                                        st.markdown("* Tweets about news on climate change are the second highest")
                                        st.markdown("* 3rd highest tweets are neutral")
                                        st.markdown("* 4th and lowest tweets are anti-climate change")

                        
                if st.checkbox("Show top 30 word frequency bar chat"):
                        word = st.multiselect("Which sentiment wouldyou like to see?", word_dict)
                        zzz = []
                        if len(word) == 0:
                                st.subheader("Top 30 most frequently used words bar graph")
                                sourc = pd.DataFrame({
                                'Count': list2, 'Word': list1 })
  
                                bar_chart = alt.Chart(sourc).mark_bar().encode(
                                        y='Count:Q',
                                        x='Word:N',
                                )
                                st.altair_chart(bar_chart, use_container_width=True)
                                if st.checkbox('Show bar graph interperetation'):
                                        st.markdown("#### Quick observations:")
                                        st.markdown('* We can see we used "climate" and "change" more than twice other words, followed by "global" and "warning", then "trump" while others are almost the same.')
                        else:
                                for i in word:
                                        zzz.append(word_dict[i])
                                
                                st.subheader("Sentiment type vs Count bar graph")
                                st.write("  Sentiment type vs Sentiment count Bar graph")
                                sourc = pd.DataFrame({
                                'Count': zzz, 'Word': word })
  
                                bar_chart = alt.Chart(sourc).mark_bar().encode(
                                        y='Count:Q',
                                        x='Word:N',
                                )
                                st.altair_chart(bar_chart, use_container_width=True)
                                if st.checkbox('Show pie chart interperetation'):
                                        st.markdown("#### Quick observations:")
                                        st.markdown('* We can see we used "climate" and "change" more than twice other words, followed by "global" and "warning", then "trump" while others are almost the same.')

                        
                      
                        
                if st.checkbox('Show Percentage distribution of sentiments pie chart'): # data is hidden if box is unchecked
                        st.subheader("Percentage distribution of sentiments")
                        labels = "Pro", "News", "Neutral", "Anti"
                        sizes = [8530, 3640, 2353, 1296]
                        explode = (0.1, 0.1, 0.1, 0.1)
               

                        fig1, ax1 = plt.subplots()
                        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True,
                        startangle=90)
                        ax1.axis('equal')

                        st.pyplot(fig1)
                        if st.checkbox('Show pie chart interperetation'):
                                        st.markdown("#### Quick observations:")
                                        st.markdown("* We see that pro tweets make up the majority with 53.92%")
                                        st.markdown("* Followed by news 23.01%, neutral 14.87% and lastly 8.19%")
 		
                
                        
        
                # Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Ridgeclfr.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
