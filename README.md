# diabetes-ml
This is a Python project where I did a preliminary data exploration of a Diabetes dataset. I then created a decision tree as well as a classifier to predict a patient's diagnosis of Diabetes given a variety of health factors and metrics that were detailed in the dataset.

# Analysis of Findings
For this dataset analysis, we analyzed a dataset for diabetes for a number of patients. The 
dataset includes various patient info such as pregnancies, glucose, blood pressure, skin thickness, 
Insulin, BMI, Diabetes Pedigree Function, Age, and overall outcome. There are 768 total patients noted 
in the dataset with no missing cells or duplicate rows. In exploring the data, it was noted that skin 
thickness had a high correlation with Insulin and blood pressure was highly correlated with BMI. Age 
was highly correlated with pregnancies, but this observation is not very valuable purely from a logical 
standpoint. It also appears that the age numbers trended towards the younger side.

When observing the Phik correlation chart, we can corroborate the above correlations, but also 
see that both age and glucose have a high correlation with overall outcome. Glucose also had a higher 
correlation with Insulin. Additionally, the Diabetes Pedigree Function and Insulin had a high correlation.
For the decision tree, we used the “sklearn” library to fit the patient information to their overall 
outcome. We used the dataset as the training set to create the decision tree, a classifier, and our 
prediction function. When the patient info is fed again into our decision tree classifier, the accuracy 
percentage is often very high and can even achieve 100%. However, this is most likely due to the use of 
a set that was used to train the data. It would be useful in the future to use half the set as a training set. 


Included in the repository is the outputted decision tree. With all of the patient info included, it results in a very 
large decision tree that can have a large depth when evaluating how to classify a given set of data in 
order to deduce an outcome. If a route is followed with a higher depth and number of decisions, we can 
note that the gini coefficient continues to decrease to 0 with a well-informed tree progression that can 
accurately determine the outcome. In the shortest path for an outcome of “true”, the path that follows 
age, BMI, pregnancies, and Diabetes Pedigree function can most quickly deduce an outcome. For the 
shortest path of “false”, the evaluation of BMI, then glucose, Insulin, and Diabetes Pedigree Function 
can have the most impactful information. One other observation was that in most paths within the 
decision tree, skin thickness and the Diabetes Pedigree Function can have higher Gini coefficients, which 
has significant implications about the impurity and how they are equally divided amongst all classes. 
From the analysis of the tree, it’s clear that some patient info is very indicative of the overall outcome 
while other pieces of patient info can only serve as supplementary for trickier cases or when the initial 
decisions in the tree are not efficient.
