
# coding: utf-8

# In[111]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[112]:


### LOADING IN THE DATAFRAME

df = pd.read_csv("profiles.csv")


# In[113]:


### EXPLORING THE DATA

# creating a list of the headers to help examine the data
df_headers = list(df)
print(df_headers)


# In[114]:


# Listing all responses for the potentially associated variables
print(df.sex.value_counts())
print()
print(df.body_type.value_counts())
print()
print(df.diet.value_counts())
print()
print(df.pets.value_counts())


# In[115]:


### VISUALIZING THE DATA


# In[116]:


# Pie chart showing the proportion of male and female respondents
dfs = df.sex.value_counts()
dfs = dfs.tolist()
data = dfs
labels = ['m','f']
plt.pie(data,labels = labels, autopct = "%0.1f%%")
plt.axis("equal")
plt.xlabel("Sex")
plt.ylabel("Number of Users")
plt.savefig("Total_Male_Female_Users_Pie.png")
plt.show()


# In[117]:


# Self-identified body type for all users
dfb = df.body_type.value_counts()
dfb = dfb.tolist()
data2 = dfb
labels2 = ["average",
"fit",
"athletic",
"thin",
"curvy",
"a little extra",
"skinny",
"full figured",
"overweight",
"jacked",
"used up",
"rather not say"]
plt.pie(data2,labels = labels2, autopct = "%0.1f%%",textprops={'fontsize': 6})
plt.axis("equal")
plt.xlabel("Body Type")
plt.ylabel("Self-identified Responses")
#plt.figure(figsize = (6,4))
fig1 = plt.gcf()
fig1.savefig("Total_Body_Type_Responses_Pie.png")
plt.show()


# In[118]:


### AUGMENTING THE DATA


# In[119]:


# Converting sex to numerical values and adding a column to the DataFrame
sex_mapping = {"m": 0, "f": 1}
df["sex_num"] = df.sex.map(sex_mapping)


# In[120]:


# Converting diet to numerical values and adding a column to the DataFrame
diet_mapping = {"mostly anything":0,
"anything":1,
"strictly anything":2,
"mostly vegetarian":3,
"mostly other":15,
"strictly vegetarian":5,
"vegetarian":4,
"strictly other":17,
"mostly vegan":6,
"other":16,
"strictly vegan":8,
"vegan":7,
"mostly kosher":9,
"mostly halal":12,
"strictly kosher":11,
"strictly halal":14,
"halal":13,
"kosher":10}
df["diet_num"] = df.diet.map(diet_mapping)


# In[121]:


# Converting body type to numerical values and adding a column to the DataFrame
body_mapping = {"average":6,
"fit":9,
"athletic":10,
"thin":8,
"curvy":5,
"a little extra":4,
"skinny":7,
"full figured":3,
"overweight":2,
"jacked":11,
"used up":0,
"rather not say":1}
df["body_num"] = df.body_type.map(body_mapping)


# In[122]:


# Converting pets to numerical values and adding a column to the DataFrame
pet_mapping = {"likes dogs and likes cats":9,
"likes dogs":4,
"likes dogs and has cats":8,
"has dogs":5,
"has dogs and likes cats":6,
"likes dogs and dislikes cats":3,
"has dogs and has cats":7,
"has cats":11,
"likes cats":10,
"has dogs and dislikes cats":2,
"dislikes dogs and likes cats":12,
"dislikes dogs and dislikes cats":0,
"dislikes cats":1,
"dislikes dogs and has cats":13,
"dislikes dogs":14}
df["pet_num"] = df.pets.map(pet_mapping)


# In[123]:


# Creating a column with the total length of the 10 essay responses, in characters, for each respondent
# ...code taken and modified from the project instructions, with thanks
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]


# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
df["essay_len"] = all_essays.apply(lambda x: len(x))


# In[124]:


# Investigating the augmented total DataFrame
print(df.head())


# In[125]:


# Breaking down body type responses, by gender to visualize and validate its use as a predictive feature

dfmf = df[["sex_num","body_num"]]
dfmf = dfmf.dropna()
dff = dfmf[dfmf.sex_num == 1]
dfm = dfmf[dfmf.sex_num == 0]

ax = plt.subplot(1,2,1)
plt.hist(dfm.body_num, bins=np.arange(13)-.5, alpha = .8, color = "blue")
plt.hist(dff.body_num, bins=np.arange(13)-.5, alpha = .9, color = "pink")
plt.xlabel("Self-identified Body Type")
plt.ylabel("Number of Responses")
ax.set_xticks(range(-1,12))
ax.set_xticklabels([ "","Used Up","Rather Not Say","Overweight","Full Figured","A Litte Extra","Curvy","Average","Skinny","Thin","Fit","Athletic","Jacked"], rotation = 90)
plt.xlim(-1,12)
plt.title("Body Type, by Gender")
plt.legend(["Male","Female"])
#plt.figure(figsize = (6,4))

# Proportionalizing the histogram to show relative responses from both genders
ax = plt.subplot(1,2,2)
plt.hist(dfm.body_num, bins=np.arange(13)-.5, alpha = .8, color = "blue",normed = True)
plt.hist(dff.body_num, bins=np.arange(13)-.5, alpha = .9, color = "pink",normed = True)
plt.xlabel("Self-identified Body Type")
plt.ylabel("Number of Responses")
ax.set_xticks(range(-1,12))
ax.set_xticklabels([ "","Used Up","Rather Not Say","Overweight","Full Figured","A Litte Extra","Curvy","Average","Skinny","Thin","Fit","Athletic","Jacked"], rotation = 90)
plt.xlim(-1,12)
plt.title("Body Type, by Gender - Normed Data")
plt.legend(["Male","Female"])
#plt.figure(figsize = (6,4))

plt.subplots_adjust(right = 1.8,wspace = .5)
fig2 = plt.gcf()
fig2.set_size_inches(9,5)
plt.tight_layout()
fig2.savefig("Body_Type_By_Gender_Combined_Hist.png")
plt.show()


# In[126]:


# Visualizing male/female responses(differences) in pet preference
dfmf1 = df[["sex_num","pet_num"]]
dfmf1 = dfmf1.dropna()
dff1 = dfmf1[dfmf1.sex_num == 1]
dfm1 = dfmf1[dfmf1.sex_num == 0]
#print(dfm1.pet_num.value_counts())
#print(dff1.pet_num.value_counts())

ax = plt.subplot(1,2,1)
plt.hist(dfm1.pet_num, bins=np.arange(16)-.5, alpha = .8, color = "blue")
plt.hist(dff1.pet_num, bins=np.arange(16)-.5, alpha = .9, color = "pink")
plt.xlabel("Pet Preferences")
plt.ylabel("Number of Responses")
ax.set_xticks(range(-1,15))
ax.set_xticklabels([ "","Dislikes Dogs and Dislikes Cats","Dislikes Cats","Has Dogs and Dislikes Cats","Likes Dogs and Dislikes Cats","Likes Dogs","Has Dogs","Has Dogs and Likes Cats","Has Dogs and Has Cats","Likes Dogs and Has Cats","Likes Dogs and Likes Cats","Likes Cats","Has Cats","Dislikes Dogs and Likes Cats","Dislikes Dogs and Has Cats","Dislikes Dogs"], rotation = 90)
plt.xlim(-1,15)
plt.title("Pet Preference, by Gender")
plt.legend(["Male","Female"])
#plt.figure(figsize = (8,8))

# Proportionalizing the histogram to show relative responses from both genders
ax = plt.subplot(1,2,2)
plt.hist(dfm1.pet_num, bins=np.arange(16)-.5, alpha = .8, color = "blue",normed = True)
plt.hist(dff1.pet_num, bins=np.arange(16)-.5, alpha = .9, color = "pink",normed = True)
plt.xlabel("Pet Preferences")
plt.ylabel("Number of Responses")
ax.set_xticks(range(-1,15))
ax.set_xticklabels([ "","Dislikes Dogs and Dislikes Cats","Dislikes Cats","Has Dogs and Dislikes Cats","Likes Dogs and Dislikes Cats","Likes Dogs","Has Dogs","Has Dogs and Likes Cats","Has Dogs and Has Cats","Likes Dogs and Has Cats","Likes Dogs and Likes Cats","Likes Cats","Has Cats","Dislikes Dogs and Likes Cats","Dislikes Dogs and Has Cats","Dislikes Dogs"], rotation = 90)
plt.xlim(-1,15)
plt.title("Pet Preference, by Gender - Normed Data")
plt.legend(["Male","Female"])
#plt.figure(figsize = (8,8))

plt.subplots_adjust(right = 1.8,wspace = .5)
fig3 = plt.gcf()
fig3.set_size_inches(9,5)
plt.tight_layout()
fig3.savefig("Pet_Preference_By_Gender_Hist.png")
plt.show()


# In[127]:


# Visualizing male/female responses(differences) in food preference
dfmf2 = df[["sex_num","diet_num"]]
dfmf2 = dfmf2.dropna()
dff2 = dfmf2[dfmf2.sex_num == 1]
dfm2 = dfmf2[dfmf2.sex_num == 0]
#print(dfm2.diet_num.value_counts())
#print(dff2.diet_num.value_counts())

ax = plt.subplot(1,2,1)
plt.hist(dfm2.diet_num, bins=np.arange(19)-.5, alpha = .8, color = "blue")
plt.hist(dff2.diet_num, bins=np.arange(19)-.5, alpha = .9, color = "pink")
plt.xlabel("Food Preferences")
plt.ylabel("Number of Responses")
ax.set_xticks(range(-1,18))
ax.set_xticklabels([ "","Mostly Anything","Anything","Strictly Anything","Mostly Vegetarian","Vegetarian","Strictly Vegetarian","Mostly Vegan","Vegan","Strictly Vegan","Mostly Kosher","Kosher","Strictly Kosher","Mostly Halal","Halal","Strictly Halal","Mostly Other","Other","Strictly Other"], rotation = 90)
plt.xlim(-1,18)
plt.title("Food Preference, by Gender")
plt.legend(["Male","Female"])
#plt.figure(figsize = (6,6))

# Proportionalizing the histogram to show relative responses from both genders
ax = plt.subplot(1,2,2)
plt.hist(dfm2.diet_num, bins=np.arange(19)-.5, alpha = .8, color = "blue",normed = True)
plt.hist(dff2.diet_num, bins=np.arange(19)-.5, alpha = .9, color = "pink",normed = True)
plt.xlabel("Food Preferences")
plt.ylabel("Number of Responses")
ax.set_xticks(range(-1,18))
ax.set_xticklabels([ "","Mostly Anything","Anything","Strictly Anything","Mostly Vegetarian","Vegetarian","Strictly Vegetarian","Mostly Vegan","Vegan","Strictly Vegan","Mostly Kosher","Kosher","Strictly Kosher","Mostly Halal","Halal","Strictly Halal","Mostly Other","Other","Strictly Other"], rotation = 90)
plt.xlim(-1,18)
plt.title("Food Preference, by Gender - Normed Data")
plt.legend(["Male","Female"])
#plt.figure(figsize = (6,6))

plt.subplots_adjust(right = 1.8,wspace = .5)
fig4 = plt.gcf()
fig4.set_size_inches(9,5)
plt.tight_layout()
fig4.savefig("Diet_Preference_By_Gender_Hist.pdf")
plt.show()


# In[128]:


# Creating a modeling DataFrame (mdf) with just the features I plan to model
mdf = df[['sex_num','diet_num','body_num','pet_num','essay_len']]


# In[129]:


# Removing any rows from the new DataFrame with NaN and returning a "cleaned" version: mdfc 
mdfc = mdf.dropna()
print(mdfc.head())


# In[130]:


print(mdfc.info())


# In[131]:


# Looking at general correlations between these features
mdfc.corr()


# In[132]:


### STATEMENT OF PROPOSED QUESTION

# I will be looking to see if a person's sex can be accurately predicted based upon their responses
# to the questions about body type, diet, and pet ownership, along with essay length.
 
# Further visualizing the pre- and post-cleaned sex response data...
# These histograms compare the male(value 0)/female (value 1) proportions in both the original and the cleaned
# data to ensure cleaning didn't materially change the shape of the male/female pool


# In[149]:


plt.subplot(1,2,1)
plt.hist(mdf.sex_num, bins=2)
plt.xlabel("Sex: male(left), female(right)")
plt.ylabel("Frequency = number of respondents")
plt.xlim(0,1)

plt.subplot(1,2,2)
plt.hist(mdfc.sex_num, bins=2)
plt.xlabel("Sex: male(left), female(right)")
plt.ylabel("Frequency = number of respondents")
plt.xlim(0,1)
plt.subplots_adjust(right = 1.8,wspace = .5)
fig5 = plt.gcf()
fig5.set_size_inches(9,5)
plt.tight_layout()
fig5.savefig("Male_Female_Pre_and_Post_Cleaning_Hists.png")
plt.show()


# In[134]:


# Scaling the data in the mdfc DataFrame to create mdfcs
from sklearn.preprocessing import MinMaxScaler

feature_data = mdfc[['diet_num', 'body_num', 'pet_num', 'essay_len']]

x = feature_data.values
min_max_scaler = MinMaxScaler()
mdfc_scaled = min_max_scaler.fit_transform(x)

mdfcs = pd.DataFrame(mdfc_scaled, columns=feature_data.columns)


# In[135]:


print(mdfcs.head())


# In[136]:


data_points = mdfcs.values.tolist()


# In[137]:


#print(data_points)


# In[138]:


labels = mdfc.sex_num.values.tolist()


# In[139]:


#print(labels)


# In[140]:


# RUNNING CLASSIFICATION TECHNIQUES

# First, splitting the data into training and test sets:

from sklearn.model_selection import train_test_split
train_data,test_data,train_labels,test_labels = train_test_split(data_points,labels,test_size = .2,random_state = 1)
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix


# In[168]:


# Classifying with K-Nearest Neighbors...used n_neighbors = 2, since I know the labels are male/female only

from sklearn.neighbors import KNeighborsClassifier
classifierK = KNeighborsClassifier(10)
classifierK.fit(train_data,train_labels)
predKNC = classifierK.predict(test_data)
print(accuracy_score(test_labels, predKNC))
print(recall_score(test_labels, predKNC))
print(precision_score(test_labels, predKNC))
print(f1_score(test_labels, predKNC))
accKNC = accuracy_score(test_labels,predKNC)


# In[169]:


# Classifying with MultinomialNB
from sklearn.naive_bayes import MultinomialNB
classifierMNB = MultinomialNB()
classifierMNB.fit(train_data,train_labels)
predMNB = classifierMNB.predict(test_data)
print(accuracy_score(test_labels, predMNB))
#print(recall_score(test_labels, predMNB))
#print(precision_score(test_labels,predMNB))
#print(f1_score(test_labels, predMNB))
accMNB = accuracy_score(test_labels,predMNB)
print(classifierMNB.predict_proba(test_data))


# In[170]:


# Classifiying with Support Vector Machines, using a radial bias kernel
# (came up slightly more accurate than a polynomial kernel).
from sklearn.svm import SVC
classifierSVM = SVC(kernel = "rbf")
classifierSVM.fit(train_data,train_labels)
print(classifierSVM.score(test_data,test_labels)) #printing an alternate score to see which it corresponds to (accuracy, precision, recall, etc.)
predSVM = classifierSVM.predict(test_data)
print(accuracy_score(test_labels,predSVM))
print(recall_score(test_labels,predSVM))
print(precision_score(test_labels,predSVM))
print(f1_score(test_labels,predSVM))
accSVM = accuracy_score(test_labels,predSVM)


# In[171]:


# RUNNING REGRESSION TECHNIQUES


# In[172]:


# Using Multiple Linear Regression
from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(train_data,train_labels)
print(mlr.score(train_data,train_labels))
print(mlr.score(test_data,test_labels))


# In[173]:


# Trying a KNN Regression - note: this scores as negative, with slightly less negative results from a "uniform"
# weighting than a "distance" weighting.
from sklearn.neighbors import KNeighborsRegressor
regressorK = KNeighborsRegressor(2,weights = "uniform")
regressorK.fit(train_data,train_labels)
print(regressorK.score(test_data,test_labels))


# In[167]:


score_data = [accKNC,accMNB,accSVM]
score_labels = ['K-Nearest Neighbors','Multinomial Naive Bayes','Support Vector Machine']
plt.title("Accuracy Comparison - 3 Classification Methods")
plt.ylabel("Accuracy Scores")
plt.plot(score_labels,score_data)
fig6 = plt.gcf()
fig6.set_size_inches(9,5)
plt.tight_layout()
fig6.savefig("Accuracy_Comparison_3_Classification_Methods.png")
plt.show()

