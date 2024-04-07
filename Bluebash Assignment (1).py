#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string


# In[2]:


df=pd.read_csv(r"C:\Users\kushw\Downloads\DS_Case_Study_beer-ratings_2020 (2)\DS_Case_Study_beer-ratings_2020\train.csv")


# In[3]:


df


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


train_data=df.drop(["review/timeStruct","review/timeUnix","user/ageInSeconds","user/birthdayRaw","user/birthdayUnix","user/gender"],axis=1, inplace=False)
train_data


# In[7]:


train_data.isna().sum()


# In[8]:


train_data.dropna(inplace=True)

train_data.reset_index(drop=True, inplace=True)


# In[9]:


train_data


# In[10]:


train_data.drop_duplicates(subset=['beer/ABV','beer/beerId','beer/brewerId','beer/name','beer/style','review/appearance','review/aroma','review/overall','review/palate','review/taste','review/text'], inplace=True)
train_data.reset_index(drop=True, inplace=True)


# In[11]:


train_data


# In[12]:


len(train_data["beer/beerId"].unique())


# In[13]:


len(train_data["beer/style"].unique())


# In[14]:


len(train_data["review/overall"].unique())


# In[15]:


len(train_data["user/profileName"].unique())


# In[16]:


len(train_data["index"].unique())


# In[17]:


train_data.groupby(["user/profileName"]).count().sort_values(by="index", ascending=False)


# In[ ]:





# In[18]:


beer_rating_counts = train_data.groupby(['beer/beerId', 'review/overall']).size().reset_index(name='count')


# In[19]:


plt.figure(figsize=(12, 6))
plt.bar(beer_rating_counts['review/overall'], beer_rating_counts['count'], color='skyblue')
plt.title('Count of Beer ID vs Rating')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.xticks(range(1, 6))  
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[20]:


beer_rating_counts.sort_values(by="count", ascending=False)


# In[21]:


train_data[train_data['beer/beerId'] == 22450.0]


# In[ ]:





# In[ ]:





# In[22]:


beer_ratings_summary = train_data.groupby('beer/beerId').agg({'review/overall': ['mean', 'count'],
                                                               'review/appearance': 'mean',
                                                               'review/aroma': 'mean',
                                                               'review/palate': 'mean',
                                                               'review/taste': 'mean'}).reset_index()

beer_ratings_summary.columns = ['beer/beerId', 
                                'mean_overall_rating', 'count_of_occurrences',
                                'mean_appearance_rating',
                                'mean_aroma_rating', 
                                'mean_palate_rating', 
                                'mean_taste_rating']


# In[23]:


beer_ratings_summary.sort_values(by="count_of_occurrences", ascending=False)


# In[24]:


plt.figure(figsize=(12, 6))
plt.scatter(beer_ratings_summary['beer/beerId'], beer_ratings_summary['mean_overall_rating'], color='skyblue')
plt.title('Mean Overall Rating vs. Beer ID')
plt.xlabel('Beer ID')
plt.ylabel('Mean Overall Rating')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[25]:


plt.figure(figsize=(12, 6))
plt.scatter(beer_ratings_summary['mean_overall_rating'], beer_ratings_summary['count_of_occurrences'], color='skyblue')
plt.title('count of occurrences vs. mean overall rating')
plt.xlabel('mean overall rating')
plt.ylabel('count of Occurrence')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[26]:


plt.figure(figsize=(12, 6))
plt.scatter(beer_ratings_summary['mean_overall_rating'], beer_ratings_summary['mean_appearance_rating'], color='skyblue')
plt.title('mean_appearance_rating vs. mean overall rating')
plt.xlabel('mean overall rating')
plt.ylabel('mean_appearance_rating')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[27]:


plt.figure(figsize=(12, 6))
plt.scatter(beer_ratings_summary['mean_overall_rating'], beer_ratings_summary['mean_aroma_rating'], color='skyblue')
plt.title('mean_aroma_rating vs. mean overall rating')
plt.xlabel('mean overall rating')
plt.ylabel('mean_aroma_rating')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[28]:


plt.figure(figsize=(12, 6))
plt.scatter(beer_ratings_summary['mean_overall_rating'], beer_ratings_summary['mean_palate_rating'], color='skyblue')
plt.title('mean_palate_rating vs. mean overall rating')
plt.xlabel('mean overall rating')
plt.ylabel('mean_palate_rating')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[29]:


plt.figure(figsize=(12, 6))
plt.scatter(beer_ratings_summary['mean_overall_rating'], beer_ratings_summary['mean_taste_rating'], color='skyblue')
plt.title('mean_taste_rating vs. mean overall rating')
plt.xlabel('mean overall rating')
plt.ylabel('mean_taste_rating')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[30]:


train_data


# In[31]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
one_hot_encoded = pd.get_dummies(train_data['beer/style'], prefix='style')

train_data_encoded = pd.concat([train_data, one_hot_encoded], axis=1)
one_hot_encoded_beerId = pd.get_dummies(train_data['beer/beerId'], prefix='style')
train_data_encoded = pd.concat([train_data, one_hot_encoded_beerId], axis=1)
train_data_encoded


# In[32]:


train_data=train_data_encoded


# In[33]:


train_data


# In[34]:


train_data['review/text'] = train_data['review/text'].str.lower()

train_data['review/text'] = train_data['review/text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

stop_words = set(stopwords.words('english'))
train_data['review/text'] = train_data['review/text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))


# In[ ]:





# In[35]:


tfidf_vectorizer = TfidfVectorizer(max_features=1000)  
tfidf_features = tfidf_vectorizer.fit_transform(train_data['review/text'])


# In[36]:


X = pd.concat([pd.DataFrame(tfidf_features.toarray()), train_data], axis=1)


# In[37]:


X


# In[38]:


X=X.drop(['index','beer/beerId','beer/brewerId','beer/name','beer/style','review/text','user/profileName'],axis =1)


# In[39]:


y = train_data['review/overall']


# In[40]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[42]:


X_train


# # Linear Regression

# In[43]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[44]:


y_pred_LR = model.predict(X_test)

mse_LR = mean_squared_error(y_test, y_pred_LR)
r2_LR = r2_score(y_test, y_pred_LR)

print("Mean Squared Error:", mse_LR)
print("R-squared (R2) Score:", r2_LR)


# In[ ]:





# In[45]:


y_pred_LR


# In[46]:


differences = np.abs(y_pred_LR - y_test)

under_threshold = np.sum(differences < 0.25)

above_threshold = np.sum(differences >= 0.25)

print("Predictions with a difference less than 0.25:", under_threshold)
print("Predictions with a difference of 0.25 or greater:", above_threshold)


# In[ ]:





# # K Nearest Neighbors

# In[47]:


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor


# In[48]:


knn_model = KNeighborsRegressor(n_neighbors=10)  
knn_model.fit(X_train, y_train)

y_pred_KNN = knn_model.predict(X_test)
mse_KNN = mean_squared_error(y_test, y_pred_KNN)
print("Mean Squared Error:", mse_KNN)
r2_KNN = r2_score(y_test, y_pred_KNN)
print("R-squared (R2) Score:", r2_KNN)


# In[49]:


differences = np.abs(y_pred_KNN - y_test)

under_threshold = np.sum(differences < 0.5)

above_threshold = np.sum(differences >= 0.5)

print("Predictions with a difference less than 0.5:", under_threshold)
print("Predictions with a difference of 0.5 or greater:", above_threshold)


# In[ ]:





# #  Decision Tree
# 

# In[50]:


from sklearn.tree import DecisionTreeRegressor


# In[51]:


dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)


# In[52]:


y_pred_DT = dt_model.predict(X_test)
mse_DT = mean_squared_error(y_test, y_pred_DT)
print("Mean Squared Error:", mse_DT)
r2_DT = r2_score(y_test, y_pred_DT)
print("R-squared (R2) Score:", r2_DT)


# In[53]:


differences = np.abs(y_pred_DT - y_test)

under_threshold = np.sum(differences < 0.5)

above_threshold = np.sum(differences >= 0.5)

print("Predictions with a difference less than 0.55:", under_threshold)
print("Predictions with a difference of 0.55 or greater:", above_threshold)


# In[ ]:





# # Tensor Flow

# In[54]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[55]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[56]:


model_TF = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  
])


# In[57]:


model_TF.compile(optimizer='adam', loss='mean_squared_error')


# In[58]:


model_TF.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)


# In[59]:


y_pred_TF = model_TF.predict(X_test_scaled)
mse_TF = mean_squared_error(y_test, y_pred_TF)
print("Mean Squared Error:", mse_TF)
r2_TF = r2_score(y_test, y_pred_TF)
print("R-squared (R2) Score:", r2_TF)


# In[60]:


y_pred_TF


# In[61]:


y_test


# In[62]:


np.abs(y_pred_TF.flatten() - y_test.values)


# In[63]:


differences = np.abs(y_pred_TF.flatten() - y_test.values)
under_threshold = np.sum(differences < 0.5)

above_threshold = np.sum(differences >= 0.5)

print("Predictions with a difference less than 0.5:", under_threshold)
print("Predictions with a difference of 0.5 or greater:", above_threshold)


# In[ ]:





# In[ ]:





# In[ ]:




