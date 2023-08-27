#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas import read_csv
import ast
import re
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


credits=read_csv("tmdb_5000_credits.csv")
movies=read_csv("tmdb_5000_movies.csv")


# In[3]:


movies=movies.merge(credits,on="title")
del credits


# In[4]:


movies=movies[["id","title","genres","keywords","overview","cast","crew"]]


# In[5]:


movies.info()


# In[6]:


# Check null and empty Value and repeated
movies.dropna(inplace=True)
movies.drop_duplicates(inplace=True)


# In[7]:


movies.info()
movies["keywords"]


# In[8]:


def convertToKeyword(strto):
    lis:list[str]=ast.literal_eval(strto)
    newlist=[]
    for i in lis:
        newlist.append(i["name"])
    return newlist


# In[9]:


movies["genres"]=movies["genres"].apply(convertToKeyword)
movies["keywords"]=movies["keywords"].apply(convertToKeyword)


# In[10]:


# get tpo 3 actor


def getCasr(cast):
    cast=ast.literal_eval(cast)
    list1=[]
    counter=0
    for i in cast:
        list1.append(i["character"])
        counter+=1
        if counter==3:
            break
    return list1

movies["cast"]=movies["cast"].apply(getCasr)


# In[11]:


movies.head()


# In[12]:


def getDirector(lis):
    lis=ast.literal_eval(lis)
    list1=[]
    for i in lis:
        if i["job"]=="Director":
            list1.append(i["name"])
            return list1

movies["crew"]=movies["crew"].apply(getDirector)


# In[13]:


movies.dropna(inplace=True)


# In[14]:


movies["cast"]=movies["cast"].apply(lambda x:[i.replace(" ","") for i in x])
movies["genres"]=movies["genres"].apply(lambda x:[i.replace(" ","") for i in x])
movies["keywords"]=movies["keywords"].apply(lambda x:[i.replace(" ","") for i in x])



# In[15]:


movies["crew"]=movies["crew"].apply(lambda x:[i.replace(" ","") for i in x] )
movies.info()


# In[16]:


movies["overview"]=movies["overview"].apply(lambda x:x.split())


# In[17]:


movies["tags"]=movies["overview"]+movies["genres"]+movies["keywords"]+movies["cast"]+movies["crew"]


# In[18]:


movies=movies[["id","title","tags"]]


# In[19]:


movies["tags"]=movies["tags"].apply(lambda x:" ".join(x))


# In[20]:


movies["tags"]=movies["tags"].apply(lambda x:x.lower())


# In[21]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def stemPor(str1):
    y=[]
    for i in str1.split():
        y.append(ps.stem(i))
    return " ".join(y)

movies["tags"]=movies["tags"].apply(stemPor)


# In[22]:


from sklearn.feature_extraction.text import CountVectorizer
countVector=CountVectorizer(stop_words='english',max_features=5000)


# In[23]:


model=countVector.fit_transform(movies["tags"]).toarray()




# In[24]:


import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)
countVector.get_feature_names_out()


# In[25]:




