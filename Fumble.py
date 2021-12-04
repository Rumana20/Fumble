
import streamlit as st
import pandas as pd
import pickle
from io import StringIO
pip install xlrd
import xlrd



def main():
   
    html_temp = """
    <div style="background-color:#184B44;padding:5px">
    <h1 style="text-align:center;"> Fumble </h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    
import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import streamlit as st

st.markdown('##')

okc = pd.read_excel("D:/5th Sem/Project/Copy of User Details_Faf.xlsx",engine=("xlrd"))

st.title("Home")
if __name__=='__main__':
    main() 
    
st.markdown('##')  
    
if st.checkbox("Show data"):
    st.write(okc)
  
st.markdown('##')  
 
   
'''

Fill These List Of Questions To Get Your Match

'''   
st.markdown('##')

     
     
#input_df = pd.DataFrame()
okc1 = pd.DataFrame() 

#def user_input_features():
name = st.text_input("Name","Type..") 
age = st.number_input("age",18,79) 
status = st.selectbox("status", ["Single","in a relationship","Unknown"]) 
gender = st.selectbox("gender", ["Female","Male"]) 
orientation = st.selectbox("orientation", ["Straight","Bisexual","Gay"]) 
body_type = st.selectbox("body_Type", ["Fit","Average","Curvy","Thin","Overweight","Rather not say"]) 
education = st.selectbox("education", ["college/university","Masters and above","other","Two-year college","High school","Med / Law school"]) 
ethnicity= st.selectbox("ethnicity", ["White","Asian","Hispanic","African American","Mixed","Unknown","others"]) 
religion = st.selectbox("religion", ["Agnosticism","Atheism","Christianity","Catholicism","Judaism","Buddhism","Islam","Hinduism","Unknown","others"]) 
smokes = st.selectbox("smokes", ["Yes","No"]) 
drink = st.selectbox("drink", ["Yes","No"]) 
diet = st.selectbox("diet", ["Anything","Vegan","Vegetarian","Halal","Kosher","other"]) 
speaks = st.text_input("speaks \n Enter your 2nd preferred language","Type..") 
sign = st.selectbox("sign", ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpion","Sagittarius","Capricorns","Aquarius","Pisces"]) 
offspring = st.selectbox("offspring", ["Wants Kids","Does not want kids","Has kid","Unknown"]) 
drugs = st.selectbox("drugs", ["Yes","No"]) 
height = st.number_input("height \n (In inches)",30,100) 
income = st.number_input("income",0,100000) 
pets = st.selectbox("pets", ["Likes Cats and Dogs","Dislikes Cats and Dogs","Likes only cats","Likes only Dogs","Unknown"]) 
job = st.selectbox("job", ["Office/Professional","Science/Tech","Business Management","Creative"]) 
essay0 = st.text_input("essay0 My self summary","Type..") 
essay1 = st.text_input("essay1 What I am doing with my life","Type..") 
essay2 = st.text_input("essay2 I am really good at ","Type..") 
essay3 = st.text_input("essay3 The first thing people usually notice about me","Type..") 
essay4 = st.text_input("essay4 Favourite books, Movies, Show,Music, Food","Type..") 
essay5 = st.text_input("essay5 The 6 things that I could never do without","Type..") 
essay6 = st.text_input("essay6 I spend a lot of time thinking about","Type") 
essay7 = st.text_input("essay7 On a typical Friday night I am","Type..") 
essay8 = st.text_input("essay8 The most private thing I am willing to admit","Type..") 
essay9 = st.text_input("essay9 You should message me if","Type..") 

   
#if st.button("ok"):
data = {"Name": name, "age": age, 
     "gender": gender, "orientation":orientation,
     "status":status,"education":education, "ethnicity":ethnicity, 
                 "religion" : religion,"smokes":smokes, "drink":drink, "body_type":body_type,"diet":diet,"job":job,
                 "speaks":speaks,"sign":sign,"offspring":offspring,"drugs":drugs,"height":height,
                 "income":income,"pets":pets,"essay0":essay0,"essay1":essay1,"essay2":essay2,
                 "essay3":essay3,"essay4":essay4,"essay5":essay5,"essay6":essay6,"essay7":essay7,
                 "essay8":essay8,"essay9":essay9}

data = pd.DataFrame(data,index=[0])
data.columns = data.columns.str.lower()
okc.columns = okc.columns.str.lower()
features = pd.DataFrame(data) 
         
         #return features
         #input_df = user_input_features()
         
#features.columns
        
#input_df = pd.DataFrame()
#okc1 = pd.DataFrame()

#give = ""
#if st.button("join"):
#     input_df = user_input_features()

#okc1 = pd.concat([data,okc],axis=1,join="inner",ignore_index=True) 
#st.write(okc1)

#if st.button("Join"): 
okc1 = pd.concat([features,okc],axis=0,join="inner",ignore_index=True) 

#okc1.columns
#okc1.head()




ok= okc1.copy(deep=True)

ok['essay']=ok[['essay0','essay1','essay2','essay3','essay4','essay5','essay6','essay7','essay8','essay9']].apply(lambda x: ' '.join(x), axis=1)
ok.drop(['essay0','essay1','essay2','essay3','essay4','essay5','essay6','essay7','essay8','essay9'],axis=1,inplace=True)

corpus_df = ok.copy(deep=True)

#regex
import re
# import natural language toolkit
import nltk
#beautiful soup
from bs4 import BeautifulSoup
#string for punctuation
import string
#stop word list
from nltk.corpus import stopwords
#import tokenizer
from nltk.tokenize import RegexpTokenizer
#import Lemmatizer
from nltk.stem import WordNetLemmatizer
#import stemmer
from nltk.stem.porter import PorterStemmer
#import html parser just in case BS4 doesn't work
import html.parser
# stop words
from nltk.corpus import stopwords

corpus_df['corpus'] = ok[['age', 'status', 'gender', 'orientation', 'body_type', 'diet', 'drink',
       'drugs', 'education', 'ethnicity', 'height', 'income', 'job',
       'offspring', 'pets', 'religion', 'sign', 'smokes', 'speaks', 'essay']].astype(str).agg(' '.join, axis=1)
corpus_df = corpus_df.astype(str)

# replaced \n
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace('\n', ' '))

# replace all nan's and removed apostrophe
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace('nan', ' '))
#removed apostrophe
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("\'", ""))
#remove dashes
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("-'", ""))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("--'", ""))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("='", ""))
#remove forward slash
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("/", ""))
#remove periods
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace(".", " "))

#remove colon
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace(":", " "))

# remove comma
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace(",", " "))

# remove left parentheses
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("(", " "))

#remove right parentheses
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace(")", " "))

#remove question marks
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("?", " "))

#remove ! mark
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("!", " "))

#remove semicolon marks
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace(";", " "))
# remove quotation marks
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace('"', " "))

# remove numbers
corpus_df['corpus'] = corpus_df['corpus'].str.replace('\d+', '')



corpus_list = corpus_df['corpus']

#tf - idf vectorization

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

tfidf = TfidfVectorizer(stop_words = "english", ngram_range = (1,3), max_df=0.8, min_df=0.2) 

corpus_tfidf = tfidf.fit(corpus_list)

corpus_2d = pd.DataFrame(tfidf.transform(corpus_list).todense(),
                   columns = tfidf.get_feature_names(),)

tfidf_vec = tfidf.fit_transform(corpus_list)

corpus_2d.head()

corpus_mat_sparse = csr_matrix(corpus_2d.values)

pd.set_option('display.max_columns',25)
pd.set_option('expand_frame_repr', False)

#Model specification
from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute') # cosine takes 1-cosine( as cosine distance)
model_knn.fit(corpus_mat_sparse)

#recommendation algorithm (cosine)
def rec(query_index):
  distances, indices = model_knn.kneighbors(corpus_2d.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 21)

  for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}: \n'.format(corpus_2d.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2} \n '.format(i, corpus_2d.index[indices.flatten()[i]],distances.flatten()[i]))
  for i in indices:
    print(okc1.loc[i,:])
 
    
 
def rec(query_index):
  distances, indices = model_knn.kneighbors(corpus_2d.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)
  result= pd.DataFrame()
  for i in indices:
    result= result.append(okc1.iloc[i,:])
  result['similarity distance']= distances.flatten()
  return result[['name',"similarity distance","age","status","orientation","body_type","ethnicity","religion","smokes","drink","diet","essay0"]]    

st.markdown('##')
st.markdown('##')   

if st.button("Join"):
    st.write("Your Matches are:")
    st.write(rec(0))
    
#okc1.columns



    

         

        


