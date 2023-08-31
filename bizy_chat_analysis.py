import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as scp
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
import string
from wordcloud import WordCloud
from nltk.util import ngrams
import re

### Load Data
df = pd.read_excel('bizy_rekap_17-08-2023to26-08-2023.xlsx')
id_stopword_dict = pd.read_csv('stopwordbahasa.csv', header=None)
id_stopword_dict = id_stopword_dict.rename(columns={0: 'stopword'})

tambahan = pd.DataFrame({'stopword': ['ya', 'ga', 'gak', 'ok', 'hai',
                                      'gimana', 'halo', 'yg', 'kak',
                                      'nya', 'aja', 'sih', 'kalo',
                                      'utk', 'udah', 'hallo', 'iya',
                                      'sy']})

id_stopword_dict = pd.concat([id_stopword_dict, tambahan], ignore_index=True)

df['created_at'] = pd.to_datetime(df['created_at'])
### Filtered by human type and user id
filtered_df = df[(df.type == 'human') & (df['user_id'].str.len() == 36)]


def remove_punc(text_df):
    # Define translation table
    translator = str.maketrans('', '', string.punctuation)

    # Apply translation table to text column
    res = text_df.apply(lambda x: x.translate(translator))
    # filtered_df['content'] = filtered_df['content'].str.replace('[^\w\s]','')

    # split_word = pd.Series(' '.join(res).split()).value_counts()

    #res = ' '.join(['' if word in id_stopword_dict.stopword.values else word for word in res.str.split(' ')])
    res = res.apply(lambda x: ' '.join([word for word in x.split() if word not in (id_stopword_dict.stopword.values)]))
    #res = re.sub('  +', ' ', str(res)) # Remove extra spaces
    res = res.str.strip()
    return res


def create_word_cloud(date1, date2):
    res = filtered_df[(filtered_df['created_at'] > date1) & (filtered_df['created_at'] < date2)]['content']
    # Start with one review:
    text = remove_punc(res).to_string()

    # Create and generate a word cloud image:
    wordcloud = WordCloud().generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot()
    #return wordcloud


def custom_ngrams(n):
  # Example sentence
  sentences = remove_punc(filtered_df['content'])
  res_list = []

  for s in sentences:
    word = s.split()

    # Tokenize the sentence into words
    #words = sentence.split()

    # Create bigrams from the list of words
    xgrams = ngrams(word, n)

    # Print the bigrams
    for xgram in xgrams:
      res_list.append(xgram)


  return res_list


def find_product_amount(product, time1, time2):
    selected = filtered_df[(filtered_df['created_at'] > time1) & (filtered_df['created_at'] < time2)]['content']
    # Start with one review:
    text = remove_punc(selected)
    split_word = pd.Series(' '.join(text).split()).value_counts()
    amount = split_word[product]


    return amount

def chat_duration(uuid_conv):
    res_df = pd.DataFrame(columns = ['user_id', 'duration'])
    temp=uuid_conv['user_id'][2]
    timestamp = []
    max_temp = 0
    min_temp = 0

    post = 0

    for index in uuid_conv.index:
        curr_user = uuid_conv['user_id'][index]
        #rint(curr_user)
        #print(temp)
        if curr_user == temp:
            timestamp.append(uuid_conv['created_at'][index])
        else:
            # count duration then store id and duration in df
            max_temp = timestamp[0]
            min_temp = timestamp[len(timestamp)-1]
            d = (abs(max_temp-min_temp)).total_seconds()
            
            #res_df=res_df.append(pd.DataFrame({'user_id': temp, 'duration': d}), ignore_index=True)
            #pd.DataFrame({"A": range(3)})
            # update temp id,max,min,duration, then empty the list
            res_df.loc[post] = [temp, d]
            post += 1
            temp = curr_user
            max_temp=0
            min_temp=0
            duration_temp=0 
            timestamp=[]

            # update list
            timestamp.append(uuid_conv['created_at'][index])

    return res_df

# Read excel file with sheet namex
df2 = pd.read_excel('bizy_rekap_29-08-2023to31-08-2023.xlsx', sheet_name='RAW')
df3 = pd.read_excel('bizy_rekap_29-08-2023to31-08-2023.xlsx', sheet_name='FE_USER')

new_df1 = df[(df['user_id'].str.len() == 36) ]
new_df2 =df2[(df2['user_id'].str.len() == 36) ]
new_df3 =df3[(df3['user_id'].str.len() == 36) ]

uuid_conv = pd.concat([new_df1, new_df2, new_df3], axis=0, ignore_index=True)
uuid_conv['created_at'] = pd.to_datetime(uuid_conv['created_at'])
#uuid_conv = new_df1 , axis=0, ignore_index=True


st.markdown("<h1 style='text-align: center; color: white;'>Bizy Chat History Analysis</h1>", unsafe_allow_html=True)


st.header("Duration Time per Session (in Second)")
st.dataframe(chat_duration(uuid_conv))

st.header("Word Cloud Based on Date Range")
date1 = st.text_input("Enter Your date1", "Type Here ...", key="1")
date2 = st.text_input("Enter Your date2", "Type Here ...", key="2")
if(st.button('Word Cloud')):
	create_word_cloud(date1, date2)


st.header("Amount of Product Show Up in Chat History")
product = st.text_input("Enter Your product", "Type Here ...", key="3")
pdate1 = st.text_input("Enter Your pdate1", "Type Here ...", key="4")
pdate2 = st.text_input("Enter Your pdate2", "Type Here ...", key="5")
if(st.button('Submit')):
	st.text(str(find_product_amount(product, pdate1, pdate2)))

st.header("Chat History N-Grams")
level = st.slider("Select the level", 1, 5)
xgrams = custom_ngrams(level)
for x in xgrams:
    st.text(x)
