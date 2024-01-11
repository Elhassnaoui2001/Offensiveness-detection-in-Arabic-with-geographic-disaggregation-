import streamlit as st
import numpy as np
import pandas as pd
import re
import string
import random

import nltk
nltk.download('stopwords')
import emoji

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('dataset/dialcet.csv')

# df.columns
df.drop(['Unnamed: 0'], axis=1, inplace=True)
dfX = df
new_df = dfX[(dfX['dialect'] == 'morocco' ) | (dfX['dialect'] == 'algerian') | (dfX['dialect'] == 'lebanon' ) | (dfX['dialect'] == 'egypt') | (dfX['dialect'] == 'tunisian') | (dfX['dialect'] == 'qatar' ) | (dfX['dialect'] == 'iraq') | (dfX['dialect'] == 'libya') | (dfX['dialect'] == 'saudi_arabia' ) | (dfX['dialect'] == 'jordan')]
lebanon = new_df[new_df['dialect'] == 'lebanon'][:12234]
egypt = new_df[new_df['dialect'] == 'egypt'][:12234]
morocco = new_df[new_df['dialect'] == 'morocco'][:12234]
tunisian = new_df[new_df['dialect'] == 'tunisian'][:12234]
algerian = new_df[new_df['dialect'] == 'algerian'][:12234]
qatar = new_df[new_df['dialect'] == 'qatar'][:12234]

iraq = new_df[new_df['dialect'] == 'iraq']
saudi_arabia = new_df[new_df['dialect'] == 'saudi_arabia']
libya = new_df[new_df['dialect'] == 'libya']
jordan = new_df[new_df['dialect'] == 'jordan']

frames = [lebanon, egypt, morocco, tunisian, algerian, qatar, iraq, saudi_arabia, libya, jordan]
df = pd.concat(frames)

dialects = df['dialect'].unique()

lbl2idx = {d: i for i, d in enumerate(dialects)}

df['dialect'] = df['dialect'].map(lbl2idx)

arabic_stopwords = set(nltk.corpus.stopwords.words("arabic"))

arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations = arabic_punctuations + english_punctuations

def remove_urls (text):
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
    return text


def remove_emails(text):
    text = re.sub(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", "",  text, flags=re.MULTILINE)
    return text

# def remove_emoji(text):
#     return emoji.get_emoji_regexp().sub(u'', text)

def remove_emoji(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

def normalization(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text

def remove_stopwords(text):
    filtered_sentence = [w for w in text.split() if not w in arabic_stopwords]
    return ' '.join(filtered_sentence)

def cleaning_content(line):
    if (isinstance(line, float)):
        return None
    line.replace('\n', ' ')
    line = remove_emails(line)
    line = remove_urls(line)
    line = remove_emoji(line)
    nline = [w if '@' not in w else 'USERID' for w in line.split()]
    line = ' '.join(nline)
    line = line.replace('RT', '').replace('<LF>', '').replace('<br />','').replace('&quot;', '').replace('<url>', '').replace('USERID', '')


    # add spaces between punc,
    line = line.translate(str.maketrans({key: " {0} ".format(key) for key in punctuations}))

    # then remove punc,
    translator = str.maketrans('', '', punctuations)
    line = line.translate(translator)

    line = remove_stopwords(line)
    line=remove_diacritics(normalization(line))
    return line

def hasDigits(s):
    return any( 48 <= ord(char) <= 57  or 1632 <= ord(char) <= 1641 for char in s)

df.Twits = df.Twits.apply(cleaning_content)

X_train ,X_test , y_train , y_test = train_test_split(df['Twits'] , df['dialect'] , test_size=0.2 , random_state=2024)

from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_cv = v.fit_transform(X_train.values)



from joblib import dump, load

# Load the Naive Bayes model from the file
loaded_nb_model = load('naive_bayes_model.joblib')

# 





def main():
    st.title('Dialect Detection')

    option = st.selectbox('Choose an option:', ['Single Comment Prediction', 'CSV File Prediction'])

    if option == 'Single Comment Prediction':
        st.subheader('Single Comment Prediction')

        comment = st.text_area('Enter your comment:')
        if st.button('Predict'):
            process_and_predict_single_comment(comment)

    elif option == 'CSV File Prediction':
        st.subheader('CSV File Prediction')

        uploaded_file = st.file_uploader('Upload your CSV file', type=['csv', 'xlsx'])
        if uploaded_file is not None:
            if st.button('Predict'):
                process_and_predict_csv_file(uploaded_file)

def process_and_predict_single_comment(comment):
    comment = cleaning_content(comment)
    list = [comment]
    list_count = v.transform(list)
    # Make predictions using the loaded model
    
    predictions = loaded_nb_model.predict(list_count)[0]
     

    # image_filename = dialect_image_mapping.get(dialect_pred.argmax().item(), 'default.png')

    # st.image(image_filename, caption='Predicted Dialect', use_column_width=True)

    st.write('Comment:', comment)
    # st.write('Prediction:', predictions)

    if predictions == 0 :
            st.write('Dialect Predict: Lebanon')
    if predictions == 1 :
            st.write('Dialect Predict: Egypt')    
    if predictions == 2 :
            st.write('Dialect Predict: Morocco')
    if predictions == 3 :
            st.write('Dialect Predict: Tunisian') 
    if predictions == 4 :
            st.write('Dialect Predict: Algerian')
    if predictions == 5 :
            st.write('Dialect Predict: Quatar') 
    if predictions == 6 :
            st.write('Dialect Predict: Iraq')
    if predictions == 7 :
            st.write('Dialect Predict: Saudim Arabia')
    if predictions == 8 :
            st.write('Dialect Predict: Libya')
    if predictions == 9 :
            st.write('Dialect Predict: Jordan') 
            
            
 

                         
def process_and_predict_csv_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = ['Comment', 'Majority_Label']
    df.Comment = df.Comment.apply(cleaning_content)
    X_test_cv = v.transform(df.Comment)
    y_pred = loaded_nb_model.predict(X_test_cv)
    # Add the predicted values to the DataFrame
    df['predicted_dialect'] = y_pred

    # Map predicted labels to country names
    country_mapping = {
        0: 'Lebanon',
        1: 'Egypt',
        2: 'Morocco',  
        3: 'Tunisian',
        4: 'Algerian',
        5: 'Quatar',
        6: 'Iraq',
        7: 'Saudim Arabia',
        8: 'Libya',
        9: 'Jordan',  
        
    }

    df['Predicted_dialect'] = df['predicted_dialect'].map(country_mapping)
    # Drop the original 'label' column
    df.drop('predicted_dialect', axis=1, inplace=True)
    df.drop('Majority_Label', axis=1, inplace=True)

    st.dataframe(df)
    
    # Save annotated dataframe to CSV
    download_link = st.download_button(
                    label="Download CSV",
                    data=df.to_csv(index=False),
                    file_name="annotated_data.csv",
                    key="download_annotated_csv",
                )

                # Display the download link
    st.markdown(download_link, unsafe_allow_html=True)
if __name__ == '__main__':
    main()
