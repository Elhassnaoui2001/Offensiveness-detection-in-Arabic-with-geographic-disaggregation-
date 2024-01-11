from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model
import streamlit as st
import pandas as pd

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


df1 = pd.read_csv('dataset/Off.csv', encoding='utf-16')
df1=df1.dropna()
# sorting by first name
df1.sort_values("Comment", inplace=True)
# dropping ALL duplicate values
df1.drop_duplicates(subset="Comment",
                     keep=False, inplace=True)
df1.Comment = df1.Comment.apply(cleaning_content)
X = df1.Comment.values
Y = df1.Majority_Label.values

X_train,X_val,Y_train,Y_val = train_test_split(X,Y,test_size=0.1, random_state = random.seed(42))

max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
# Load the best model
best_model = load_model('best_model_LSTM (1).h5')
# Example sentence
# sentence = 'I am happy'


# # Display or use the predicted class as needed
# print("Predicted Class:", predicted_class)


def main():
    st.title('Offensive Text Analysis')

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
    # Assuming tokenizer is the tokenizer used during training
    
    # Tokenize and pad the sequence
    sequences = tok.texts_to_sequences([comment])
    padded_sequence = pad_sequences(sequences, maxlen=max_len)  # Adjust max_length based on your training data

    # Make predictions using the loaded model
    predictions = best_model.predict(padded_sequence)

    # Assuming binary classification, extract the predicted class
    predicted_class = 0 if predictions[0][0] < 0.5 else 1
     

    # image_filename = dialect_image_mapping.get(dialect_pred.argmax().item(), 'default.png')

    # st.image(image_filename, caption='Predicted Dialect', use_column_width=True)

    st.write('Comment:', comment)
    if predicted_class == 1 :
            st.write('Prediction: Offensive')
    else :
            st.write('Prediction: Non-Offensive')    



def process_and_predict_csv_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = ['Comment', 'Majority_Label']
    df.Comment = df.Comment.apply(cleaning_content)
    # Assuming df is your DataFrame
    texts = df['Comment'].tolist()

    # Tokenize and pad the sequences
    sequences = tok.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    
    predictions = best_model.predict(padded_sequences)

    # Assuming binary classification, extract the predicted class
    predicted_classes = [0 if pred[0] < 0.5 else 1 for pred in predictions]

    # Add the predicted classes to the DataFrame
    df['Majority_Label'] = predicted_classes

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
