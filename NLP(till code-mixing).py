import pandas as pd
import random
from googletrans import Translator
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np


df = pd.read_csv("C:\\Users\\Saksham Tripathi\\Downloads\\hindi_dataset.xlsx - hindi_dataset.csv")

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove links
    text = re.sub(r'#\S+', '', text)  # Remove hashtags
    text = re.sub(r'\n', ' ', text)  # Replace newlines 
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.strip()  # Remove leading spaces
  
    return text

data['text'] = data['text'].apply(clean_text)
  df = df.dropna() #dropping NaN values
def translate(sentence):
    translator = Translator(service_urls=['translate.google.com'])
    try:
        result = translator.translate(sentence, src='hi', dest='en').text
    except:
        result = ""
    return result

def code_mix(sentence, cm_index):
    words = sentence.split()
    n = len(words)
    num_translated = int(n * cm_index)
    indices = random.sample(range(n), num_translated)
    for i in indices:
        hindi_word = translate(words[i])
        if hindi_word:
            words[i] = hindi_word
    return " ".join(words)


cmi_values = [0.1, 0.3, 0.5, 0.7, 0.9]

num_sentences = 100


batch_size = 16
num_epochs = 5


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')

for cm_index in cmi_values:
  
    df_cm = df.sample(n=num_sentences)
    df_cm['text_cm'] = df_cm['text'].apply(lambda x: code_mix(x, cm_index))


    inputs = tokenizer(df_cm['text_cm'].tolist(), padding=True, truncation=True, max_length=128)

   
    labels = df_cm['label'].replace({'HATE': 1, 'OFFN': 2, 'PRFN': 0})
