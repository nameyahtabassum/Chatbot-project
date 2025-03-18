import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random 
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl','rb'))

model = load_model("Chatbot_Model.h5")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    print("\nDEBUG: Tokenized Sentence Words ->", sentence_words)
    print("DEBUG: Known Words in Model ->", words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1  # Mark word as present

    print("DEBUG: Final Bag of Words ->", bag)
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    print("Predicted Raw Scores:", res)  # Debugging: See what the model outputs

    ERROR_THRESHOLD = 0.5
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    if not results:  # If no intent is confidently predicted
        print("No confident prediction. Returning 'unknown' intent.")
        return [{'intent': 'unknown', 'probability': '0'}]

    results.sort(key=lambda x: x[1], reverse=True)

    print("Top Prediction:", classes[results[0][0]], "with probability", results[0][1])  # Debugging

    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    list_of_intents = intents_json['intents']

    tag = intents_list[0]['intent']
    for i in list_of_intents:
        if i['tag']==tag:
            result = random.choice(i['responses'])
            break
    return result
print("Great! Bot is running!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)