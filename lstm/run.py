import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("book_language_model.h5")

book_path = r"C:\Users\arshb\OneDrive\Desktop\pattern\lstm\alchamist.txt"
with open(book_path, "r", encoding="utf-8") as file:
    book_text = file.read()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([book_text])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in book_text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])

def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

seed_text = input("Enter text: ")
next_words = int(input("Number of words to be generate: "))

generated_text = generate_text(seed_text.lower(), next_words, max_sequence_len)
print(generated_text)
