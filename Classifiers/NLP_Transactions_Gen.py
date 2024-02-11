import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

# Sample transaction data (replace with your dataset)
transaction_descriptions = ["Grocery shopping", "Dining out at Italian restaurant", "Gas station purchase", "Online shopping at Amazon"]
df = pd.read_table('transactions.csv',names=['Descriptions'], delimiter=',')
df[['Descriptions','Amount','Category']] = df['Descriptions'].str.split(',',expand=True)
df = df.drop([0, 1])
df1 = df.loc[:,['Descriptions']]
transaction_descriptions = np.array(df1['Descriptions'])

# Tokenize the transaction descriptions
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(transaction_descriptions)

# Convert text to sequences of integers
sequences = tokenizer.texts_to_sequences(transaction_descriptions)

# Pad sequences to ensure consistent lengths
sequences = pad_sequences(sequences, padding='post')

# Create training data (X) and target data (y)
X = sequences[:, :-1]  # Use all but the last word as input
y = sequences[:, 1:]   # Predict the next word

vocab_size = len(tokenizer.word_index) + 1

# Define and compile an RNN model
model = keras.Sequential([
    Embedding(input_dim=vocab_size, output_dim=64),
    LSTM(64, return_sequences=True),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(X, y, epochs=200)

# Generate a new transaction description
seed_text = "ATM"
seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]

generated_sequence = seed_sequence.copy()
for _ in range(4):  # Generate a 5-word description
    next_word_probs = model.predict(np.array([generated_sequence]))[0][-1]
    next_word_index = np.random.choice(range(vocab_size), p=next_word_probs)
    generated_sequence.append(next_word_index)

generated_text = ' '.join([tokenizer.index_word[index] for index in generated_sequence])
print(generated_text)



