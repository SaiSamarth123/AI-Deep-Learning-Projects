# NLP and ChatBots (spaCy, NLTK and LSTM)

# Import required libraries for machine learning and natural language processing
import tensorflow as tf
import spacy
import nltk
from nltk.corpus import stopwords

# Load English tokenizer, tagger, parser, NER and word vectors from spaCy
nlp = spacy.load("en")

# Download and load stopwords from NLTK
stop_words = stopwords.words("english")


def preprocess_text(text):
    """
    Function to preprocess text by tokenizing and lemmatizing using spaCy, and removing stopwords.
    :param text: A string containing the text to be processed
    :return: A list of lemmatized tokens without stopwords
    """
    # Tokenize the text using spaCy
    doc = nlp(text)

    # Filter out stopwords and punctuation, and then lemmatize the tokens
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    return tokens


# Load past conversation data for training the chatbot
conversations = []
with open("conversations.txt", "r") as f:
    for line in f:
        # Process each line in the file and append to conversations list
        conversations.append(preprocess_text(line))

# Prepare training data for the LSTM model
X = []
y = []
for conversation in conversations:
    for i in range(len(conversation) - 1):
        # Append each word and its subsequent word to the respective lists
        X.append(conversation[i])
        y.append(conversation[i + 1])

# Define the LSTM model architecture
model = tf.keras.Sequential(
    [
        # Embedding layer to transform indices into dense vectors of a fixed size
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64),
        # LSTM layer with 64 memory units
        tf.keras.layers.LSTM(64),
        # Dense layer to output probabilities over the vocabulary
        tf.keras.layers.Dense(vocab_size, activation="softmax"),
    ]
)

# Compile the model specifying the optimizer, loss function, and metrics to track
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model on the data for a fixed number of epochs
model.fit(X, y, epochs=100)

# Functionality to interact with the chatbot
while True:
    user_input = input("User: ")
    # Predict the response based on user input
    chatbot_response = model.predict(user_input)
    print("Chatbot: ", chatbot_response)

# Save the trained model to a file
model.save("chatbot_model.h5")
