import os

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd
import io

if __name__ == '__main__':
    tweak = True

    # !!! Get the dataset !!!
    dir_name = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(dir_name, "dataset/sentiment.csv")
    dataset = pd.read_csv(path)

    sentences = dataset['text'].tolist()
    labels = dataset['sentiment'].tolist()

    # Separate out the sentences and labels into training and test sets
    training_size = int(len(sentences) * 0.8)

    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    # Make labels into numpy arrays for use with the network later
    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)

    # !!! Tokenize the dataset !!!
    if tweak is False:
        vocab_size = 1000
        embedding_dim = 16
        max_length = 100
        trunc_type = 'post'
        padding_type = 'post'
        oov_tok = ""
    else:
        vocab_size = 500
        embedding_dim = 16
        max_length = 50
        trunc_type = 'post'
        padding_type = 'post'
        oov_tok = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    print("Word index({}): {}".format(type(word_index), word_index))
    print("Word index of 'the': {}".format(word_index["the"]))

    sequences = tokenizer.texts_to_sequences(training_sentences)
    print("First Sequence({}): {}".format(type(sequences[0]), sequences[0]))
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type,
                           truncating=trunc_type)
    print("First Padded Sequence({}): {}".format(type(padded[0]), padded[0]))

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length,
                                   padding=padding_type, truncating=trunc_type)

    # !!! Review a Sequence !!!
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    print(decode_review(padded[0]))
    print(training_sentences[0])

    # !!! Train a Basic Sentiment Model with Embeddings !!!
    # Build a basic sentiment network
    # Note the embedding layer is first,
    # and the output is only 1 node as it is either 0 or 1 (negative or positive)
    if tweak is False:
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(6, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            # tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(embedding_dim, return_sequences=True)),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(embedding_dim)),
            tf.keras.layers.Dense(6, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    num_epochs = 10
    model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))

    # !!! Get files for visualizing the network !!!
    # First get the weights of the embedding layer
    # http://projector.tensorflow.org/
    e = model.layers[0]
    weights = e.get_weights()[0]
    print(weights.shape)  # shape: (vocab_size, embedding_dim)

    # Write out the embedding vectors and metadata
    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')
    for word_num in range(1, vocab_size):
        word = reverse_word_index[word_num]
        embeddings = weights[word_num]
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
    out_v.close()
    out_m.close()

    # !!! Predicting Sentiment in New Reviews !!!
    # Use the model to predict a review
    fake_reviews = ['I love this phone', 'I hate spaghetti',
                    'Everything was cold',
                    'Everything was hot exactly as I wanted',
                    'Everything was green',
                    'the host seated us immediately',
                    'they gave us free chocolate cake',
                    'not sure about the wilted flowers on the table',
                    'only works when I stand on tippy toes',
                    'does not work when I stand on my head']

    print(fake_reviews)

    # Create the sequences
    padding_type = 'post'
    sample_sequences = tokenizer.texts_to_sequences(fake_reviews)
    fakes_padded = pad_sequences(sample_sequences, padding=padding_type, maxlen=max_length)

    print('\nHOT OFF THE PRESS! HERE ARE SOME NEWLY MINTED, ABSOLUTELY GENUINE REVIEWS!\n')

    classes = model.predict(fakes_padded)

    # The closer the class is to 1, the more positive the review is deemed to be
    for x in range(len(fake_reviews)):
        print(fake_reviews[x])
        print(classes[x])
