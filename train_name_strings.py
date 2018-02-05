'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import Input, LSTM, RepeatVector
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from utils import read_txt_file, add_space
from copy import deepcopy
import numpy as np
import random
import sys

RANDOM_SEED = 1918
np.random.seed(RANDOM_SEED)

# load name file
all_names = read_txt_file("all_together/all_new.txt")
data_size = len(all_names)

chars = ' '.join(all_names)
chars = sorted(list(set(chars)))
chars_len = len(chars)
print('total chars:', chars_len)

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# make all name strings up to a max lenght
maxlen = 22
for i, names in enumerate(all_names):
    all_names[i] = add_space(names, maxlen)


# randomly pair targets to inputs
targets = deepcopy(all_names)
random.shuffle(targets)

print('Vectorization...',)
X = np.zeros((len(all_names), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(all_names), maxlen, len(chars)), dtype=np.bool)
# y = np.zeros((len(all_names), len(chars)), dtype=np.bool)
for i, name_string in enumerate(all_names):
    for t, char in enumerate(name_string):
        X[i, t, char_indices[char]] = 1

for i, name_string in enumerate(targets):
    for t, char in enumerate(name_string):
        y[i, t, char_indices[char]] = 1

        # y[i, t, char_indices[char]] = 1
    # y[i, char_indices[targets[i]]] = 1
print('Success!')

# %% Models section
# input placeholder
input_strings = Input(shape=(maxlen, chars_len))

# encode the representation of the input
model_input = Input(shape=(maxlen, chars_len))
encoded = LSTM(64)(model_input)

# reconstruction of the input
decoded = RepeatVector(maxlen)(encoded)
decoded = LSTM(chars_len, return_sequences=True)(model_input)

# the model
seq_autoencoder = Model(model_input, decoded)
# encoder = Model(model_input, encoded)
seq_autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# show model summary
seq_autoencoder.summary()
seq_autoencoder.fit(X, y,
                    epochs=2,
                    batch_size=128,
                    shuffle=True)

# %% tests
new_names = ['ayamatanga', 'omoluorogbo', 'aobachiba', 'alayonimi']
X_test = np.zeros((len(new_names), maxlen, len(chars)), dtype=np.bool)
for i, name_string in enumerate(new_names):
    for t, char in enumerate(name_string):
        X_test[i, t, char_indices[char]] = 1

preds = seq_autoencoder.predict(X_test)

for results in preds:
    print(results)


# # build the model: a single LSTM
# print('Build model...')
# model = Sequential()
# model.add(LSTM(128, input_shape=(maxlen, len(chars))))
# model.add(Dense(len(chars)))
# model.add(Activation('softmax'))

# optimizer = RMSprop(lr=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# def sample(preds, temperature=1.0):
#     # helper function to sample an index from a probability array
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)

# # train the model, output generated text after each iteration
# for iteration in range(1, 60):
#     print()
#     print('-' * 50)
#     print('Iteration', iteration)
#     model.fit(X, y,
#               batch_size=128,
#               epochs=1)

#     x = np.zeros((1, maxlen, len(chars)))
#     pred = model.predict(x, verbose=0)

#     print('prediction:', pred)

#     # start_index = random.randint(0, len(text) - maxlen - 1)

#     # for diversity in [0.2, 0.5, 1.0, 1.2]:
#     #     print()
#     #     print('----- diversity:', diversity)

#     #     generated = ''
#     #     name_string = text[start_index: start_index + maxlen]
#     #     generated += name_string
#     #     print('----- Generating with seed: "' + name_string + '"')
#     #     sys.stdout.write(generated)

#     #     for i in range(400):
#     #         x = np.zeros((1, maxlen, len(chars)))
#     #         for t, char in enumerate(name_string):
#     #             x[0, t, char_indices[char]] = 1.

#     #         preds = model.predict(x, verbose=0)[0]
#     #         next_index = sample(preds, diversity)
#     #         next_char = indices_char[next_index]

#     #         generated += next_char
#     #         name_string = name_string[1:] + next_char

#     #         sys.stdout.write(next_char)
#     #         sys.stdout.flush()
#     #     print()

