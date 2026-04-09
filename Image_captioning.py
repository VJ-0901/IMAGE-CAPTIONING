import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

base_model = ResNet50(weights='imagenet')
cnn_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)



def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    feature = cnn_model.predict(img, verbose=0)
    return feature[0]



captions = {
    r"C:\Users\HP\Desktop\ImageCaptioning\dog.jpg": "start a dog running in grass end",
    r"C:\Users\HP\Desktop\ImageCaptioning\cat.jpg": "start a cat sitting on the floor end",
    r"C:\Users\HP\Desktop\ImageCaptioning\bike.jpg": "start a man riding a bike end"
}


all_captions = list(captions.values())

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

vocab_size = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(all_captions)
max_length = max(len(seq) for seq in sequences)


X1, X2, y = [], [], []

for img_name, caption in captions.items():
    seq = tokenizer.texts_to_sequences([caption])[0]

    for i in range(1, len(seq)):
        in_seq, out_seq = seq[:i], seq[i]
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]

        img_feature = extract_features(img_name)

        X1.append(img_feature)
        X2.append(in_seq)
        y.append(out_seq)

X1 = np.array(X1)
X2 = np.array(X2)
y = np.array(y)

image_input = Input(shape=(2048,))
img_dense = Dense(256, activation='relu')(image_input)
img_dropout = Dropout(0.5)(img_dense)

text_input = Input(shape=(max_length,))
text_embed = Embedding(vocab_size, 256, mask_zero=True)(text_input)
text_lstm = LSTM(256)(text_embed)

decoder = tf.keras.layers.add([img_dropout, text_lstm])
decoder_dense = Dense(256, activation='relu')(decoder)
outputs = Dense(vocab_size, activation='softmax')(decoder_dense)

model = Model(inputs=[image_input, text_input], outputs=outputs)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

print(model.summary())

print("\nTraining model...")
model.fit([X1, X2], y, epochs=200, verbose=1)


def generate_caption(photo):
    in_text = "start"

    for i in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)

        yhat = model.predict([photo, seq], verbose=0)
        yhat = np.argmax(yhat)

        word = None
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break

        if word is None:
            break

        in_text += " " + word

        if word == "end":
            break

    return in_text.replace("start", "").replace("end", "").strip()


print("\n=== Testing ===")

img_path = r"C:\Users\HP\Desktop\ImageCaptioning\dog.jpg"

photo = extract_features(img_path).reshape(1, 2048)

caption = generate_caption(photo)

print("Generated Caption:", caption)
