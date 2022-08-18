import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import requests, json # API Loteria Caixa Brasil
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout



lista = []
# Instancia API da loteria.
nomedaloteria = 'quina'
token = 'pzmnI5imsG8Rkg5'
concurso = 5926 # Ultimo concurso

# Catch 10 jogos
while concurso >= 5516:
    response = requests.get(f'https://apiloterias.com.br/app/resultado?loteria={nomedaloteria}&token={token}&concurso={concurso}')

    # Adicionando dados a lista 
    json_data = json.loads(response.content)
    dezenas = json_data['dezenas']
    # for dezena in dezenas:
    lista.append(dezenas)
    arr = np.array(lista)
    concurso -= 1
print(np.array(arr))


df = pd.DataFrame(np.array(arr))
df.head()

print(df)  

scaler = StandardScaler().fit(df.values)
transformed_dataset = scaler.transform(df.values)
transformed_df = pd.DataFrame(data=transformed_dataset, index=df.index)

number_of_rows = df.values.shape[0]
window_length = 7
number_of_features = df.values.shape[1]

train = np.empty([number_of_rows-window_length, window_length, number_of_features], dtype=float)
label = np.empty([number_of_rows-window_length, number_of_features], dtype=float)
window_length = 7

for i in range(0, number_of_rows-window_length):
    train[i] = transformed_df.iloc[i:i+window_length, 0: number_of_features]
    label[i] = transformed_df.iloc[i+window_length: i+window_length+1, 0: number_of_features]
    
train.shape
label.shape

# train[0]

# Inicio do Keras

batch_size = 100
modelo = Sequential()
modelo.add(Bidirectional(LSTM(240,
            input_shape=(window_length, number_of_features),
            return_sequences=True)))
modelo.add(Dropout(0.2))
modelo.add(Bidirectional(LSTM(240,
            input_shape=(window_length, number_of_features),
            return_sequences=True)))
modelo.add(Dropout(0.2))
modelo.add(Bidirectional(LSTM(240,
            input_shape=(window_length, number_of_features),
            return_sequences=True)))
modelo.add(Bidirectional(LSTM(240,
            input_shape=(window_length, number_of_features),
            return_sequences=False)))
modelo.add(Dense(59))
modelo.add(Dense(number_of_features))
modelo.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

modelo.fit(train, label,
           batch_size=100, epochs=1000)

to_predict = np.array(arr)
scaled_to_predict = scaler.transform(to_predict)

scaled_predicted_output_1 = modelo.predict(np.array([scaled_to_predict]))
print('jogo')
print(scaler.inverse_transform(scaled_predicted_output_1).astype(int)[0])

#for s in range(len(json_data)):
#	if json_data[s]["dezenas"] == to_find: 
#           print("O valor de {} é {}.".format(json_data[s]["CodigoAmigavel"], json_data[s]["PrecoVenda"]))
