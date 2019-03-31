import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector,Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import arrow

train = pd.read_csv('高峰值2019.csv')
print(train)

def buildTrain(train):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-37):
        X_train.append(np.array(train.iloc[i:i+30]))
        Y_train.append(np.array(train.iloc[i+30:i+37]["尖峰負載(MW)"]))
    return np.array(X_train), np.array(Y_train)

def shuffle(X,Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]

def splitData(X,Y,rate):
    X_train = X[int(X.shape[0]*rate):]
    Y_train = Y[int(Y.shape[0]*rate):]
    X_val = X[:int(X.shape[0]*rate)]
    Y_val = Y[:int(Y.shape[0]*rate)]
    return X_train, Y_train, X_val, Y_val

scaler = MinMaxScaler(feature_range=(0, 1))
train_norm = scaler.fit_transform(train.drop('date',axis=1))
train_norm = pd.DataFrame(train_norm)
train_norm['尖峰負載(MW)'] = train_norm[0]
train_norm = train_norm.drop(0,axis=1)

# build Data, use last 30 days to predict next 7 days
X_train, Y_train = buildTrain(train_norm)

# shuffle the data, and random seed is 10
X_train, Y_train = shuffle(X_train, Y_train)


def lstm_stock_model(shape):
    model = Sequential()
    model.add(LSTM(256, input_shape=(shape[1], shape[2]), return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    model.add(Dense(30,activation='linear'))
    model.add(Dense(7,activation='linear'))
    model.compile(loss="mean_absolute_error", optimizer="adam",metrics=['mean_absolute_error'])
    model.summary()
    return model


#model = lstm_stock_model(X_train.shape)
#callback = EarlyStopping(monitor="mean_absolute_error", patience=10, verbose=1, mode="auto")

#history = model.fit(X_train, Y_train, epochs=1000, batch_size=5, validation_split=0.1, callbacks=[callback],shuffle=True)

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.show()


#Predict1 = model.predict(np.array(train_norm[54:84]).reshape((1,30,1)))
#print(Predict1)

#model.save('my_model.h5')

# load
model = load_model('my_model.h5')
print('test after load: ', model.predict(np.array(train_norm[54:84]).reshape((1,30,1))))
Predict1 = model.predict(np.array(train_norm[54:84]).reshape((1,30,1)))
trainPredict = scaler.inverse_transform(Predict1)
print(trainPredict)

def get_date_range(start, limit, level='day',format='YYYY-MM-DD'):
    start = arrow.get(start, format)  
    result=(list(map(lambda dt: dt.format(format) , arrow.Arrow.range(level, start, 		   limit=limit))))
    dateparse2 = lambda dates:pd.datetime.strptime(dates,'%Y-%m-%d')
    return map(dateparse2, result)

te = pd.DataFrame(trainPredict).T
te.index = get_date_range('2019-03-26',7)
print(te)
train.index = pd.to_datetime(train['date'])
plt.plot(train['尖峰負載(MW)'],color='blue')
plt.plot(te,color='red')
plt.show()

te['尖峰負載(MW)'] = te[0]
te = te.drop(0,axis=1)
train = train.append(te['2019-03-31':'2019-04-01'])
data_new = train.drop('date',axis=1)
#print(data_new)


####################################################################################

train_norm1 = scaler.fit_transform(data_new)
train_norm1 = pd.DataFrame(train_norm1)
train_norm1['尖峰負載(MW)'] = train_norm1[0]
train_norm1 = train_norm1.drop(0,axis=1)

# build Data, use last 30 days to predict next 7 days
X_train1, Y_train1 = buildTrain(train_norm1)

# shuffle the data, and random seed is 10
X_train1, Y_train1 = shuffle(X_train1, Y_train1)

Predict2 = model.predict(np.array(train_norm1[61:91]).reshape((1,30,1)))
trainPredict2 = scaler.inverse_transform(Predict2)
#print(trainPredict2)

answer = pd.DataFrame(trainPredict2).T
answer['date'] = range(20190402,20190409)
answer['peak_load(MW)'] = answer[0]
answer = answer.drop(0,axis=1)

answer.to_csv('submission.csv',index=False)