import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf
import os

from keras import backend as K

import gc
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import datetime

from sklearn.metrics import mean_squared_error


start = int(sys.argv[1])
time = int(sys.argv[2])
#start = 9
finish = start + 1


test_year = 2016
validation_years = [2015,2018]
train_years = [2011,2012,2013,2014,2017,2019]


folder_result = 'resultados_' + str(test_year)
if not os.path.isdir(folder_result):
    os.makedirs(folder_result)

def separate_datasets():
    df_dengue = pd.read_csv('dengue_'+str(test_year)+'.csv', delimiter=';')
    df_dengue['DATA'] =  pd.to_datetime(df_dengue['DATA'],format='%Y-%m-%d',errors='coerce')
    
    train_data = df_dengue[(df_dengue['DATA']>=str(train_years[0])+'-01-01') & (df_dengue['DATA']<=str(train_years[0])+'-12-31')]
    
    for index in range(1,len(train_years)):
        data = df_dengue[(df_dengue['DATA']>=str(train_years[index])+'-01-01') & (df_dengue['DATA']<=str(train_years[index])+'-12-31')]
        train_data = train_data.append(data)           
    train_data = train_data.sort_values('DATA', ascending=True)
    
    neighborhoods = df_dengue.keys().to_list()[:-1]
    
    #dengue data
    validation_data_1 = df_dengue[(df_dengue['DATA']>=str(validation_years[0])+'-01-01') & (df_dengue['DATA']<=str(validation_years[0])+'-12-31')]
    validation_data_2 = df_dengue[(df_dengue['DATA']>=str(validation_years[1])+'-01-01') & (df_dengue['DATA']<=str(validation_years[1])+'-12-31')]

    validation_data_1 = train_data.append(validation_data_1)
    validation_data_2 = train_data.append(validation_data_2)
    
    test_data = df_dengue[(df_dengue['DATA']>=str(test_year)+'-01-01') & (df_dengue['DATA']<=str(test_year)+'-12-31')]
    train_data = train_data.append(test_data)
    
    df_dengue = df_dengue.filter(neighborhoods)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_dengue[neighborhoods] = scaler.fit_transform(df_dengue[neighborhoods])

    train_data = train_data.filter(neighborhoods)
    validation_data_1 = validation_data_1.filter(neighborhoods)
    validation_data_2 = validation_data_2.filter(neighborhoods)

    train_data[neighborhoods] = scaler.transform(train_data[neighborhoods])
    validation_data_1[neighborhoods] = scaler.transform(validation_data_1[neighborhoods])
    validation_data_2[neighborhoods] = scaler.transform(validation_data_2[neighborhoods])

    return train_data, validation_data_1, validation_data_2, scaler, neighborhoods

def predict (X_test, model, forecasting_time, scaler, ts_to_compare):
    
    model.reset_states() 

    #Preparando dados e estados do LSTM
    begin = np.expand_dims(X_test[1][-1], axis=0)
    for i in range(2,forecasting_time+1):
        begin = np.concatenate((begin, np.expand_dims(X_test[i][-1], axis=0)), axis=0)

    for i in range(0, forecasting_time):
        model.predict([np.array([X_test[i]])])

    
    #Predição
    preds = []
    base = X_test[forecasting_time]
    for i in range(52 - forecasting_time):
        y_pred = model.predict([np.array([base])])
        preds.append(y_pred)

        base = np.delete(base, (0), axis=0)
        base = np.concatenate((base, y_pred), axis = 0)

    
    preds = np.array(preds)
    preds = np.squeeze(preds,axis=1)
    preds = np.concatenate((begin, preds), axis = 0)
    
    #preds = scaler.inverse_transform(preds)

    '''plt.plot(np.sum(preds,axis=1))
    plt.plot(np.sum(ts_to_compare,axis=1))
    plt.show()
    a'''

    preds_loss = scaler.inverse_transform(preds)
    #real = scaler.inverse_transform(ts_to_compare)
    real = ts_to_compare
   
    #plt.plot(np.sum(preds,axis=1))
    #plt.plot(np.sum(real,axis=1))
    #plt.show()
    #a

    preds_loss = np.transpose(preds_loss)
    preds = np.transpose(preds)
    real = np.transpose(real)

    gc.collect()
    keras.backend.clear_session()
    K.clear_session()
    
    return preds, mean_squared_error(real, preds_loss)

for forecasting_time in range(start,finish):
    if os.path.isdir(folder_result+'/forecasting_time_' + str(forecasting_time)):
        continue
    print('forecasting_time: ' + str(forecasting_time) + ' year: ' + str(test_year))        


    df_dengue, validation_1, validation_2, scaler, neighborhoods = separate_datasets()
        
    time_steps = 52
    test_size = 52
    
    train_size = int(len(df_dengue)-(test_size))

    #dengue data
    train, test = df_dengue.iloc[0:train_size], df_dengue.iloc[(train_size-time_steps):len(df_dengue)]
    _, validation_1 = validation_1.iloc[0:train_size], validation_1.iloc[(train_size-time_steps):len(validation_1)]
    _, validation_2 = validation_2.iloc[0:train_size], validation_2.iloc[(train_size-time_steps):len(validation_2)]


    df_dengue_2 = pd.read_csv('dengue_'+str(test_year)+'.csv', delimiter=';')
    df_dengue_2['DATA'] =  pd.to_datetime(df_dengue_2['DATA'],format='%Y-%m-%d',errors='coerce')
    test_data = df_dengue_2[(df_dengue_2['DATA']>=str(test_year)+'-01-01') & (df_dengue_2['DATA']<=str(test_year)+'-12-31')]
    validation_1_data = df_dengue_2[(df_dengue_2['DATA']>=str(validation_years[0])+'-01-01') & (df_dengue_2['DATA']<=str(validation_years[0])+'-12-31')]
    validation_2_data = df_dengue_2[(df_dengue_2['DATA']>=str(validation_years[1])+'-01-01') & (df_dengue_2['DATA']<=str(validation_years[1])+'-12-31')]

    ts_to_compare_test_real = test_data.filter(neighborhoods).to_numpy()
    ts_to_compare_validation_1_real = validation_1_data.filter(neighborhoods).to_numpy()
    ts_to_compare_validation_2_real = validation_2_data.filter(neighborhoods).to_numpy()

    def create_dataset(X, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = np.expand_dims(X.iloc[i:(i + time_steps),0].to_numpy(),axis = 0)
            for j in range(1,len(neighborhoods)):
                v = np.concatenate([v,np.expand_dims(X.iloc[i:(i + time_steps),j].to_numpy(),axis = 0)], axis = 0)
            Xs.append(np.transpose(v))
        
        
        for i in range(len(X) - time_steps):
            v = np.expand_dims(X.iloc[i + time_steps,0],axis = 0)
            for j in range(1,len(neighborhoods)):
                v = np.concatenate([v,np.expand_dims(X.iloc[i + time_steps,j],axis = 0)], axis = 0)
            ys.append(np.transpose(v))
        return np.array(Xs), np.array(ys)
    
    def create_dataset_week(X, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = np.expand_dims(X.iloc[i:(i + time_steps),0].to_numpy(),axis = 0)
            Xs.append(np.transpose(v))
        
        
        for i in range(len(X) - time_steps):
            v = np.expand_dims(X.iloc[i + time_steps,0],axis = 0)
            ys.append(np.transpose(v))
        return np.array(Xs), np.array(ys)
    

    #dengue data
    X_train, y_train = create_dataset(train, time_steps)
    X_test, y_test = create_dataset(test, time_steps)

    X_validation_1, y_validation_1 = create_dataset(validation_1, time_steps)
    X_validation_2, y_validation_2 = create_dataset(validation_2, time_steps)


    print('Trainning data')
    print(X_train.shape)
    print(y_train.shape)

    print('Validation data 1')
    print(X_validation_1.shape)
    print(y_validation_1.shape)

    print('Validation data 2')
    print(X_validation_2.shape)
    print(y_validation_2.shape)

    print('Test data')
    print(X_test.shape)
    print(y_test.shape)
        
    def getModel(neurons, layer_dropout, lstm_stack_num, optimizer):
        input_dengue = layers.Input(batch_shape = (None, X_train.shape[1], X_train.shape[2]))   

        lstm_stack = None
        if (lstm_stack_num == 1):
            lstm_stack = layers.LSTM(neurons)(input_dengue)
            lstm_stack = layers.Dropout(layer_dropout)(lstm_stack)
        else:
            lstm_stack = layers.LSTM(neurons, return_sequences=True)(input_dengue)
            lstm_stack = layers.Dropout(layer_dropout)(lstm_stack)

        for i in range(1, lstm_stack_num - 1):
            lstm_stack = layers.LSTM(neurons, return_sequences=True)(lstm_stack)
            lstm_stack = layers.Dropout(layer_dropout)(lstm_stack)

        if (lstm_stack_num > 1):
            lstm_stack = layers.LSTM(neurons)(lstm_stack)
            lstm_stack = layers.Dropout(layer_dropout)(lstm_stack)

        output_dense = layers.Dense(y_test.shape[1], activation='sigmoid', use_bias=True)(lstm_stack)
        model = Model(inputs=[input_dengue], outputs=output_dense)
        model.compile(loss='mean_squared_error',optimizer=optimizer)
        
        return model

    # start params to test
    '''param_grid = dict(
        batch_size = [64],
        epochs = [1],
        neurons = [1], 
        layer_dropout=[0.9], 
        lstm_stack_num = [1],
        optimizer=['rmsprop'])'''
    # end params to test

    # start params to run
    param_grid = dict(
        batch_size = [64], 
        epochs = [200, 250, 300, 350],
        neurons = [50, 150, 200, 300], 
        layer_dropout=[0.0,0.1, 0.2], 
        lstm_stack_num = [1,2,3],
        optimizer=['rmsprop'])
    # end params to run
    
    best_params = {}
    loss = 1000000000
    total = len(param_grid['batch_size']) + len(param_grid['epochs']) + len(param_grid['neurons']) + len(param_grid['layer_dropout']) + len(param_grid['lstm_stack_num'])  + len(param_grid['optimizer'])
    total = total * total
    count = 0
    for epoch in param_grid['epochs']:
        count = count + 1
        for neuron in param_grid['neurons']:
            count = count + 1
            for dropout in param_grid['layer_dropout']:
                count = count + 1
                for stack_num in param_grid['lstm_stack_num']:
                    count = count + 1
                    for opt in param_grid['optimizer']:
                        count = count + 1
                        print(str(count) + ' de ' + str(total))
                        model = getModel(neurons=neuron, layer_dropout=dropout, lstm_stack_num = stack_num, optimizer=opt)
                        
                        '''history = model.fit(
                            [X_train], y_train,
                            epochs=epoch,
                            batch_size=64,
                            shuffle=False, verbose=0
                        )'''
                        history = model.fit(
                            [X_train], y_train,
                            epochs=epoch,
                            batch_size=64,
                            shuffle=False, verbose=0
                        )

                        preds_v_1, loss_v_1 = predict(X_validation_1, model, forecasting_time, scaler, ts_to_compare_validation_1_real)
                        preds_v_2, loss_v_2 = predict(X_validation_2, model, forecasting_time, scaler, ts_to_compare_validation_2_real)
                                            
                        if ((loss_v_1 + loss_v_2) / 2) < loss:
                            loss = ((loss_v_1 + loss_v_2) / 2)
                            best_params['neurons'] = neuron
                            best_params['epochs'] = epoch
                            best_params['layer_dropout'] = dropout
                            best_params['lstm_stack_num'] = stack_num
                            best_params['optimizer'] = opt
                        print('loss: ' + str(((loss_v_1 + loss_v_2) / 2)) + ' / chosen_loss: ' + str(loss))
                        model = None

    print('choosen loss: ' + str(loss))
    model = getModel(neurons=best_params['neurons'], layer_dropout=best_params['layer_dropout'], lstm_stack_num = best_params['lstm_stack_num'], optimizer=best_params['optimizer'])
    
    '''history = model.fit(
        [X_train, X_arrival_train], y_train,
        epochs=best_params['epochs'],
        batch_size=64,
        validation_split=0.1,
        shuffle=False
    )'''
    history = model.fit(
        [X_train], y_train,
        epochs=best_params['epochs'],
        batch_size=64,
        validation_split=0.1,
        shuffle=False
    )
    
    preds, loss = predict(X_test, model, forecasting_time, scaler, ts_to_compare_test_real)
    
    dados_pred = {}
    for index in range(len(neighborhoods)):
        dados_pred[neighborhoods[index]] = preds[index].tolist()

    
    df_dengue = df_dengue.append(pd.DataFrame(data=dados_pred))
    df_dengue[neighborhoods] = scaler.inverse_transform(df_dengue[neighborhoods])

    df_dados_real_predito = df_dengue.tail(2*52)

    dados_real = df_dados_real_predito[0:52][neighborhoods].to_numpy()
    dados_predito = df_dados_real_predito [52:2*52][neighborhoods].to_numpy()

    dados_predito = np.transpose(dados_predito)
    dados_real = np.transpose(dados_real)

    path = folder_result+'/'+str(time)+'_forecasting_time_' + str(forecasting_time) + '/'
    if not os.path.isdir(path):
        os.makedirs(path)

    dataframe_dict = {}
    i = 0
    

    sum_pred = []
    sum_real = []
    for neighborhood in neighborhoods:
        predicted = dados_predito[i]
        real = dados_real[i]

        dataframe_dict[neighborhood+'_predicted'] = predicted
        dataframe_dict[neighborhood+'_real'] = real

        #chart per neighborhood
        plt.figure(figsize=(7,5))
        plt.plot(range(len(real)),real, 'black',linewidth=2.0, marker='o')
        plt.plot(range(len(predicted)),predicted, 'orange',linewidth=2.0, marker='o')
        plt.legend(['real ' + str(test_year),'predicted'])
        plt.title(neighborhood)
        plt.savefig(folder_result+'/'+str(time)+'_forecasting_time_' + str(forecasting_time) + '/'+neighborhood+'.png')
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()

        #accumulate
        if len(sum_pred) == 0:
            sum_pred = predicted
            sum_real = real
        else:
            sum_pred = sum_pred + predicted
            sum_real = sum_real + real

        i = i + 1

    #chart fortaleza
    plt.figure(figsize=(10,5))
    plt.plot(range(len(sum_real)),sum_real, 'black',linewidth=2.0, marker='o')
    plt.plot(range(len(sum_pred)),sum_pred, 'orange',linewidth=2.0, marker='o')
    
    plt.title('FORTALEZA')
    plt.savefig(folder_result+'/'+str(time)+'_forecasting_time_' + str(forecasting_time) + '/fortaleza.png')
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    df = pd.DataFrame(dataframe_dict)
    df.to_csv(folder_result+'/'+str(time)+'_forecasting_time_' + str(forecasting_time) + '/result.csv', sep=',')
