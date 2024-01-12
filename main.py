import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import matplotlib.pyplot as plt

def train_val_test_split(df):
    df['Date'] = pd.to_datetime(df['Date'])
    train_date_end = '2023-08-15'
    lookback_for_val = '2023-08-05'
    val_date_end = '2023-08-30'
    lookback_for_test = '2023-08-01'

    train_df = df[df['Date'] <= train_date_end].reset_index(drop=True)
    val_df = df[(df['Date'] >= lookback_for_val) & (df['Date'] <= val_date_end)].reset_index(drop=True)
    test_df = df[df['Date'] >= lookback_for_test].reset_index(drop=True)

    print("Training Data:")
    print(train_df)
    print("Validation Data:")
    print(val_df)
    print("Testing Data:")
    print(test_df)

    for df in [train_df, val_df, test_df]:
        df.drop(columns=['Date'], inplace=True)
    return train_df, val_df, test_df

def standardize(train, val, test):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)
    return train_scaled, val_scaled, test_scaled, scaler

def windows(data, lookback, lookfront):                    
    X, y = [], []
    for i in range(len(data) - lookback - lookfront):
        X.append(data[i:i+lookback])                            #Multivariate - all 11 features
        y.append(data[i+lookback:i+lookback+lookfront, 0])      #Predict only BTC close price (first feature)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\vijay\Desktop\Personal_projects\BTC-pred-model\merged_preprocessed_data.csv') #Change accordingly
    train_data, val_data, test_data = train_val_test_split(df) 
    train_scaled, val_scaled, test_scaled, scaler = standardize(train_data, val_data, test_data) 

    lookback_steps = 10                                         #Lookback last x days 
    predict_steps = 1                                           #and predict next y days
    X_train, y_train = windows(train_scaled, lookback=lookback_steps, lookfront=predict_steps)
    print('X train shape:', X_train.shape, 'y train shape:', y_train.shape)
    X_val, y_val = windows(val_scaled, lookback=lookback_steps, lookfront=predict_steps)
    print('X val shape:', X_val.shape, 'y val shape:', y_val.shape)
    X_test, y_test = windows(test_scaled, lookback=lookback_steps, lookfront=predict_steps)
    print('X test shape:', X_test.shape, 'y test shape:', y_test.shape)

    # Creating LSTM model
    custom_optimizer = Adam(learning_rate=0.0001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    checkpoint_callback = ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss', 
                                          mode='min', verbose=1)
   
    model = Sequential()
    model.add(LSTM(10, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(5, return_sequences=False))
    model.add(Dense(5))
    model.add(Dense(y_train.shape[1]))
    model.compile(optimizer=custom_optimizer, loss='mse')

    # Train the model
    history = model.fit(X_train, y_train, batch_size=16, epochs=20, shuffle=False, validation_data=(X_val, y_val),
                        validation_batch_size = 1, callbacks=[early_stopping, checkpoint_callback])
    
    # Plot training & validation loss
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    #Test the model
    predictions = model.predict(X_test)
    print(predictions.shape)
    print(y_test.shape)

    pred_price_list = []
    org_price_list = []

    for i in range(predictions.shape[0]):
        prediction = predictions[i].reshape(-1,1)
        y_test_compare = y_test[i].reshape(-1,1)
        print('Shape of prediction -', prediction.shape, 'and shape of target -', y_test_compare.shape)

        repeat_pred = np.repeat(prediction, X_train.shape[2], axis=1)
        repeat_y = np.repeat(y_test_compare, X_train.shape[2], axis=1)
        print('Shape of repeat pred -', repeat_pred.shape, 'and shape of repeat y -', repeat_y.shape)

        pred_price = scaler.inverse_transform(repeat_pred)[:, 0].reshape(-1,1)
        org_price = scaler.inverse_transform(repeat_y)[:, 0].reshape(-1,1)
        print('Shape of preds -', pred_price.shape, 'and shape of orginal price -', org_price.shape)
        
        print('Original price -', org_price)
        org_price_list.append(org_price)
        print('Predicted price -', pred_price)
        pred_price_list.append(pred_price)

    
    #Plot original and predicted test prices to visualize prediction capability
    org_prices = np.concatenate(org_price_list)
    pred_prices = np.concatenate(pred_price_list)
    plt.plot(org_prices, label='Original Prices', marker='o')
    plt.plot(pred_prices, label='Predicted Prices', marker='o')
    plt.xlabel('Days')
    plt.ylabel('BTC price in $')
    plt.title('Original and Predicted BTC Prices')
    plt.legend()
    plt.show()

    




    
