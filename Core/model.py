from tensorflow.keras.models import Sequential as sq
from tensorflow.keras.layers import LSTM, Dense
from keras.models import load_model as lm
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE, r2_score
from sklearn.preprocessing import MinMaxScaler as MM
import numpy as np
import matplotlib.pyplot as plt

class model_train_seq_LSTM :

    def __init__(self, dataframe, scaler) :
        self.dataframe = dataframe
        self.scaler = scaler

    def prep_data_lstm(self, feature_col = 'Close', look_back = 60) :   #"look back" refers to the number of previous time steps that the model uses to make predictions for the next time step.
        """Prepare data for LSTM by creating sequences"""
        data = self.dataframe[feature_col].values     # Extract the feature column values
        data = data.reshape(-1, 1)  # Reshape to (n_samples, 1)
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:i + look_back])     # Append sequences of length 'look_back'
            y.append(data[i + look_back])       # Append the next value as the target
        X, y = np.array(X), np.array(y)
        # Split data while preserving the number of features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)
        return X_train, X_test, y_train, y_test.reshape(-1, 1)

    def build_train_lstm(self, feature_col = 'Close', look_back = 60, epochs = 100, batch_size = 32) :
        """Build and train the LSTM model"""
        X_train, X_test, y_train, y_test = self.prep_data_lstm(feature_col, look_back)

        self.X_train = X_train  # Save the training features
        self.y_train = y_train  # Save the training targets

        model = sq()    # Initialize the model
        model.add(LSTM(units = 50, return_sequences = True, input_shape = (look_back, 1)))
        model.add(LSTM(units = 50))   #this layer does not have return_sequences=True, which means it will only return the final output of the sequence, not the full sequence.
        model.add(Dense(1))     #This is the output layer of the model. Since we are predicting a single value (the next day's stock price), this layer has one neuron.

        #LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)) adds an LSTM layer to the model.
        #50 is the number of units in this LSTM layer, which means the layer will have 50 LSTM cells.
        #return_sequences=True means this layer will return the full sequence of outputs for each input sequence, not just the final output. This is necessary because we are stacking another LSTM layer on top.
        #input_shape=(X_train.shape[1], 1) specifies the shape of the input data for this layer.
        #X_train.shape[1] is the number of time steps (look-back period), and 1 is the number of features (since we are using just one feature, the stock price).

        model.compile(optimizer = 'adam', loss = 'mean_squared_error')     # Compile the model using Adam optimizer and mean squared error loss

        self.model = model    # Save the trained model
        self.X_test = X_test  # Save the test features
        self.y_test = y_test  # Save the test targets

        # Train the model with the training data
        self.history = self.model.fit(self.X_train, self.y_train, epochs = epochs, batch_size = batch_size, validation_data = (self.X_test, self.y_test))

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc = 'upper right')
        plt.show()
        


    def predict_plot(self):
        """Make predictions and plot the results"""
        predictions = self.model.predict(self.X_test)   # Predict the test data
        predictions = predictions.reshape(-1, 1)    # Reshape the predictions

        # Extract the 'Close' column from the original dataframe
        self.close_scaler = MM(feature_range = (0, 1))  # Create a new scaler for just the 'Close' column
        self.close_scaler.fit(self.dataframe['Close'].values.reshape(-1, 1))  # Fit the scaler to the 'Close' column

        # Inverse transform predictions using the scaler fitted on 'Close' column
        predictions = self.close_scaler.inverse_transform(predictions)
        # Inverse transform actual 'Close' values
        actual = self.close_scaler.inverse_transform(self.y_test)

        plt.figure(figsize = (14, 7))
        plt.plot(actual, label = 'Actual Stock Price')
        plt.plot(predictions, label = 'Predicted Stock Price')
        plt.title('Actual vs Predicted Stock Prices')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

    def evaluate_model(self):
        """Calculates and prints evaluation metrics."""
        predictions = self.model.predict(self.X_test)   # Predict the test data
        predictions = self.close_scaler.inverse_transform(predictions)    # Inverse transform predictions
        actual = self.close_scaler.inverse_transform(self.y_test)      # Inverse transform actual values

        mse = MSE(actual, predictions)
        rmse = np.sqrt(mse)
        mae = MAE(actual, predictions)
        r2 = r2_score(actual, predictions)

        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R-squared (R2): {r2}")

    def save_model(self, model_path = 'C:/Users/saumy/Stock market price prediction/stock_model.h5') :
        """Saves the trained model to a file."""
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path = 'C:/Users/saumy/Stock market price prediction/stock_model.h5') :
        """Loads a saved model from a file."""
        self.model = lm(model_path)
        print(f"Model loaded from {model_path}")
