import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
import os

# Load data
temp_inputs_path = './TempInputsDF.csv'
temp_outputs_path = './TempOutDF.csv'

# Read the datasets
temp_in_df = pd.read_csv(temp_inputs_path)
temp_out_df = pd.read_csv(temp_outputs_path)

# Preprocess timestamps


def encode_timestamps(df, time_cols):
    """
    Adds sin and cos features for year, month, and day cycles for multiple datetime columns.
    """
    for time_col in time_cols:
        df[time_col] = pd.to_datetime(df[time_col])  # Ensure datetime format
        df[f'sin_{time_col}_month'] = np.sin(
            2 * np.pi * df[time_col].dt.month / 12)
        df[f'cos_{time_col}_month'] = np.cos(
            2 * np.pi * df[time_col].dt.month / 12)
        df[f'sin_{time_col}_day'] = np.sin(
            2 * np.pi * df[time_col].dt.day / 31)
        df[f'cos_{time_col}_day'] = np.cos(
            2 * np.pi * df[time_col].dt.day / 31)
        df.drop(columns=[time_col], inplace=True)
    return df


temp_in_df = encode_timestamps(temp_in_df, ['TimeStep'])
temp_out_df = encode_timestamps(temp_out_df, ['TimeStart', 'TimeEnd'])

# Filter tanks based on status


def filter_in_service(df, monthly_data, tanks):
    for tank in tanks:
        status_col = f'{tank}Status'
        if status_col in monthly_data.columns:
            if monthly_data[status_col].iloc[0] == 0:  # Tank is out of service
                df.drop(columns=[f'{tank}OGValue'], inplace=True)
    return df

# Normalize OGValues


def normalize_column(df_list, column, min_val, max_val):
    for df in df_list:
        df[column] = (df[column] - min_val) / (max_val - min_val)
    return df_list

# Split data into sequences by month


def segment_by_month(input_df, output_df):
    segmented_inputs = []
    for _, row in output_df.iterrows():
        start_date, end_date = row['sin_TimeStart_month'], row['sin_TimeEnd_month']
        monthly_data = input_df[
            (input_df['sin_TimeStep_month'] >= start_date) &
            (input_df['cos_TimeStep_month'] <= end_date)
        ]
        segmented_inputs.append(monthly_data)
    return segmented_inputs


x = segment_by_month(temp_in_df, temp_out_df)
y = temp_out_df

# Define model


class TimeSeriesPredictor(Model):
    def __init__(self, input_dim, context_dim, embedding_dim, hidden_dim, num_tanks, tank_embedding_dim):
        super(TimeSeriesPredictor, self).__init__()
        self.encoder = layers.LSTM(hidden_dim, return_sequences=False)
        self.context_network = layers.Dense(embedding_dim, activation='relu')
        self.tank_embedding = layers.Embedding(num_tanks, tank_embedding_dim)
        self.decoder = layers.Dense(input_dim, activation='linear')

    def call(self, daily_inputs, monthly_context, tank_indices):
        daily_embedding = self.encoder(daily_inputs)
        context_embedding = self.context_network(monthly_context)
        tank_embeddings = self.tank_embedding(tank_indices)
        fused_embedding = tf.concat(
            [daily_embedding, context_embedding, tank_embeddings], axis=-1)
        return self.decoder(fused_embedding)


# Parameters
input_dim = len(x[0].columns) - 1  # Exclude TimeStep
context_dim = y.shape[1] - 2  # Exclude TimeStart, TimeEnd
embedding_dim = 128
hidden_dim = 128
num_tanks = 3
tank_embedding_dim = 16

model = TimeSeriesPredictor(
    input_dim, context_dim, embedding_dim, hidden_dim, num_tanks, tank_embedding_dim)

# Compile model
model.compile(optimizer='adam', loss='mse')

# Prepare data for training
X_train_daily_data = x[:-1]
X_train_monthly_data = y.iloc[:-1]
test = x[-1]

# Train model
epochs = 10
for epoch in range(epochs):
    for i, daily_data in enumerate(X_train_daily_data):
        daily_data_tensor = tf.convert_to_tensor(
            daily_data.values, dtype=tf.float32)
        monthly_context = tf.convert_to_tensor(
            X_train_monthly_data.iloc[i].values, dtype=tf.float32)
        # Assuming tanks are indexed as 0, 1, 2
        tank_indices = tf.constant([0, 1, 2])
        with tf.GradientTape() as tape:
            predictions = model(daily_data_tensor,
                                monthly_context, tank_indices)
            loss = tf.reduce_mean(
                tf.square(predictions - daily_data_tensor))  # Placeholder loss
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(f"Epoch {epoch+1}/{epochs} completed.")
