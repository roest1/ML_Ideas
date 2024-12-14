# Data Summary

2020-2024 Input sensor data (daily frequency) from Aveva Pi for Heavy Olefins Tanks with calculated monthly compliance metrics and annual reports.

TankIDs:
* TBD910
* TBD911
* TBD912

# Temperature Input Data

Input daily sensor data

* Each row is a day in time 
  
### columns:

* TimeStep 
* {TankID}OGValue
* {TankID}OGTentativeQuality
* {TankID}OGTrueQuality
* {TankID}ProcessedValue
* {TankID}ProcessedTentativeQuality
* {TankID}ProcessedTrueQuality

Dimensions (Rows, Cols) = (1613, 20)

To parse input into lstm encoder, 
the months are split by ensuring each sequence starts when .TimeStep.day = 1

# Temperature Output Data

Calculated monthly average temperature

* Each row corresponds to one month

### columns:

* TimeStart
* TimeEnd
* {TankID}Status
* {TankID}MonthlyAvgTemp
* {TankID}AnnualAvgTemp
* {TankID}PeriodsOfGoodQualityDataPct
* {TankID}PeriodsOfGoodQualityDataDays

Dimensions (Rows, Cols) = (53, 17)

---

## Training methodology

We can establish a way for the model to read both (or create separate models for reading) the sequence of inputs designated to the respective outputs. Reading both the inputs and outputs of a month can hopefully give the model some help assigning how to weight of the input sensor values. For example, giving inaccurate results if the original data quality isn't good. 

By training on data from 1/1/2020 to 4/30/2024  

and testing on data from 5/1/2024 to 5/31/2024 

we just test the latest month, but this model could be adapted later to use a sliding window approach.

* Drop input tanks and potentially output monthly values where the output.status is OOS  (be able to make up for this in model architecture or accuracy metrics) 

Tanks go in and out of service, but we won't be predicting values for out of service tanks. This will be determined on a monthly basis. After this is taken care of for each batch in training, get the number of tanks in service so the model can adjust its architecture. I want to make sure the model is going to be able to uniquely identify each tank. Maybe then we shouldn't turn off (reduce the input shape of tanks). 

Need a neat way of turning tanks on and off in the network. 

* Add cyclic time features sin/cos for year, month, and day

* Do one hot encoding on categorical features (TentativeQuality, TrueQuality)



LSTM/GRU-based Encoder (processes daily time series)
Input Shape = (NumMonths, DaysInMonth(dynamic), NumFeatures)

Output: Monthly embedding

monthly context integration:

    Use dense layer to combine monthly embeddings from the encoder with the monthly outputs from temp_out_df. Make sure to correctly align the right month (range of rows in temp_in_df) with the right row in temp_out_df

.	Input Encoding for Time-Series Data (Daily Inputs):
	•	Process daily input data (temp_in_df) through a Time-Series Encoder (e.g., LSTM/GRU).
	•	This encoder processes each sequence of daily data (for a given month) and generates a fixed-length embedding representing that month’s daily patterns.
Reason: This allows the model to understand the sequential structure and fluctuations in daily input data.
	2.	Historical Monthly Output Integration:
	•	Add the historical monthly outputs (temp_out_df) as a secondary input to the network.
	•	Pass these outputs through a Monthly Context Network (dense layers) to produce embeddings that summarize the historical context.
Reason: The historical outputs provide information about patterns in input-to-output relationships, helping the model assign importance to certain input features based on previous months’ outputs.

Fusion of Time-Series and Monthly Context:
	•	Concatenate the embeddings from the Time-Series Encoder and Monthly Context Network.
	•	Optionally, include cyclic time features (e.g., sin/cos encodings for months/years) at this stage to provide temporal awareness.
Reason: This combination allows the model to integrate the sequential input patterns with the contextual influence of historical outputs.
	4.	Prediction Layer for Future Inputs:
	•	Pass the fused embedding through fully connected layers to predict the daily inputs for the next month.
	•	Use a dense decoder or LSTM decoder to generate daily predictions for the next month.
Reason: This ensures the model focuses on predicting future daily inputs while leveraging historical output information as additional context.


Loss function: MSE Between daily input true and predicted values
Optimizer: Adam

Need an embedding layer so the model knows what tank its predicting, 

encoding layer to read time series data

context network to bring previous encoding layer and concatenate with the monthly outputs

decoder layer to predict the next og value series for each tank


The model needs to know which tank it is predicting values for. How can we accomplish this? Another embedding layer? 





