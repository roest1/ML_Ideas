import numpy as np
# import tensorflow as tf
# print("TensorFlow version:", tf.__version__)
# print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# additional optimizations
#pip install nvidia-pyindex
#pip install nvidia-tensorflow[horovod]

'''
INPUTS:
====================
Each month will have these input tables:

Current Limit table:
(TankID, Status, Op Temp, Material Stored, Flow: annual gal/yr, Flow: hourly gal/yr, Temperature: annual avg)
one hot encoding for tankID will be necessary and then need to translate back later to see predicted output for each tank
status (OOS/In Service) can be one-hot encoded into 0 and 1
Material Stored is a little bit more variable, but it is in string format, so we will have to also do some one-hot encoding so the model 
can do its thing with them. 

Possible feature: (Flow: annual gal/yr) / 12.0 = monthly target

NOTE: tentative quality (below) is "G" when status = In Service and "B" when status = "OOS"
This tentative quality will have to be calculated for the input data:


Temperature Data:
(TankID, Tentative Value, Tentative quality,True value, True Quality)
tentative quality will be a string "G" or "B", we can one-hot encode these to 1 and 0
true quality will be boolean, which can also be one-hot encoded. 
- Thought process: choosing good qualities to be 1 and bad to be 0 will produce a higher weighing sum

Throughput data will have the same structure as above. 

Attached to each sensor in throughput data, is the tanks dimensions:
(TankID, Diameter (ft), height (ft), density (lb/gal))

The above data (values) get mutated twice before calculating:

Throughput Output data
(TankID, Monthly Throughput (gal/month), Monthly Throughput Target (gal/month), Max Pumping Rate (gal/hr), Max Pumping Rate Limit (gal/hr), Periods of Good Quality Data (%), Periods of Good Quality Data (hrs), # hrs Max Pumping Rate > Limit)

Temperature Output Data
(TankID, Monthly Avg Temperature (F), Annual Avg Temperature (F), Periods of Good Quality Data (%), Periods of Good Quality Data (days))

I have access to Temperature and throughput input time series data for each tank from like 2020 to 2024. 
I need to calculate tentative quality, and then i have c# code that can process the inputs into outputs. 
We can also pull any intermediate values to potentially use for features later on in modeling. 

I also have rolling average throughputs and temperatures outputs which could be used as a feature in the model.

From then I need to define a neural network architecture to be able to forecast output data of future months. 
'''


'''
(things to predict/forecast for all months or more so the most recent months and I need values for each tank) OUTPUTS:
====================
Monthly Throughput

Avg Monthly Temperature (F)
'''

import json
from pathlib import Path

def inspect_json_structure(json_path):
    def describe_structure(data, indent=0):
        """Recursively describe the structure of a JSON object."""
        if isinstance(data, dict):
            print(" " * indent + "{")
            for key, value in data.items():
                print(" " * (indent + 2) + f"'{key}': ", end="")
                describe_structure(value, indent + 2)
            print(" " * indent + "}")
        elif isinstance(data, list):
            print(f"[{len(data)} items]")
            if data:
                describe_structure(data[0], indent + 2)  # Check the first item
        else:
            print(type(data).__name__)  # Primitive types

    try:
        json_path = Path(json_path)  # Ensure it is a Path object
        if not json_path.exists():
            print(f"File not found: {json_path}")
            return
        
        # Read the JSON file
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        # Print the structure
        describe_structure(data)
    except Exception as e:
        print(f"An error occurred: {e}")


def inspect_json_structure_and_shapes(json_path):
    def describe_structure(data, indent=0):
        """Recursively describe the structure of a JSON object."""
        if isinstance(data, dict):
            print(" " * indent + "{")
            for key, value in data.items():
                print(" " * (indent + 2) + f"'{key}': ", end="")
                describe_structure(value, indent + 2)
            print(" " * indent + "}")
        elif isinstance(data, list):
            try:
                # Attempt to compute shape using NumPy if it is a list of numbers/lists
                array = np.array(data)
                if array.ndim > 1:
                    print(f"Array with shape {array.shape}")
                else:
                    print(f"List with {len(data)} items")
            except Exception:
                # Fallback if it can't be converted to a NumPy array
                print(f"[{len(data)} items]")
                if data:  # Check the structure of the first item
                    describe_structure(data[0], indent + 2)
        else:
            print(type(data).__name__)  # Primitive types

    try:
        json_path = Path(json_path)  # Ensure it is a Path object
        if not json_path.exists():
            print(f"File not found: {json_path}")
            return

        # Read the JSON file
        with open(json_path, 'r') as file:
            data = json.load(file)

        # Print the structure and shapes
        describe_structure(data)
    except Exception as e:
        print(f"An error occurred: {e}")


def extract_keys(json_path):
    def traverse(node, parent_key=""):
        keys = []
        if isinstance(node, dict):
            for key, value in node.items():
                full_key = f"{parent_key}.{key}" if parent_key else key
                keys.append(full_key)
                keys.extend(traverse(value, full_key))
        elif isinstance(node, list):
            if node:  # If the list is not empty, process only the first item
                list_key = f"{parent_key}[0]"
                keys.extend(traverse(node[0], list_key))
        return keys

    try:
        json_path = Path(json_path)
        if not json_path.exists():
            print(f"File not found: {json_path}")
            return

        with open(json_path, 'r') as file:
            data = json.load(file)

        keys = traverse(data)
        for key in keys:
            print(key)
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
# inspect_json_structure_and_shapes('your_json_file.json')


# Example usage
# Replace 'example.json' with the path to your JSON file
# inspect_json_structure(Path("..", "Preprocessing", "TempOutputs.json"))
# inspect_json_structure(Path("..", "Preprocessing", "ThroughputOutputs.json"))
# extract_keys(Path("..", "Preprocessing", "TempOutputs.json"))
extract_keys(Path("..", "Preprocessing", "ThroughputOutputs.json"))

'''
TempOutputs.json structure:

List<TeperatureCalculations>

.OriginalData // this is the time series value classes. Each class has the following indented attributes
    .TimeStep (dateTime)
    .TBD910 (TankMetaData)
    .TBD911
    .TBD912
        (TankMetaData has .Value (double), .TentativeQuality (string), .TrueQuality (boolean))

.ProcessedData // this is the same thing as originalData with some changes to the values. 
// This could be good for the model to understand how inputs need to be adjusted sometimes. 

we can perhaps concatenate original data and processed data along .TimeStep
then have two TankMetaData features (original and processed) for each tank

These are time series values so we need a way to assign the date a weight to each step in the series. 
For example to capture seasonal relationships. We have year, month, day, hour frequency. 

This list of time series data for each tank will correspond to an .OutputData object

.OutputData
    .TankID (string)
    .Status (string)
    .MonthlyAvgTemp (double)
    .AnnualAvgTemp (double)
    .PeriodsOfGoodQualityDataPct (double)
    .PeriodsOfGoodQualityDataDays (int)

I want the model to be able to predict the future. So I want it to understand the sensitivity of the original (input data) and then 
what needs to sometimes happen (ProcessedData or CalculatedData) and then how those series make up the output data (emissions records)
I want the model to be able to predict what will happen on the sensor side, do the necessary processing, and predict where the output (emissions records) will be. 

This is just my first attempt idea. Later, I will try to just predict the input data and do the calculations manually on the predictions from those sensors and then test the outputs from the manual calculations on the true backtested outputs. 


'''

'''
ThroughputOutputs.json structure:

List<ThroughputCalculations>

.OriginalData, .ProcessedData, and .CalculatedData are very similar to temperature, but they contain measurements for more tanks.

These tanks are:

TBD910
TBD911
TBD912
TBD913
TB3301
TBD301
TUT604
TUT605
TUT918
TOL400
TOL600
G354
G356

.OutputData
    .TankID (string)
    .Status (string)
    .MonthlyThroughput (double)
    .MonthlyThroughputTarget (double)
    .MaxPumpingRate (double)
    .MaxPumpingRateLimit (double)
    .PeriodsOfGoodQualityDataPct (double)
    .MaxPumpRateExceedingLimitHours (int)

.TankDimensionsReadOnly
    this has attributes for all the tanks
    .TBD910
    .TBD911
    etc...
        .Diameter (double)
        .Height (double)
        .Density (double)
'''

'''
Train Test Split

Use all historical data as training

Test on latest month
'''

'''
Inputs/Outputs:


Try this at first 

(from TempOutputs.json and ThroughputOutputs.json)

Inputs: 
    TempOutputs.json.OriginalData, .ProcessedData, .OutputData
    ThroughputOutputs.json.OriginalData, .ProcessedData, .CalculatedData, .OutputData, .TankDimensionsReadOnly

Outputs:
(for both temperature and throughputs)
    OriginalData (only values because we will have our model's predictions assume data quality is 100%)
    OutputData


Later on we can try removing the prediction of OutputData from the Outputs and just manually calculate it and use those metrics in the error functions



'''