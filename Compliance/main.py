import tensorflow as tf
tf.config.optimizer.set_jit(True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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

# Example usage
# Replace 'example.json' with the path to your JSON file
# inspect_json_structure(Path("..", "Preprocessing", "TempOutputs.json"))
# inspect_json_structure(Path("..", "Preprocessing", "ThroughputOutputs.json"))
