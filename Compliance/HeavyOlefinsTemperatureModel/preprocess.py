# preprocess.py
import pandas as pd
import json
import calendar
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import tensorflow as tf

TANKS = ['TBD910', 'TBD911', 'TBD912']

def load_json(fp:str) -> list:
    with open(fp, 'r') as f:
        return json.load(f)
    
def load_temperature_df() -> pd.DataFrame:
    # Populating df with output from C# pipeline
    temp_data = load_json('./TempOutputs.json')

    rows = []
    for month in temp_data:
        ogv = month['OriginalData']
        pcv = month['ProcessedData']
        outv = month['OutputData']
        for og, pc, out in zip(ogv, pcv, outv):
            row = {'TimeStep': og['TimeStep']}
            for t in TANKS:
                row.update({
                    f'{t}OGValue': og[t]['Value'],
                    f'{t}OGTentativeQuality': og[t]['TentativeQuality'],
                    f'{t}OGTrueQuality': og[t]['TrueQuality'],
                    f'{t}ProcessedValue': pc[t]['Value'],
                    f'{t}ProcessedTentativeQuality': pc[t]['TentativeQuality'],
                    f'{t}ProcessedTrueQuality': pc[t]['TrueQuality'],
                    f'{t}Status': out['Status'],
                    f'{t}MonthlyAvgTemp': out['MonthlyAvgTemp'],
                    f'{t}AnnualAvgTemp': out['AnnualAvgTemp'],
                    f'{t}PeriodsOfGoodQualityDataPct': out['PeriodsOfGoodQualityDataPct'],
                    f'{t}PeriodsOfGoodQualityDataDays': out['PeriodsOfGoodQualityDataDays'],
                })
            rows.append(row)
    df = pd.DataFrame(rows)

    # True Quality only pertains to Processed values and Tentative Quality only pertains to OG values
    drop_cols = []
    for t in TANKS:
        drop_cols.append(f'{t}OGTrueQuality')
        drop_cols.append(f'{t}ProcessedTentativeQuality')
        drop_cols.append(f'{t}AnnualAvgTemp') # all Nan
    df.drop(columns=drop_cols, inplace=True)
    
    # The processed values and output values in df, say for day x, can not be calculated until day x + 1 has been observed
    # Reflect true availability of processed values
    # this aligns x'th processed value and output values with the x + 1th og values. 
    for t in TANKS:
        df[f'{t}ProcessedValue'] = df[f'{t}ProcessedValue'].shift(1) 
        df[f'{t}Status'] = df[f'{t}Status'].shift(1)
        df[f'{t}MonthlyAvgTemp'] = df[f'{t}MonthlyAvgTemp'].shift(1)
        #df[f'{t}AnnualAvgTemp'] = df[f'{t}AnnualAvgTemp'].shift(1)
        df[f'{t}PeriodsOfGoodQualityDataPct'] = df[f'{t}PeriodsOfGoodQualityDataPct'].shift(1)
        df[f'{t}PeriodsOfGoodQualityDataDays'] = df[f'{t}PeriodsOfGoodQualityDataDays'].shift(1)
    
    for col in df.select_dtypes(include=['object']).columns:
        if ('OGValue' in col) or ('ProcessedValue' in col) or ('MonthlyAvgTemp' in col):
            df[col] = pd.to_numeric(df[col], errors='coerce') # convert continuous columns to numeric (dropping nan's)
            
    return df.iloc[1:] # first day of processed/output data is not observable until we see the next day



def one_hot_encoder(dataframe:pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    for t in TANKS:
        df[f'{t}OGTentativeQuality'] = df[f'{t}OGTentativeQuality'].map({'G':1, 'B':0})
        df[f'{t}ProcessedTrueQuality'] = df[f'{t}ProcessedTrueQuality'].astype(int)
        df[f'{t}Status'] = df[f'{t}Status'].map({'IN SERVICE': 1, '-': 0})
    return df


def encode_timestamps(dataframe:pd.DataFrame) -> pd.DataFrame:
        df = dataframe.copy()
        
        t = 'TimeStep'
        df[t] = pd.to_datetime(df[t])

        yrs = df[t].dt.year 
        months = df[t].dt.month
        days = df[t].dt.day

        passed_days = (df[t] - pd.to_datetime(yrs.astype(str) + '-01-01')).dt.days + 1
        num_days_in_yr = yrs.apply(lambda yr: 366 if calendar.isleap(yr) else 365)
        pos_within_yr = passed_days / num_days_in_yr

        # Capture time-related trends spanning multiple years (ie: global warming)
        # ok to include min/max years outside of training set
        #   for example, we will always know the dates our prediction set will be based on
        #   Not ok to select a min(year) that is outside of data bounds
        min_yr = yrs.min()
        max_yr = yrs.max()
        pos_within_yr_scaled = (yrs - min_yr) / (max_yr - min_yr + 1)
        df['sin(year)'] = np.sin(2 * np.pi * pos_within_yr_scaled)
        df['cos(year)'] = np.cos(2 * np.pi * pos_within_yr_scaled)
        
        # Capture time-related/seasonality trends within each year (ie: winter vs summer)
        alpha_yr = 2 * np.pi * pos_within_yr
        df['sin(year_cycle)'] = np.sin(alpha_yr)
        df['cos(year_cycle)'] = np.cos(alpha_yr)

        # Capture seasonality within each month
        # NOTE: This and the daily cycle might not be as great of a feature as the yearly cycles
        
        # Current values are subtracted by 1 to ensure time scales begin at 0 radians
        df['sin(month)'] = np.sin(2 * np.pi * (months - 1) / 12.0)
        df['cos(month)'] = np.cos(2 * np.pi * (months - 1) / 12.0)

        # list containing how many days in the corresponding matching index month
        days_in_month = [calendar.monthrange(year, month)[1] for year, month in zip(yrs, months)]
        pos_within_month = (days - 1) / days_in_month
        alpha_day = 2 * np.pi * pos_within_month
        df['sin(day)'] = np.sin(alpha_day)
        df['cos(day)'] = np.cos(alpha_day)

       # df.drop(columns=[t], inplace=True)
        return df

def apply_feature_engineering(dataframe:pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()

    # IF data quality is bad these features will filter out the input sequences
    # Hopefully teaching our model to only worry about good quality data
    for t in TANKS:
        df[f'{t}OGValue*OGTentativeQuality'] = df[f'{t}OGValue'] * df[f'{t}OGTentativeQuality']
        df[f'{t}ProcessedValue*ProcessedTrueQuality'] = df[f'{t}ProcessedValue'] * df[f'{t}ProcessedTrueQuality']

    return df 

def data_split(dataframe:pd.DataFrame, val_pct:float, test_pct:float) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df = dataframe.copy()
    trainval, test = train_test_split(df, test_size=test_pct, shuffle=False)
    train, val = train_test_split(trainval, test_size=val_pct, shuffle=False)

    t = 'TimeStep'
    train_dates = train[t].reset_index(drop=True)
    val_dates = val[t].reset_index(drop=True)
    test_dates = test[t].reset_index(drop=True)

    train.drop(columns=[t], inplace=True)
    val.drop(columns=[t], inplace=True)
    test.drop(columns=[t], inplace=True)

    train.index = train_dates
    val.index = val_dates
    test.index = test_dates
    return train, val, test

def normalize_training_set(dataframe:pd.DataFrame) -> (pd.DataFrame, dict):
    df = dataframe.copy()
    min_max_cont_vals = {}
    for t in TANKS:
        og_col = f'{t}OGValue'
        if og_col in df.columns:
            min = df[og_col].min()
            max = df[og_col].max()
            min_max_cont_vals[og_col] = {'min': min, 'max': max}
            df[og_col] = (df[og_col] - min) / (max - min)
        p_col = f'{t}ProcessedValue'
        if p_col in df.columns:
            min = df[p_col].min()
            max = df[p_col].max()
            min_max_cont_vals[p_col] = {'min': min, 'max': max}
            df[p_col] = (df[p_col] - min) / (max - min)
        ma_col = f'{t}MonthlyAvgTemp'
        if ma_col in df.columns:
            min = df[ma_col].min()
            max = df[ma_col].max()
            min_max_cont_vals[ma_col] = {'min': min, 'max': max}
            df[ma_col] = (df[ma_col] - min) / (max - min)
        ogxog = f'{t}OGValue*OGTentativeQuality'
        if ogxog in df.columns:
            min = df[ogxog].min()
            max = df[ogxog].max()
            min_max_cont_vals[ogxog] = {'min': min, 'max': max}
            df[ogxog] = (df[ogxog] - min) / (max - min)
        pcxpc = f'{t}ProcessedValue*ProcessedTrueQuality'
        if pcxpc in df.columns:
            min = df[pcxpc].min()
            max = df[pcxpc].max()
            min_max_cont_vals[pcxpc] = {'min': min, 'max': max}
            df[pcxpc] = (df[pcxpc] - min) / (max - min)
    return df, min_max_cont_vals

def normalize_testing_set(dataframe:pd.DataFrame, min_max_cont_vals:dict) -> pd.DataFrame:
    df = dataframe.copy()
    for t in TANKS:
        og_col = f'{t}OGValue'
        if og_col in df.columns:
            min = min_max_cont_vals[og_col]['min']
            max = min_max_cont_vals[og_col]['max']
            df[og_col] = (df[og_col] - min) / (max - min)
        p_col = f'{t}ProcessedValue'
        if p_col in df.columns:
            min = min_max_cont_vals[p_col]['min']
            max = min_max_cont_vals[p_col]['max']
            df[p_col] = (df[p_col] - min) / (max - min)
        ma_col = f'{t}MonthlyAvgTemp'
        if ma_col in df.columns:
            min = min_max_cont_vals[ma_col]['min']
            max = min_max_cont_vals[ma_col]['max']
            df[ma_col] = (df[ma_col] - min) / (max - min)
        ogxog = f'{t}OGValue*OGTentativeQuality'
        if ogxog in df.columns:
            min = min_max_cont_vals[ogxog]['min']
            max = min_max_cont_vals[ogxog]['max']
            df[ogxog] = (df[ogxog] - min) / (max - min)
        pcxpc = f'{t}ProcessedValue*ProcessedTrueQuality'
        if pcxpc in df.columns:
            min = min_max_cont_vals[pcxpc]['min']
            max = min_max_cont_vals[pcxpc]['max']
            df[pcxpc] = (df[pcxpc] - min) / (max - min)
        
    return df

@dataclass
class Config:
    num_input_days:int
    feature_cols:list
    target_cols:list
    
def create_sequences(dataframe:pd.DataFrame, C:Config) -> (np.ndarray, np.ndarray, np.ndarray):
    df = dataframe.copy()
    masks, X, y = [], [], []
    for start in range(len(df) - C.num_input_days):
        
        input = df[C.feature_cols].iloc[start:start + C.num_input_days].values
        output = df[C.target_cols].iloc[start + C.num_input_days].values
        
        mask = []
        last_input_day = start + C.num_input_days - 1
        for t in TANKS:
            if ((df[f'{t}Status'].iloc[last_input_day] == 0) or
                (df[f'{t}OGTentativeQuality'].iloc[last_input_day] == 0) or
                (df[f'{t}ProcessedTrueQuality'].iloc[last_input_day] == 0)):
                mask.append(0) # don't make predictions on next day
            else:
                mask.append(1) # make prediction on next day
        X.append(input)
        y.append(output)
        masks.append(mask)
    return np.array(X), np.array(y), np.array(masks)

def create_tf_dataset(X:np.ndarray, y:np.ndarray, masks:np.ndarray, batch_size:int=32) -> tf.data.Dataset:
    # Batch size to be a power of 2 is best for hardware optimization
    dataset = tf.data.Dataset.from_tensor_slices((X, y, masks))
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
