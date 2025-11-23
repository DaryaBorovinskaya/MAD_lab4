import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

file_path = './adult 3 (1).csv'
df = pd.read_csv(file_path)
df = df.replace('?', pd.NA).dropna()

numeric_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_cols = ['workclass', 'education', 'marital-status',
                    'occupation', 'relationship', 'race', 'gender',
                    'native-country']

df_analysis = df.copy()
df_analysis['income_num'] = (df_analysis['income'] == '>50K').astype(int)

X = df_analysis[numeric_cols]
y = df_analysis['income_num']

def get_processed_data():
    return df_analysis, X, y, numeric_cols, categorical_cols