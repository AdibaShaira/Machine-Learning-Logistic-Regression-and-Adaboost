import pandas as pd
import numpy as np
df = pd.read_csv("Telco.csv")
for col in df:
   dfn=pd.isna(df[col])
   print(pd.isna(df))
