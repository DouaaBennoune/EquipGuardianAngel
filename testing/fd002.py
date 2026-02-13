import pandas as pd
import numpy as np
fd003_rul=pd.read_csv('/mnt/c/Users/PC/Downloads/AI/Equip Gardian Angel/train_FD003_aligned.csv',sep=',')
fd003=fd003_rul.drop(columns='RUL')
fd003.to_csv('fd003.csv',index=False)
