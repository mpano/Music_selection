import pandas as pd
df = pd.read_excel('dc.xlsx')

duplicate = df[df.duplicated()]
print(duplicate)
input("delete duplicate type enter")
df.drop_duplicates(inplace = True)
print(df)

