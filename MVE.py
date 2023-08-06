import pandas as pd


df = pd.DataFrame({'x':[1,2,3], 'y':[4,5,6]})
print(f"original DataFrame: {df}")


# create logical series that creates a view if called on the original series
lg = df['x'] > 1
# manipulate original series in a view
df.loc[lg, 'y'] = 7
print(f">> 1. original DataFrame: {df}")

# save view to a new series (this will create a deep copy, if a pointer is not sufficient, i.e. if the view differs from the original data)
new_series = df.loc[lg, 'y']
# override the series by a plain number
new_series = 8
# assume you intended s.th. else
new_df = df[lg]
new_df['y'] = 9
print(f">> 2. original DataFrame: {df}, \nnew Series: {new_series}, \nnew DataFrame: {new_df}")

# iterate over rows (no view)
for idx, row in df.iterrows():
    if row['x'] > 1:
        row['y'] = 10
print(f">> 3. original DataFrame: {df}")


