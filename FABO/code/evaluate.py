import pandas as pd

def get_average_stats(path):
    df = pd.read_csv(path)
    grouped = df.groupby('Experiment')[['Rank', "Found", "% in Top 250"]].mean()
    return grouped