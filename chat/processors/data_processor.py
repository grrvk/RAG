import pandas as pd

def get_data(path='chat/data/datasets/updated_cocktails.csv'):
    df = pd.read_csv(path)
    cocktails_info_column = df['Combined']
    return cocktails_info_column.values.tolist()


data = get_data()