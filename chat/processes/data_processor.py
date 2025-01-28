import pandas as pd


class DataProcessor:
    def __init__(self, path):
        self.path_to_data = path

    def load_data(self):
        df = pd.read_csv(self.path_to_data)
        cocktails_info_column = df['Combined']
        return cocktails_info_column.values.tolist()

    def postprocess_data(self, response):
        count = len(response)
        return [count]