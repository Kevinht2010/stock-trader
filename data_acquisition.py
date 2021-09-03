import pandas as pd


def get_stock_data(stock_file):
    df = pd.read_csv(stock_file)
    stock_data = df[['Adj Close', 'Volume']].values.tolist()
    return stock_data
