import requests
import urllib
import urllib.request
import sys
import pandas as pd

from bs4 import BeautifulSoup

class Scraper:
    binsider_link = 'https://markets.businessinsider.com/index/market-capitalization/s&p_100?p={}'
    yhoo_link = 'https://query1.finance.yahoo.com/v7/finance/download/{}?period1=1562112000&period2=1625270400&interval=1d&events=history&includeAdjustedClose=true'
    train_data_path = 'train_data/{}.csv'
    eval_data_path = 'eval_data/{}.csv'

    def __init__(self):
        pass

    def get_page(self, page_number: int):
        page = requests.get(self.binsider_link.format(page_number))

        if page.status_code != 200:
            sys.exit("Error: Invalid page")

        src = page.content
        soup = BeautifulSoup(src, 'lxml')
        stock_links = soup.select('a[href*=stocks]')
        stock_ids = []

        ## first four & last four links are non-stock links
        for link in stock_links[5:-5]:
            stock_id = link['href'][8:]
            self.get_stock(stock_id)
            stock_ids.append(stock_id)
        
        stock_id_df = pd.DataFrame({'id': stock_ids})
        stock_id_df.to_csv('stock_ids.csv',encoding='utf-8',index=False)


    def get_stock(self, stock_id: str):
        download_link = (self.yhoo_link.format(stock_id))
        train_path = self.train_data_path.format(stock_id)
        eval_path = self.eval_data_path.format(stock_id)
        try:
            urllib.request.urlretrieve(download_link, train_path)
            self.transform_data(train_path, eval_path)
        except urllib.error.HTTPError as ex:
            print("Error: Invalid download - " + download_link)


    def transform_data(self, train_path: str, eval_path: str):
        data = pd.read_csv(train_path)
        train_data = data[:-100]
        eval_data = data[-100:]
        train_data.to_csv(train_path, encoding='utf-8', index=False)
        eval_data.to_csv(eval_path, encoding='utf-8', index=False)

a = Scraper()
a.get_page(1)