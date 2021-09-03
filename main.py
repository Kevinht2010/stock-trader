import warnings

import numpy as np
import pandas as pd
import logging

from data_acquisition import get_stock_data
from agent import Agent
from train import train_model
from evaluate import evaluate_model


def main(stock_name, train_stock, val_stock, window_size=30,
         batch_size=32, ep_count=6, model_name="model",
         pretrained=False, debug=False):
    print('#'*50)
    print(">>> Initializing Agent...")
    agent = Agent(window_size, pretrained=pretrained, model_name=model_name)
    print("<<< Agent Initialized.")
    print(">>> Preparing Training Data...")
    train_data = get_stock_data(train_stock)
    print("<<< Training Data Prepared.")
    print(">>> Preparing Validation Data...")
    val_data = get_stock_data(val_stock)
    print("<<< Validation Data Prepared.")
    print('#'*50)
    print(">>> Start Training...")
    for episode in range(1, ep_count + 1):
        print('#'*50)
        print("Episode ", episode)
        print(">>> Training Model...")
        train_result = train_model(
            agent, episode, train_data, ep_count=ep_count,
            batch_size=batch_size, window_size=window_size
        )
        print("<<< Model Trained.")
        print(">>> Evaluating Model...")
        eval_result = evaluate_model(
            agent, val_data, window_size, debug
        )
        print("<<< Model Evaluated.")
        print("Results for Episode:", episode)
        print("Train Results: ", train_result)
        print("Eval Results: ", eval_result)
        break

    print("<<< Training Done.")


if __name__ == "__main__":
    df = pd.read_csv("data/stock_ids.csv")
    stock_ids = df["id"].to_numpy()
    # trains with first 10 stocks, with 30 day window and 6 epochs
    for idx in range(10):
        id = stock_ids[idx]
        print('#'*50)
        print("Training for:", id)
        if idx != 0:
            main(stock_name=id,
                 train_stock='data/train_data/{}.csv'.format(id),
                 val_stock='data/eval_data/{}.csv'.format(id)
                 )
        else:
            main(stock_name=id,
                 train_stock='data/train_data/{}.csv'.format(id),
                 val_stock='data/eval_data/{}.csv'.format(id)
                 )
        break
