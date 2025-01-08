import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Union

__date_format__ = '%Y-%m-%d'


def get_stock_data(stock_symbols=["AAPL", "MSFT", "GOOGL", "BA", "WBA", "INTC"], 
        start_date="2020-01-01", end_date="2024-10-31", 
        save_path='./stock_data', resave=False):
    
    print('... fetching real-world stock data ...')
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    stock_data = {}
    for symbol in stock_symbols:
        save_as = save_path/f"{symbol}.csv"
        if not save_as.is_file() or resave:
            try:
                import yfinance as yf
                stock_df = yf.download(symbol, start=start_date, end=end_date)
                stock_df = stock_df.droplevel(level='Ticker', axis=1)
                stock_df.columns.name = None
                stock_df.index.name = None
                stock_df.index = stock_df.index.tz_localize(None)
                stock_df.to_csv(save_as, index=True)
            except ImportError:
                # Install the required library
                # !pip install yfinance
                raise ImportError('The "yfinance" has not been installed!')
        else:
            stock_df = pd.read_csv(save_as, index_col=0, parse_dates=True)
        dates = [datetime.strptime(date.strftime('%Y-%m-%d'), '%Y-%m-%d') for date in stock_df.index]
        prices = stock_df["Adj Close"].to_list()
        stock_data[symbol] = dict(zip(dates, prices))
        
    return stock_data


class Stock:
    
    # make `default_stock_data` a private attribute
    __default_stock_data = get_stock_data()
    # alternatively, make `default_stock_data` a public attribute
#     default_stock_data = get_stock_data()
    
    def __init__(self, name, ticker, historical_prices):
        # ticker might change over time, `name` is intended to hold the permanent name of a stock
        self.name = name
        self.ticker = ticker
        self.historical_prices = historical_prices  # Dictionary of {date: price}
        self.trading_dates = sorted(historical_prices.keys())  # Only store valid trading dates
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.historical_prices[self.trading_dates[idx]]
        elif isinstance(idx, datetime):
            return self.historical_prices.get(idx, None)
        elif isinstance(idx, str):
            time_idx = datetime.strptime(idx, __date_format__)
            return self.historical_prices.get(time_idx, None)
        else:
            raise IndexError(f"Unknown index type {type(idx)}!")
        
    def __in__(self, time:datetime):
        return time in self.trading_dates
            
    def __repr__(self):
        k0 = self.trading_dates[0]
        v0 = self.historical_prices[k0]
        k0_str = k0.strftime('%Y-%m-%d')
        k1 = self.trading_dates[-1]
        v1 = self.historical_prices[k1]
        k1_str = k1.strftime('%Y-%m-%d')
        return f"name:{self.name}|ticker:{self.ticker}|data:{{{k0_str}:{v0:.2f}, ..., {k1_str}:{v1:.2f}}}"

    def get_most_recent_trading_date_before(self, before_date):
        return [date for date in self.trading_dates if date <= before_date][-1]

    def get_trading_dates_in_range(self, start_date, end_date):
        # Filter trading dates within the specified date range
        return [date for date in self.trading_dates if start_date <= date <= end_date]
    
    @classmethod
    def from_ticker(cls, symbol, stock_data=None):
        if stock_data is None:
            stock_data = cls.__default_stock_data
        return cls(symbol, symbol, stock_data[symbol])

    def calculate_moving_average(self, k):
        """
        Calculate the K-day moving average of the stock prices.

        :param k: Number of days for the moving average
        :return: A pandas Series with the moving average values
        """
        prices_series = pd.Series(self.historical_prices)
        moving_average = prices_series.rolling(window=k).mean()
        return moving_average

    def visualize(self, start_date, end_date, k_values=None, figsize=(10, 5)):
        """
        Visualize the stock prices and optionally plot K-day moving averages.

        :param start_date: Start date for the visualization
        :param end_date: End date for the visualization
        :param k_values: List of integers representing the K values for moving averages
        :return: The Axes object for further customization
        """
        trading_dates = self.get_trading_dates_in_range(start_date, end_date)
        prices = [self.historical_prices[date] for date in trading_dates]

        fig, ax = plt.subplots(figsize=figsize)
        # Plot the stock price with a thicker line
        ax.plot(trading_dates, prices, label=f"{self.name} ({self.ticker})", linewidth=1)
        
        # Plot moving averages for each k in k_values with thinner lines
        if k_values:
            for k in k_values:
                moving_average = self.calculate_moving_average(k)
                moving_average = moving_average.loc[start_date:end_date]
                ax.plot(moving_average.index, moving_average, label=f'{k}-Day MA', 
                    linewidth=.5, linestyle='--')

        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.set_title(f"Price of {self.name} from {start_date.strftime(__date_format__)} to {end_date.strftime(__date_format__)}")
        ax.legend()
        ax.grid(True)
        
        return ax


def annotate_transactions(ax, transact_df:pd.DataFrame, fontsize=None):
    
    legend_labels = set()
    for _, row in transact_df.iterrows():
        action = row['action']
        stock_name = row['stock_name']
        date = row['t']
        price = row['p']
        # Choose a color and marker based on the action
        color = 'green' if action == 'buy' else 'red'
        marker = '^' if action == 'buy' else 'v'
        # Add the transaction point
        ax.scatter(date, price, color=color, marker=marker)
        # Annotate the transaction on the plot
        # ax.annotate(f"{action} {stock_name}", (date, price), textcoords="offset points", xytext=(0,10), ha='center')
        # Add to legend only if not already added
        if action not in legend_labels:
            ax.scatter([], [], color=color, marker=marker, label=action)
            legend_labels.add(action)
        
    # Add the legend
    ax.legend(fontsize=fontsize)