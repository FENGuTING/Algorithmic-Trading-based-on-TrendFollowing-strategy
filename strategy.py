import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Literal

from .portfolio import Portfolio, FailedTransaction
from .stock import Stock


@dataclass
class HistPerformance: # record the historical performance of a portfolio 
    dates: List[datetime] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    
    def extend(self, hist_performance:'HistPerformance'):
        self.dates.extend(hist_performance.dates)
        self.values.extend(hist_performance.values)


@dataclass
class PendingTransaction:
    stock: Stock # the trading object
    action: Literal['buy', 'sell'] # the trading direction
    price: float # the desired trading price
    quantity: int = 0 # the desired trading quantity
    value: float = 1e-10 # the estimated value of the transaction
    

@dataclass
class TradableStocks:
    date: datetime
    to_buy: List[PendingTransaction]
    to_sell: List[PendingTransaction]
    
    def not_empty(self):
        return (len(self.to_buy) > 0) or (len(self.to_sell) > 0)


# Assuming Stock and Portfolio classes are already defined as before
# Only the TSTrendFollow class logic has been updated
class TSTrendFollow:
    
    def __init__(self, lookback_days=5, epsilon_up=0.05, epsilon_down=0.05, max_invest_ratio=1.0):
        self.lookback_days = lookback_days
        self.epsilon_up = epsilon_up
        self.epsilon_down = epsilon_down
        self.max_invest_ratio = max_invest_ratio
        self.portfolio = Portfolio()
        self.hist_perf = HistPerformance()
        self.pending_transactions: List[PendingTransaction] = []
        
    def reset(self):
        self.portfolio = Portfolio()
        self.hist_perf = HistPerformance()
        self.pending_transactions: List[PendingTransaction] = []
        
    def is_above(self, price, moving_average):
        return price >= moving_average*(1+self.epsilon_up)

    def is_below(self, price, moving_average):
        return price <= moving_average*(1-self.epsilon_down)
    
    def get_tradable_stocks(self, date:datetime, stock_pool:List[Stock]):
        tradable_stocks = TradableStocks(date, [], [])
        
        # Check each stock for tradability and trend-following criteria
        for stock in stock_pool:
            # Ensure the stock has a price on the given date
            current_price = stock[date]
            if current_price is None:
                continue

            # Calculate moving average for the past lookback_days trading dates
            past_prices = [stock[d] for d in stock.trading_dates if d < date][-self.lookback_days:]
            # Only consider if we have enough past data
            if len(past_prices) == self.lookback_days:
                moving_average = np.mean(past_prices)
                # Determine if the stock satisfies buy/sell condition
                if self.is_above(current_price, moving_average):
                    tradable_stocks.to_buy.append(PendingTransaction(stock, "buy", current_price, value=current_price-moving_average))
                elif self.is_below(current_price, moving_average) and (stock.name in self.portfolio.holdings):
                    tradable_stocks.to_sell.append(PendingTransaction(stock, "sell", current_price))
                else: # in-between
                    pass
                
        return tradable_stocks
    
    def trade_tradable_stocks(self, tradable_stocks:TradableStocks, verbose:bool):
        date = tradable_stocks.date

        # Execute sales for stocks that meet the sell condition
        # do this first, because it allows us to have more cash for buying
        for pt in tradable_stocks.to_sell:
            quantity = self.portfolio.holdings[pt.stock.name].quantity
            self.portfolio.sell_stock(pt.stock, quantity, date, pt.price, verbose=verbose)

        # Calculate budget share based on relative performance for buy stocks
        to_buy = tradable_stocks.to_buy
        scale = sum(pt.value for pt in to_buy) if to_buy else 1 # the normalization constant
        for pt in to_buy:
            # Allocate budget proportionally among stocks that satisfy buy criteria    
            allocation_ratio = pt.value / scale
            available_cash = self.portfolio.cash * self.max_invest_ratio * allocation_ratio
            pt.quantity = int(available_cash / pt.price) # might be zero!
        # Execute the buy transactions
        for pt in to_buy:
            if pt.quantity > 0:
                self.portfolio.buy_stock(pt.stock, pt.quantity, date, pt.price, verbose=verbose)

    def register_transaction(self, action:Literal['buy', 'sell'], stock:Stock, quantity:int):
        self.pending_transactions.append(
            PendingTransaction(stock, action, None, quantity)
        )
    
    def execute_pending_transactions(self, time:datetime, verbose):
        succeeded = [False]*len(self.pending_transactions)
        for i, pt in enumerate(self.pending_transactions):
            action = self.portfolio.buy_stock if pt.action == 'buy' else self.portfolio.sell_stock
            try:
                if verbose: print(f"... try transaction {pt} at time {time} ...")
                action(pt.stock, pt.quantity, time, verbose=verbose)
            except FailedTransaction as e:
                if verbose: print(e)
            else:
                if verbose: print('... succeeded ...')
                succeeded[i] = True
        self.pending_transactions = [pt for i, pt in enumerate(self.pending_transactions) if not succeeded[i]]
            
    def __call__(self, stock_pool:List[Stock], investment_amount:float, start_date:datetime, end_date:datetime, verbose=False):
        # Deposit initial investment
        self.portfolio.deposit_cash(investment_amount, start_date)

        # Generate list of all trading dates across the stock pool
        trading_dates = sorted(set([date for stock in stock_pool for date in stock.trading_dates 
            if start_date <= date <= end_date]))

        for date in trading_dates:
            self.execute_pending_transactions(date, verbose)
            tradable_stocks = self.get_tradable_stocks(date, stock_pool)
            if tradable_stocks.not_empty():
                self.trade_tradable_stocks(tradable_stocks, verbose)
        # update the portfolio performance history
        self.hist_perf.extend(self.summarize_historical_performance(start_date, end_date))
        
        # Return the final portfolio value at the end date
        return self.portfolio.get_value_c(end_date)
    
    def visualize_bt_results(self):
        # "bt" stands for back testing
        # Plotting portfolio performance
        plt.figure(figsize=(8, 5))
        plt.plot(self.hist_perf.dates, self.hist_perf.values, label="Portfolio Value")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.title("Trend-Following Strategy Portfolio Performance")
        plt.legend()
        plt.show()
    
    def summarize_historical_performance(self, start_date:datetime, end_date:datetime):
        portfolio_values = []
        n_days = (end_date - start_date).days + 1
        dates = [start_date+timedelta(days=i) for i in range(n_days)]
        for date in dates:
            portfolio_values.append(self.portfolio.get_value_h(date))
        return HistPerformance(dates, portfolio_values)
    
    @property
    def holdings(self):
        return self.portfolio.holdings