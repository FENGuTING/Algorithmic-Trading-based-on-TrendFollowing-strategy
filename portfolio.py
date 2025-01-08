import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Literal

from .stock import Stock


__time_format__ = '%Y-%m-%d'


@dataclass
class TRA: # for storing a single transaction record
    q: int # quantity
    p: float # price
    t: datetime # transaction time
    idx: int # global transaction idx
    
    def __iter__(self):
        return iter((self.q, self.p, self.t, self.idx))


@dataclass
class TRAExt: # for storing a single extended transaction record
    action: Literal['buy', 'sell', 'deposit-e', 'withdraw-e'] # transaction type
    stock_name: str # stock name
    q: int # quantity
    p: float # price
    t: datetime # transaction time
    idx: int # global transaction idx
    cash: float = field(init=False) # cash balance after the transaction
    
    def __repr__(self):
        return (f"TRAExt(action='{self.action}', "
                f"stock_name='{self.stock_name}', "
                f"quantity={self.q}, "
                f"price={self.p:.2f}, "
                f"time={self.t.strftime(__time_format__)}), "
                f"cash_balance={self.cash:.2f}")


@dataclass
class StockHolding:
    stock: Stock # the stock object that is being held
    quantity: int # the current holding amount of the stock object


@dataclass
class DepositRecord:
    amount: int
    time: datetime
    idx: int # global transaction idx
    withdraw: bool = False # whether this deposit is a withdrawal
    external: bool = False # whether this deposit is from external sources
    

@dataclass
class StockFlowData:
    stock_name: str
    inflow_Cash: float # we have spent this amount of cash on the given stock
    inflow_QTY: int # the quantity of the given stock has been increased by this amount
    outflow_Cash: float # we have earned this amount of cash on the given stock
    outflow_QTY: int # the quantity of the given stock has been decreased by this amount
    on_hand_QTY: int = field(init=False) # in net value, how many units of stock do we have?
    on_hand_Cash: int = field(init=False) # in net value, how many units of cash do we have?
    
    def __post_init__(self):
        self.on_hand_QTY = self.inflow_QTY - self.outflow_QTY
        self.on_hand_Cash = self.inflow_Cash - self.outflow_Cash


@dataclass
class StockPLData: # check `Portfolio.get_average_holding_cost`
    cost: float 
    price: float 
    quantity: int 
    total_PL: float
    unrealized_PL: float
    realized_PL: float


class FailedTransaction(Exception):
    def __init__(self, message):
        super().__init__(message)


def analyze_trade_flow(stock_name, buy_records:List[TRA], sell_records:List[TRA]):
    # this is called 'outflow_Cash' because we have spent money buying that stock
    outflow_Cash = sum([trans.q*trans.p for trans in buy_records])
    # this is called 'inflow_Cash' because we have earned money from selling that stock
    inflow_Cash = sum([trans.q*trans.p for trans in sell_records])
    # this is called 'inflow_QTY' because the stock quantity has been increased by that amount
    inflow_QTY = sum([trans.q for trans in buy_records])
    # this is called 'outflow_QTY' because the stock quantity has been decreased by that amount
    outflow_QTY = sum([trans.q for trans in sell_records])
    return StockFlowData(stock_name, inflow_Cash, inflow_QTY, outflow_Cash, outflow_QTY)


# Define the Portfolio class with buy and sell actions
class Portfolio:
    
    def __init__(self):
        self.cash = 0
        self.deposit_history: List[DepositRecord] = []
        # self.holdings[stock.name] = {"stock": stock, "quantity": quantity}
        self.holdings: Dict[str, StockHolding] = {}
        # self.trade_history[stock_name] = {'buy': [], 'sell': []}
        self.trade_history: \
            Dict[str, Dict[Literal['buy', 'sell'], List[TRA]]] = {}
        self.last_trade_time = None
        self.global_transaction_idx = 0

    def deposit_cash(self, amount, date, tracked=True):
        # sanity check, should not deposit earlier than the lastly recorded transaction
        self.check_then_update_trade_time(date)
        self.cash += amount
        if tracked:
            self.global_transaction_idx += 1
            self.deposit_history.append(DepositRecord(amount, date, self.global_transaction_idx, external=True))

    def buy_stock(self, stock:Stock, quantity:int, time:datetime|int, price:float=None, verbose:bool=False, tracked:bool=True):
        if isinstance(time, int):
            time = stock.trading_dates[time]
        # if the trading price has not been provided, query it on the fly
        if price is None:
            # sanity check 1, `date` may not be a trading date
            if time not in stock.historical_prices:
                raise FailedTransaction(f"No price available for {stock.name} on {time}.")
            price = stock[time]
        total_cost = price * quantity
        # sanity check 2, should not spend more than the amount of cash on hand
        if total_cost > self.cash:
            raise FailedTransaction("Insufficient funds to buy stock.")
        # sanity check 3, should not buy earlier than the lastly recorded transaction
        self.check_then_update_trade_time(time)
        # execute the transaction
        self.cash -= total_cost
        if stock.name in self.holdings:
            self.holdings[stock.name].quantity += quantity
        else:
            self.holdings[stock.name] = StockHolding(stock, quantity)
        if tracked:
            # record the transaction
            self.add_to_trade_history('buy', stock.name, quantity, price, time, verbose=verbose)
    
    def sell_stock(self, stock: Stock, quantity: int, time: datetime|int, price: float=None, verbose=False, tracked=True):
        if isinstance(time, int):
            time = stock.trading_dates[time]
        # if the trading price has not been provided, query it on the fly
        if price is None:
            # sanity check 1, `date` may not be a trading date
            if time not in stock.historical_prices:
                raise FailedTransaction(f"No price available for {stock.name} on {time}.")
            price = stock[time]
        # sanity check 2, should not sell more than the holding amount of the given stock
        if (stock.name not in self.holdings) or (self.holdings[stock.name].quantity < quantity):
            raise FailedTransaction("Insufficient stock quantity to sell.")
        # sanity check 3, should not sell earlier than the lastly recorded transaction
        self.check_then_update_trade_time(time)
        # execute the transaction
        total_revenue = price * quantity
        self.cash += total_revenue
        self.holdings[stock.name].quantity -= quantity
        if self.holdings[stock.name].quantity == 0:
            del self.holdings[stock.name]
        if tracked:
            # record the transaction
            self.add_to_trade_history('sell', stock.name, quantity, price, time, verbose=verbose)
    
    def check_then_update_trade_time(self, time):
        if self.last_trade_time is not None:
            if self.last_trade_time > time:
                raise FailedTransaction(f"Current trade time {time} is before the last trade time {self.last_trade_time}!")
        self.last_trade_time = time
    
    def add_to_trade_history(self, action, stock_name, quantity, price, time:datetime, verbose):
        if stock_name not in self.trade_history:
            self.trade_history[stock_name] = {'buy': [], 'sell': []}
        action_records = self.trade_history[stock_name][action]
        self.global_transaction_idx += 1
        # note that list_object.append is an inplace operation,
        # which modifies the list_object itself
        action_records.append(TRA(quantity, price, time, self.global_transaction_idx))
        self.deposit_history.append(DepositRecord(quantity*price, time, self.global_transaction_idx, withdraw=action=='buy'))
        if verbose:
            unit = 'units' if quantity > 1 else 'unit'
            info = f"{action} {quantity} {unit} of {stock_name} stock with price {price:.2f} on {time.strftime(__time_format__)}"
            n_char = len(info)
            print('+'*n_char, info, '-'*n_char, sep='\n')
    
    def undo_recent_transactions(self, n_transaction=1):
        trade_history = self.trade_history_flat[::-1]
        for i, trans_undo in enumerate(trade_history):
            if i >= n_transaction:
                break
            if trans_undo.action == 'buy' or trans_undo.action == 'sell':
                # delete the transaction to be reversed from the recorded history
                trans_rec = self.trade_history[trans_undo.stock_name][trans_undo.action].pop()
                dp_rec = self.deposit_history.pop()
                # sanity check, make sure we are using the correct information
                assert (dp_rec.amount == trans_undo.p*trans_undo.q) and (dp_rec.time == trans_undo.t) \
                    and (dp_rec.withdraw == (trans_undo.action == 'buy')) and (dp_rec.idx == trans_undo.idx), \
                    f"{dp_rec}, {trans_undo}"
                assert (trans_rec.p == trans_undo.p) and (trans_rec.q == trans_undo.q) and (trans_rec.t == trans_undo.t) \
                    and (trans_rec.idx == trans_undo.idx)
                # reverse the action
                new_act = self.sell_stock if trans_undo.action == 'buy' else self.buy_stock
                # get the corresponding instance method then undo the transaction
                # note that we will not record the undo transaction itself by passing `tracked=False`
                new_act(self.holdings[trans_undo.stock_name].stock, trans_undo.q, trans_undo.t, trans_undo.p, tracked=False)
            else: # a cash deposit/withdraw operation
                # delete the transaction to be reversed from the recorded history
                dp_rec = self.deposit_history.pop()
                # sanity check, make sure we are using the correct information
                assert (dp_rec.amount == trans_undo.p) and (dp_rec.time == trans_undo.t) and (dp_rec.idx == trans_undo.idx)
                # reverse the action
                # note that we will not record the undo transaction itself by passing `tracked=False`
                self.deposit_cash(-dp_rec.amount, dp_rec.time, tracked=False)
            # manually update the last trade time
            if i+1 < len(trade_history):
                self.last_trade_time = trade_history[i+1].t
            else:
                self.last_trade_time = None

    @property
    def holding_stocks(self):
        return list(self.holdings.keys())
    
    @property
    def trade_history_flat(self):
        _trade_history_flat: List[TRAExt] = \
            [TRAExt('withdraw-e' if dp_rec.withdraw else 'deposit-e', 'cash', 1, dp_rec.amount, dp_rec.time, dp_rec.idx) for dp_rec in self.deposit_history if dp_rec.external]
        for stock_name, stock_records in self.trade_history.items():
            for action in ['buy', 'sell']:
                records = [TRAExt(action, stock_name, *record) for record in stock_records[action]]
                _trade_history_flat.extend(records)
        _trade_history_flat = sorted(_trade_history_flat, key=lambda x: x.idx)
        # populate the cash balance history
        for i, trans in enumerate(_trade_history_flat):
            if i == 0:
                cash_balance = 0
            else:
                cash_balance = _trade_history_flat[i-1].cash           
            if trans.action == 'deposit-e' or trans.action == 'sell':
                _trade_history_flat[i].cash = cash_balance + trans.p*trans.q
            elif trans.action == 'withdraw-e' or trans.action == 'buy':
                _trade_history_flat[i].cash = cash_balance - trans.p*trans.q
            else:
                raise ValueError(f"Unknown transaction type: {trans.action}")
        return _trade_history_flat 
    
    def eval_realized_PL(self, buy_records:List[TRA], sell_records:List[TRA], favor_investor:bool):
        # adopt the maximizing realized cost principle when selling the "items" in inventory
        # note that by construction, the transaction records have been sorted by transaction time,
        # while each transaction record is of structure (quantity, price, date)
        # now, treat each purchase record as an "item" in the inventory
        inventory = buy_records.copy()
        # re-sort inventory items according to transaction price, from the cheapest one to the most expensive one
        inventory.sort(key=lambda x: x.p, reverse=not favor_investor)
        # print(inventory)
        realized_PL = 0 # 'PL' stands for P&L (profit and loss)
        sell_from_item = inventory.pop()
        sell_from_QTY = sell_from_item.q
        for to_sell_item in sell_records:
            to_sell_QTY = to_sell_item.q
            while (to_sell_QTY > 0) and (sell_from_QTY > 0):
                sold_QTY = min(sell_from_QTY, to_sell_QTY)
                to_sell_QTY -= sold_QTY
                sell_from_QTY -= sold_QTY
                # update `realized_pl` with the formula (sell_price - purchase_price)*sold_quantity
                realized_PL += (to_sell_item.p - sell_from_item.p)*sold_QTY
                if sell_from_QTY == 0:
                    # have sold out the to-sell item
                    # move to the next most expensive item in the inventory
                    if len(inventory) > 0:
                        sell_from_item = inventory.pop()
                        sell_from_QTY = sell_from_item.q
        try:
            assert to_sell_QTY == 0
        except UnboundLocalError:
            pass
        return realized_PL
    
    def get_market_price(self, stock_name, before_date):
        the_stock = self.holdings[stock_name].stock
        if before_date is None:
            # set it to the last trading date in the historical data of the given stock
            at_date = -1
        else:
            at_date = the_stock.get_most_recent_trading_date_before(before_date)
        # print(at_date)
        price_at_date = the_stock[at_date]
        return price_at_date
    
    def compute_total_PL(self, market_price:float, flow_data:StockFlowData):
        # sanity check, the resulted quantity should equal to what we have recorded
        if flow_data.on_hand_QTY > 0:
            _QTY_on_hand = self.holdings[flow_data.stock_name].quantity
            assert flow_data.on_hand_QTY == _QTY_on_hand, f"{flow_data.on_hand_QTY} vs {_QTY_on_hand}"
        else: # flow_data.on_hand_QTY=0
            assert flow_data.stock_name not in self.holdings
        # compute total P&L with the formula 
        #  (market_value_of_the_holding_stock + cash_inflow_from_the_sold_stock - cash_outflow_on_the_purchased_stock)
        market_value = flow_data.on_hand_QTY * market_price
        total_PL = market_value + flow_data.on_hand_Cash
        return total_PL
        
    def get_average_holding_cost(self, stock_name, before_date:datetime=None, favor_investor=True):
        buy_records = self.trade_history[stock_name]['buy']
        sell_records = self.trade_history[stock_name]['sell']
        flow_data = analyze_trade_flow(stock_name, buy_records, sell_records)
        market_price = self.get_market_price(stock_name, before_date) if flow_data.on_hand_QTY > 0 else 0
        # ------ 1. compute realized P&L (profit and loss) ------
        realized_PL = self.eval_realized_PL(buy_records, sell_records, favor_investor)
        # ------ 2. compute total P&L ------
        total_PL = self.compute_total_PL(market_price, flow_data)
        # ------ 3. compute unrealized P&L ------
        # now, the unrealized component of P&L can be computed as below
        unrealized_PL = total_PL - realized_PL
        # ------ 4. compute average cost ------
        # and the average holding cost can be inferred based on the following equation
        # unrealized_PL = (price_at_date - average_cost) * QTY_on_hand
        if flow_data.on_hand_QTY > 0:
            average_cost = market_price - unrealized_PL/flow_data.on_hand_QTY
        else:
            average_cost = 0
        return StockPLData(average_cost, market_price, flow_data.on_hand_QTY, total_PL, unrealized_PL, realized_PL)

    def get_stocks_PL_df(self, before, favor_investor):
        stock_names = self.trade_history.keys()
        rows_data = [self.get_average_holding_cost(stock_name, before, favor_investor) for stock_name in stock_names]
        data_df = pd.DataFrame(rows_data, 
            index=stock_names)
        data_df.columns = ['Cost', 'Price', 'QTY', 'P/L', 'Unrealized P/L', 'Realized P/L']
        return data_df
    
    def inspect_asset_value(self, before=None, favor_investor=True):
        before = datetime.now() if before is None else before
        time_info = f"Snapshot taken on: {before.strftime(__time_format__)}\n"
        time_info += f"Lastly traded on: {self.last_trade_time.strftime(__time_format__)}"
        cash_info = f"Cash: {self.cash:.2f}"
        data_df = self.get_stocks_PL_df(before, favor_investor)
        asset_info = data_df.to_string(float_format=lambda x: f"{x:.2f}")
        total_info = f"Investment P&L: {data_df['P/L'].sum():.2f}"
        return '\n'.join((time_info, cash_info, 'Stock:', asset_info, total_info))
    
    def get_value_c(self, time, holdings=None, cash=None):
        '''
        Calculate the total value of the **current** or a given portfolio holdings on a specific date,
          using the most recent trading date if necessary
        '''
        total_value = self.cash if cash is None else cash
        if holdings is None:
            holdings = self.holdings
        for holding in holdings.values():
            # Attempt to get the stock price for the requested date
            stock_price = holding.stock[time]
            # If no price available, use the most recent available trading date's price
            if stock_price is None:
                try:
                    most_recent_date = holding.stock.get_most_recent_trading_date_before(time)
                    stock_price = holding.stock[most_recent_date]
                except IndexError:
                    pass
            # Add value to total if a price is found
            if stock_price:
                total_value += stock_price * holding.quantity
        return total_value

    def get_value_h(self, time):
        '''
        Calculate the total value of the **historical** portfolio holdings on a specific date, 
          using the most recent trading date if necessary
        '''
        hist_holdings: Dict[str, StockHolding] = {}
        deposit_history = [dp_rec.amount for dp_rec in self.deposit_history 
                if (dp_rec.time <= time) and (not dp_rec.withdraw) and dp_rec.external]
        withdraw_history = [dp_rec.amount for dp_rec in self.deposit_history 
                if (dp_rec.time <= time) and (dp_rec.withdraw) and dp_rec.external]
        cash = sum(deposit_history) - sum(withdraw_history)
        for stock_name, stock_records in self.trade_history.items():
            buy_records = [trans for trans in stock_records['buy'] if trans.t <= time]
            sell_records = [trans for trans in stock_records['sell'] if trans.t <= time]
            flow_data = analyze_trade_flow(stock_name, buy_records, sell_records)
            if flow_data.on_hand_QTY > 0:
                hist_holdings[stock_name] = StockHolding(Stock.from_ticker(stock_name), flow_data.on_hand_QTY)
            cash += flow_data.on_hand_Cash
        return self.get_value_c(time, hist_holdings, cash)

    def __repr__(self):
        return self.inspect_asset_value()