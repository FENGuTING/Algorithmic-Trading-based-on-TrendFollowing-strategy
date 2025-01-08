# Trend-Following Stock Automated Trading Model

This project implements an automated stock trading model based on Object-Oriented Programming (OOP) principles. The model consists of three core modules: **Stock**, **Portfolio**, and **Strategy**, designed to simulate and optimize trading strategies.

---

## Features

### Stock Module
The **Stock** module is responsible for managing stock fundamental data:
- Fetches historical and real-time price and volume data.
- Supports data transformation into various analytical formats (e.g., time-series aggregation, indicator calculation).
- Provides flexible interfaces for other modules to access stock data.

### Portfolio Module
The **Portfolio** module manages the real-time status of the asset portfolio:
- Tracks dynamic changes in holdings (e.g., quantity, cost).
- Records trading history, including transaction time, price, and profit/loss details.
- Calculates and updates total asset value, floating profit/loss, and drawdowns in real-time to support decision-making.

### Strategy Module
The **Strategy** module generates and optimizes trading strategies:
- Generates buy and sell signals based on trend-following principles, such as moving average crossovers.
- Incorporates Bollinger Bands to optimize stop-loss and take-profit conditions, minimizing premature exits or deep losses.
- Offers flexible parameter adjustments for testing and optimizing trading logic.



