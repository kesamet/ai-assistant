from datetime import datetime, timedelta
from functools import cache

import numpy as np
import pandas as pd
import quantstats as qs
import ta
import yfinance as yf
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.agents import tool

# from openbb import obb


class Stock:
    def __init__(self, symbol: str):
        self.symbol = symbol

    @cache
    def get_prices(self) -> pd.DataFrame:
        start_date = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
        df = yf.download(self.symbol, start=start_date)
        df.columns = ["open", "high", "low", "close", "adjclose", "volume"]
        # df = obb.equity.price.historical(
        #     self.symbol, start_date=start_date, provider="yfinance"
        # ).to_df()
        # df.index = pd.to_datetime(df.index)
        return df

    @staticmethod
    def add_technicals(df, fillna: bool = True):
        df["pct_change"] = df["close"].pct_change() * 100
        df["sma20"] = ta.trend.SMAIndicator(
            close=df["close"], window=20, fillna=fillna
        ).sma_indicator()
        df["sma50"] = ta.trend.SMAIndicator(
            close=df["close"], window=50, fillna=fillna
        ).sma_indicator()
        df["sma150"] = ta.trend.SMAIndicator(
            close=df["close"], window=150, fillna=fillna
        ).sma_indicator()
        df["sma200"] = ta.trend.SMAIndicator(
            close=df["close"], window=200, fillna=fillna
        ).sma_indicator()
        df["atr"] = ta.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], window=14, fillna=fillna
        ).average_true_range()
        df["rsi"] = ta.momentum.RSIIndicator(
            df["close"], window=14, fillna=fillna
        ).rsi()
        df["52wk_high"] = df["close"].rolling(window=252).max()
        df["52wk_low"] = df["close"].rolling(window=252).min()

        adr = (df["high"] - df["low"]).rolling(window=20).mean()
        df["adr_pct"] = (adr / df["close"]).fillna(0) * 100

        y = df["close"].values
        p = np.polyfit(range(len(y)), y, deg=1)
        df["trendline_slope"] = p[0]
        return df

    def get_price_history(self, lookback: int = 30) -> pd.DataFrame:
        df = self.get_prices()
        if df.empty:
            return df

        df = self.add_technicals(df)
        df = df[-lookback:][::-1]
        return df

    def get_quantstats(self) -> pd.DataFrame:
        df = self.get_prices()
        if df.empty:
            return df

        stock_ret = qs.utils.download_returns(self.symbol, period=df.index)
        bench_ret = qs.utils.download_returns("^GSPC", period=df.index)
        stats = qs.reports.metrics(
            stock_ret, mode="full", benchmark=bench_ret, display=False
        )
        return stats

    # def get_fundamental_ratios(self) -> pd.DataFrame:
    #     return obb.equity.fundamental.ratios(symbol=self.symbol).to_df()

    # def get_fundamental_metrics(self) -> pd.DataFrame:
    #     return obb.equity.fundamental.metrics(symbol=self.symbol, with_ttm=True).to_df()#[::-1]

    # def get_fundamental_multiples(self) -> pd.DataFrame:
    #     return obb.equity.fundamental.multiples(symbol=self.symbol).to_df()

    # def get_profile(self) -> pd.DataFrame:
    #     return obb.equity.profile(symbol=self.symbol).to_df()


def _wrap(x: pd.DataFrame | str) -> str:
    if isinstance(x, pd.DataFrame):
        x = x.to_markdown()
        return f"<observation>\n\n{x}\n\n</observation>\n"

    x = str(x)
    return f"<observation>\n{x}\n</observation>\n"


class StockStatsInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol to fetch data for")


@tool(args_schema=StockStatsInput)
def get_stock_price_history(symbol: str) -> str:
    """Fetch a Stock's Price History by Symbol."""

    stock = Stock(symbol)
    try:
        df = stock.get_price_history()
        if df.empty:
            return _wrap(f"No data found for the given symbol {symbol}")
        return _wrap(df)
    except Exception as e:
        return _wrap(f"Error: {e}")


@tool(args_schema=StockStatsInput)
def get_stock_quantstats(symbol: str) -> str:
    """Fetch a Stock's Portfolio Analytics For Quants by Symbol."""

    stock = Stock(symbol)
    try:
        df = stock.get_quantstats()
        if df.empty:
            return _wrap(f"No data found for the given symbol {symbol}")
        return _wrap(df)
    except Exception as e:
        return _wrap(f"Error: {e}")


# @tool(args_schema=StockStatsInput)
# def get_stock_ratios(symbol: str) -> str:
#     """Fetch an Extensive Set of Financial and Accounting Ratios for a Given Company Over Time."""

#     stock = Stock(symbol)
#     try:
#         df = stock.get_fundamental_ratios()
#         if df.empty:
#             return _wrap(f"No data found for the given symbol {symbol}")
#         return _wrap(df)
#     except Exception as e:
#         return _wrap(f"Error: {e}")


# @tool(args_schema=StockStatsInput)
# def get_key_metrics(symbol: str) -> str:
#     """Fetch Fundamental Metrics by Symbol."""

#     try:
#         df = obb.equity.fundamental.metrics(symbol=symbol, with_ttm=True).to_df()
#         if df.empty:
#             return _wrap(f"No data found for the given symbol {symbol}")
#         return _wrap(df)
#     except Exception as e:
#         return _wrap(f"Error: {e}")


# @tool(args_schema=StockStatsInput)
# def get_stock_profile(symbol: str) -> str:
#     """Fetch a Company's General Information By Symbol. This includes company name, industry, and sector data."""

#     stock = Stock(symbol)
#     try:
#         df = stock.get_profile()
#         if df.empty:
#             return _wrap(f"No data found for the given symbol {symbol}")
#         return _wrap(df)
#     except Exception as e:
#         return _wrap(f"Error: {e}")


# @tool(args_schema=StockStatsInput)
# def get_valuation_multiples(symbol: str) -> str:
#     """Fetch a Company's Valuation Multiples by Symbol."""

#     stock = Stock(symbol)
#     try:
#         df = stock.get_fundamental_multiples()
#         if df.empty:
#             return _wrap(f"No data found for the given symbol {symbol}")
#         return _wrap(df)
#     except Exception as e:
#         return _wrap(f"Error: {e}")


# @tool
# def get_gainers() -> str:
#     """Fetch Top Price Gainers in the Stock Market."""

#     try:
#         gainers = obb.equity.discovery.gainers(sort="desc", provider='yfinance').to_df()
#         if gainers.empty:
#             return _wrap("No gainers found")
#         return _wrap(gainers)
#     except Exception as e:
#         return _wrap(f"Error: {e}")


# @tool
# def get_losers() -> str:
#     """Fetch Stock Market's Top Losers."""

#     try:
#         losers = obb.equity.discovery.losers(sort="desc", provider='yfinance').to_df()
#         if losers.empty:
#             return _wrap("No losers found")
#         return _wrap(losers)
#     except Exception as e:
#         return _wrap(f"Error: {e}")
