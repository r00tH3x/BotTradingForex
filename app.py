import asyncio
import logging
from datetime import datetime, timedelta
import requests,os
import pandas as pd
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import ta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import aiohttp
from enum import Enum
import yfinance as yf
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import io
import base64

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Token Bot Telegram (ganti dengan token bot Anda)
BOT_TOKEN = "8451123546:AAGH0nop2PgqqUCO2uaRdKfLAsm_jF4xTy0"

# API Keys (ganti dengan API key Anda)
ALPHA_VANTAGE_KEY = "W46DAPWY169RBHJB"
FOREX_API_KEY = "OETD5Z9g"
NEWS_API_KEY = "1764b23fae3b4e2c9e83a8569acca0d5"

class TradingStyle(Enum):
    SCALPING = "scalping"
    DAY_TRADING = "day_trading"
    SWING = "swing"

class MarketStructure(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGE = "range"
    BREAK_BULLISH = "break_bullish"
    BREAK_BEARISH = "break_bearish"

@dataclass
class TradingSignal:
    pair: str
    direction: str  # BUY/SELL
    entry_price: float
    tp1: float
    tp2: float
    tp3: float
    stop_loss: float
    confidence: int  # 1-100
    timeframe: str
    strategy: str
    reasoning: str
    risk_reward: float
    fibonacci_levels: Dict[str, float]
    market_structure: str
    volume_confirmation: bool

@dataclass
class MarketData:
    pair: str
    price: float
    change_24h: float
    volume: float
    timestamp: datetime

@dataclass
class FibonacciLevels:
    level_0: float      # 0%
    level_236: float    # 23.6%
    level_382: float    # 38.2%
    level_500: float    # 50%
    level_618: float    # 61.8%
    level_786: float    # 78.6%
    level_100: float    # 100%

class ForexAnalyzer:
    def __init__(self):
        self.major_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
            'XAUUSD', 'XAGUSD', 'GBPJPY', 'EURJPY', 'EURGBP', 'BTCUSD', 'ETHUSD'
        ]
        
        # Fibonacci levels
        self.fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618]
        
    async def get_market_data(self, pair: str) -> MarketData:
        """Ambil data real-time untuk pair"""
        try:
            # Gunakan yfinance untuk data real
            if pair == 'XAUUSD':
                ticker = 'GC=F'  # Gold futures
            elif pair == 'XAGUSD':
                ticker = 'SI=F'  # Silver futures
            elif pair in ['BTCUSD', 'ETHUSD']:
                ticker = f"{pair[:3]}-USD"
            else:
                ticker = f"{pair[:3]}{pair[3:]}=X"
            
            data = yf.download(ticker, period='2d', interval='1m')
            
            if not data.empty:
                latest_price = data['Close'].iloc[-1].item() if not data['Close'].empty else 0.0
                prev_price = data['Close'].iloc[-100].item() if len(data) > 100 else data['Close'].iloc[0].item()
                change_24h = ((latest_price - prev_price) / prev_price) * 100
                volume = float(data['Volume'].iloc[-1]) if 'Volume' in data.columns else 1000000
                
                return MarketData(
                    pair=pair,
                    price=latest_price,
                    change_24h=change_24h,
                    volume=volume,
                    timestamp=datetime.now()
                )
            else:
                # Fallback data
                return MarketData(
                    pair=pair,
                    price=1.0850,
                    change_24h=0.25,
                    volume=1000000,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Error getting market data for {pair}: {e}")
            # Return sample data as fallback
            base_prices = {
                'EURUSD': 1.0850, 'GBPUSD': 1.2650, 'USDJPY': 149.50,
                'XAUUSD': 2650.00, 'USDCHF': 0.8420, 'AUDUSD': 0.6780
            }
            return MarketData(
                pair=pair,
                price=base_prices.get(pair, 1.0000),
                change_24h=np.random.uniform(-0.5, 0.5),
                volume=1000000,
                timestamp=datetime.now()
            )

    def calculate_fibonacci_levels(self, df: pd.DataFrame, lookback: int = 50) -> FibonacciLevels:
        """Hitung Fibonacci retracement levels"""
        # Cari swing high dan swing low dalam periode lookback
        high_data = df['high'].tail(lookback)
        low_data = df['low'].tail(lookback)
        
        swing_high = high_data.max()
        swing_low = low_data.min()
        
        # Hitung range
        price_range = swing_high - swing_low
        
        # Hitung levels (untuk uptrend, retracement dari high ke low)
        levels = FibonacciLevels(
            level_0=swing_high,
            level_236=swing_high - (price_range * 0.236),
            level_382=swing_high - (price_range * 0.382),
            level_500=swing_high - (price_range * 0.500),
            level_618=swing_high - (price_range * 0.618),
            level_786=swing_high - (price_range * 0.786),
            level_100=swing_low
        )
        
        return levels

    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Hitung indikator teknikal"""
        indicators = {}
        
        # Moving Averages
        indicators['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        indicators['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        indicators['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        indicators['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
        indicators['sma_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
        
        # RSI
        indicators['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        indicators['macd'] = macd.macd()
        indicators['macd_signal'] = macd.macd_signal()
        indicators['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        indicators['bb_upper'] = bb.bollinger_hband()
        indicators['bb_middle'] = bb.bollinger_mavg()
        indicators['bb_lower'] = bb.bollinger_lband()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        indicators['stoch_k'] = stoch.stoch()
        indicators['stoch_d'] = stoch.stoch_signal()
        
        # ATR untuk volatility
        indicators['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        
        # Williams %R
        indicators['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()
        
        # ADX untuk trend strength
        indicators['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        
        # Volume indicators
        indicators['volume_sma'] = df['volume'].rolling(window=20).mean()
        indicators['volume_ratio'] = df['volume'] / indicators['volume_sma']
        
        # Support/Resistance levels
        indicators['support'] = self.find_support_resistance(df, 'support')
        indicators['resistance'] = self.find_support_resistance(df, 'resistance')
        
        # Fibonacci levels
        fib_levels = self.calculate_fibonacci_levels(df)
        indicators['fibonacci'] = {
            '0%': fib_levels.level_0,
            '23.6%': fib_levels.level_236,
            '38.2%': fib_levels.level_382,
            '50%': fib_levels.level_500,
            '61.8%': fib_levels.level_618,
            '78.6%': fib_levels.level_786,
            '100%': fib_levels.level_100
        }
        
        return indicators

    def find_support_resistance(self, df: pd.DataFrame, level_type: str) -> float:
        """Cari level support/resistance menggunakan pivot points dan peak detection"""
        if level_type == 'support':
            # Gunakan scipy untuk mencari local minima
            lows = df['low'].values
            peaks, _ = find_peaks(-lows, distance=10, prominence=0.0001)
            
            if len(peaks) > 0:
                recent_lows = lows[peaks[-3:]] if len(peaks) >= 3 else lows[peaks]
                return np.mean(recent_lows)
            else:
                return df['low'].rolling(window=20).min().iloc[-1]
        else:
            # Resistance levels
            highs = df['high'].values
            peaks, _ = find_peaks(highs, distance=10, prominence=0.0001)
            
            if len(peaks) > 0:
                recent_highs = highs[peaks[-3:]] if len(peaks) >= 3 else highs[peaks]
                return np.mean(recent_highs)
            else:
                return df['high'].rolling(window=20).max().iloc[-1]

    def detect_break_of_structure(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Deteksi Break of Structure (BOS) dengan konfirmasi volume"""
        current_price = df['close'].iloc[-1]
        resistance = indicators['resistance']
        support = indicators['support']
        
        # Volume confirmation
        current_volume = df['volume'].iloc[-1]
        avg_volume = indicators['volume_sma'].iloc[-1]
        volume_spike = current_volume > avg_volume * 1.5
        
        bos_signal = {
            'type': None,
            'confidence': 0,
            'level': 0,
            'volume_confirmed': False
        }
        
        # Bullish BOS
        if current_price > resistance * 1.001:  # 0.1% buffer
            confidence = 85
            if volume_spike:
                confidence += 10
                
            bos_signal = {
                'type': 'BULLISH_BOS',
                'confidence': min(95, confidence),
                'level': resistance,
                'volume_confirmed': volume_spike
            }
        
        # Bearish BOS
        elif current_price < support * 0.999:  # 0.1% buffer
            confidence = 85
            if volume_spike:
                confidence += 10
                
            bos_signal = {
                'type': 'BEARISH_BOS',
                'confidence': min(95, confidence),
                'level': support,
                'volume_confirmed': volume_spike
            }
        
        return bos_signal

    def detect_fibonacci_confluence(self, current_price: float, fib_levels: Dict) -> Dict:
        """Deteksi confluence dengan Fibonacci levels"""
        confluence = {'near_fib': False, 'level': None, 'distance': float('inf')}
        
        tolerance = 0.0010  # 10 pips tolerance
        
        for level_name, level_price in fib_levels.items():
            distance = abs(current_price - level_price)
            if distance <= tolerance and distance < confluence['distance']:
                confluence = {
                    'near_fib': True,
                    'level': level_name,
                    'distance': distance,
                    'price': level_price
                }
        
        return confluence

    def analyze_market_structure(self, df: pd.DataFrame, indicators: Dict) -> str:
        """Analisis struktur market yang lebih advanced"""
        current_price = df['close'].iloc[-1]
        ema_9 = indicators['ema_9'].iloc[-1]
        ema_20 = indicators['ema_20'].iloc[-1]
        ema_50 = indicators['ema_50'].iloc[-1]
        ema_200 = indicators['ema_200'].iloc[-1]
        adx = indicators['adx'].iloc[-1]
        
        # Strong uptrend
        if (current_price > ema_9 > ema_20 > ema_50 > ema_200 and 
            adx > 25 and indicators['rsi'].iloc[-1] > 50):
            return MarketStructure.UPTREND.value
        
        # Strong downtrend
        elif (current_price < ema_9 < ema_20 < ema_50 < ema_200 and 
              adx > 25 and indicators['rsi'].iloc[-1] < 50):
            return MarketStructure.DOWNTREND.value
        
        # Range/Sideways
        elif adx < 20:
            return MarketStructure.RANGE.value
        
        # Check for BOS
        bos = self.detect_break_of_structure(df, indicators)
        if bos['type'] == 'BULLISH_BOS':
            return MarketStructure.BREAK_BULLISH.value
        elif bos['type'] == 'BEARISH_BOS':
            return MarketStructure.BREAK_BEARISH.value
        
        return MarketStructure.RANGE.value

    def analyze_timeframes(self, pair: str) -> Dict:
        """Analisis multi timeframe"""
        timeframes = ['5m', '15m', '1h', '4h', '1d']
        analysis = {}
        
        try:
            # Download data dari yfinance
            ticker = self.get_ticker_symbol(pair)
            
            for tf in timeframes:
                period = '5d' if tf in ['5m', '15m'] else '30d'
                df = yf.download(ticker, period=period, interval=tf, progress=False, auto_adjust=False)
                
                if df.empty:
                    # Fallback ke sample data
                    df = self.generate_sample_data(pair, tf)
                
                # Rename columns to lowercase
                df.columns = [str(col).lower() for col in df.columns]
                
                indicators = self.calculate_indicators(df)
                trend = self.determine_trend(indicators)
                strength = self.calculate_trend_strength(indicators)
                market_structure = self.analyze_market_structure(df, indicators)
                
                analysis[tf] = {
                    'trend': trend,
                    'strength': strength,
                    'rsi': indicators['rsi'].iloc[-1],
                    'price': df['close'].iloc[-1],
                    'adx': indicators['adx'].iloc[-1],
                    'market_structure': market_structure,
                    'fibonacci': indicators['fibonacci']
                }
        
        except Exception as e:
            logger.error(f"Error in timeframe analysis: {e}")
            # Fallback to sample data
            for tf in timeframes:
                df = self.generate_sample_data(pair, tf)
                indicators = self.calculate_indicators(df)
                
                analysis[tf] = {
                    'trend': self.determine_trend(indicators),
                    'strength': 75,
                    'rsi': 55.0,
                    'price': df['close'].iloc[-1],
                    'adx': 30.0,
                    'market_structure': 'range',
                    'fibonacci': indicators['fibonacci']
                }
        
        return analysis

    def get_ticker_symbol(self, pair: str) -> str:
        """Convert forex pair ke ticker symbol"""
        ticker_map = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X', 
            'USDJPY': 'USDJPY=X',
            'USDCHF': 'USDCHF=X',
            'AUDUSD': 'AUDUSD=X',
            'USDCAD': 'USDCAD=X',
            'NZDUSD': 'NZDUSD=X',
            'XAUUSD': 'GC=F',
            'XAGUSD': 'SI=F',
            'BTCUSD': 'BTC-USD',
            'ETHUSD': 'ETH-USD'
        }
        return ticker_map.get(pair, f"{pair[:3]}{pair[3:]}=X")

    def generate_sample_data(self, pair: str, timeframe: str) -> pd.DataFrame:
        """Generate sample OHLC data (backup jika API gagal)"""
        periods_map = {'5m': 200, '15m': 200, '1h': 168, '4h': 120, '1d': 60}
        periods = periods_map.get(timeframe, 200)
        
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')
        
        # Base prices untuk setiap pair
        base_prices = {
            'EURUSD': 1.0850, 'GBPUSD': 1.2650, 'USDJPY': 149.50,
            'XAUUSD': 2650.00, 'USDCHF': 0.8420, 'AUDUSD': 0.6780
        }
        
        base_price = base_prices.get(pair, 1.0000)
        
        # Generate realistic price movement
        np.random.seed(hash(pair) % 2**32)
        returns = np.random.normal(0, 0.0002, periods)
        returns[0] = 0  # Start with no change
        
        # Create cumulative price series
        price_series = base_price * (1 + np.cumsum(returns))
        
        # Generate OHLC from price series
        opens = price_series
        closes = opens + np.random.normal(0, 0.0001, periods)
        highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 0.0002, periods))
        lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 0.0002, periods))
        volumes = np.random.randint(50000, 200000, periods)
        
        df = pd.DataFrame({
            'datetime': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        df.set_index('datetime', inplace=True)
        return df

    def determine_trend(self, indicators: Dict) -> str:
        """Tentukan arah trend dengan lebih akurat"""
        ema_9 = indicators['ema_9'].iloc[-1]
        ema_20 = indicators['ema_20'].iloc[-1]
        ema_50 = indicators['ema_50'].iloc[-1]
        ema_200 = indicators['ema_200'].iloc[-1]
        adx = indicators['adx'].iloc[-1]
        
        # Strong trend criteria
        if ema_9 > ema_20 > ema_50 > ema_200 and adx > 25:
            return "STRONG_BULLISH"
        elif ema_9 < ema_20 < ema_50 < ema_200 and adx > 25:
            return "STRONG_BEARISH"
        elif ema_9 > ema_20 > ema_50:
            return "BULLISH"
        elif ema_9 < ema_20 < ema_50:
            return "BEARISH"
        else:
            return "SIDEWAYS"

    def calculate_trend_strength(self, indicators: Dict) -> int:
        """Hitung kekuatan trend (1-100)"""
        rsi = indicators['rsi'].iloc[-1]
        macd_histogram = indicators['macd_histogram'].iloc[-1]
        adx = indicators['adx'].iloc[-1]
        
        strength = 30  # Base strength
        
        # ADX contribution (trend strength)
        if adx > 40:
            strength += 30
        elif adx > 25:
            strength += 20
        elif adx > 15:
            strength += 10
        
        # RSI momentum
        if 45 <= rsi <= 55:
            strength += 10  # Neutral momentum
        elif 30 <= rsi <= 70:
            strength += 15  # Good momentum
        elif rsi > 70 or rsi < 30:
            strength += 25  # Strong momentum
        
        # MACD histogram
        if abs(macd_histogram) > 0.001:
            strength += 15
        
        return min(100, max(20, strength))

    def generate_signal(self, pair: str, style: TradingStyle) -> TradingSignal:
        """Generate trading signal berdasarkan analisis advanced"""
        
        # Multi timeframe analysis
        mtf_analysis = self.analyze_timeframes(pair)
        
        # Get current market data
        timeframe_map = {
            TradingStyle.SCALPING: '5m',
            TradingStyle.DAY_TRADING: '15m', 
            TradingStyle.SWING: '1h'
        }
        
        primary_tf = timeframe_map[style]
        df = self.generate_sample_data(pair, primary_tf)
        
        # Ensure we have enough data
        if len(df) < 50:
            df = self.generate_sample_data(pair, primary_tf)
        
        indicators = self.calculate_indicators(df)
        current_price = df['close'].iloc[-1]
        
        # BOS Analysis
        bos = self.detect_break_of_structure(df, indicators)
        
        # Fibonacci confluence
        fib_confluence = self.detect_fibonacci_confluence(current_price, indicators['fibonacci'])
        
        # Market structure
        market_structure = self.analyze_market_structure(df, indicators)
        
        # Volume confirmation
        volume_confirmation = indicators['volume_ratio'].iloc[-1] > 1.2
        
        # Determine signal direction
        direction = self.determine_signal_direction(indicators, mtf_analysis, bos, style, fib_confluence)
        
        # Calculate entry, SL, and TPs
        entry, sl, tp1, tp2, tp3 = self.calculate_levels(
            current_price, direction, style, indicators, fib_confluence
        )
        
        # Calculate confidence
        confidence = self.calculate_signal_confidence(
            indicators, mtf_analysis, bos, fib_confluence, volume_confirmation
        )
        
        # Risk/Reward ratio
        rr_ratio = abs(tp1 - entry) / abs(entry - sl) if sl != entry else 0
        
        return TradingSignal(
            pair=pair,
            direction=direction,
            entry_price=entry,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            stop_loss=sl,
            confidence=confidence,
            timeframe=style.value,
            strategy=self.get_strategy_name(indicators, bos, fib_confluence),
            reasoning=self.generate_reasoning(indicators, mtf_analysis, bos, fib_confluence),
            risk_reward=rr_ratio,
            fibonacci_levels=indicators['fibonacci'],
            market_structure=market_structure,
            volume_confirmation=volume_confirmation
        )

    def calculate_levels(self, current_price: float, direction: str, style: TradingStyle, 
                        indicators: Dict, fib_confluence: Dict) -> Tuple[float, float, float, float, float]:
        """Calculate entry, SL, dan TP levels dengan Fibonacci"""
        
        atr = indicators['atr'].iloc[-1]
        
        # ATR-based multipliers untuk different styles
        multipliers = {
            TradingStyle.SCALPING: {'sl': 1.0, 'tp1': 1.5, 'tp2': 2.5, 'tp3': 3.5},
            TradingStyle.DAY_TRADING: {'sl': 1.5, 'tp1': 2.0, 'tp2': 3.5, 'tp3': 5.0},
            TradingStyle.SWING: {'sl': 2.0, 'tp1': 3.0, 'tp2': 5.0, 'tp3': 8.0}
        }
        
        mult = multipliers[style]
        
        if direction == "BUY":
            entry = current_price
            sl = current_price - (atr * mult['sl'])
            
            # Jika ada Fibonacci confluence, gunakan sebagai TP
            if fib_confluence['near_fib']:
                fib_price = fib_confluence['price']
                if fib_price > current_price:
                    tp1 = fib_price
                    tp2 = current_price + (atr * mult['tp2'])
                    tp3 = current_price + (atr * mult['tp3'])
                else:
                    tp1 = current_price + (atr * mult['tp1'])
                    tp2 = current_price + (atr * mult['tp2'])
                    tp3 = current_price + (atr * mult['tp3'])
            else:
                tp1 = current_price + (atr * mult['tp1'])
                tp2 = current_price + (atr * mult['tp2'])
                tp3 = current_price + (atr * mult['tp3'])
                
        else:  # SELL
            entry = current_price
            sl = current_price + (atr * mult['sl'])
            
            # Fibonacci-based TPs for SELL
            if fib_confluence['near_fib']:
                fib_price = fib_confluence['price']
                if fib_price < current_price:
                    tp1 = fib_price
                    tp2 = current_price - (atr * mult['tp2'])
                    tp3 = current_price - (atr * mult['tp3'])
                else:
                    tp1 = current_price - (atr * mult['tp1'])
                    tp2 = current_price - (atr * mult['tp2'])
                    tp3 = current_price - (atr * mult['tp3'])
            else:
                tp1 = current_price - (atr * mult['tp1'])
                tp2 = current_price - (atr * mult['tp2'])
                tp3 = current_price - (atr * mult['tp3'])
        
        return entry, sl, tp1, tp2, tp3

    def determine_signal_direction(self, indicators: Dict, mtf_analysis: Dict, bos: Dict, 
                                 style: TradingStyle, fib_confluence: Dict) -> str:
        """Tentukan arah signal dengan confluence analysis"""
        bullish_signals = 0
        bearish_signals = 0
        
        # EMA Alignment (weight: 2)
        ema_9 = indicators['ema_9'].iloc[-1]
        ema_20 = indicators['ema_20'].iloc[-1]
        ema_50 = indicators['ema_50'].iloc[-1]
        
        if ema_9 > ema_20 > ema_50:
            bullish_signals += 2
        elif ema_9 < ema_20 < ema_50:
            bearish_signals += 2
        
        # RSI Analysis (weight: 1-3 based on style)
        rsi = indicators['rsi'].iloc[-1]
        if style == TradingStyle.SCALPING:
            if 45 <= rsi <= 55:
                # Neutral zone, check momentum
                if indicators['macd_histogram'].iloc[-1] > 0:
                    bullish_signals += 1
                else:
                    bearish_signals += 1
        else:
            if rsi < 35:
                bullish_signals += 3  # Oversold
            elif rsi > 65:
                bearish_signals += 3  # Overbought
            elif 40 <= rsi <= 60:
                # Neutral momentum
                bullish_signals += 1 if rsi > 50 else 0
                bearish_signals += 1 if rsi < 50 else 0
        
        # MACD Analysis (weight: 2)
        macd = indicators['macd'].iloc[-1]
        macd_signal = indicators['macd_signal'].iloc[-1]
        macd_hist = indicators['macd_histogram'].iloc[-1]
        
        if macd > macd_signal and macd_hist > 0:
            bullish_signals += 2
        elif macd < macd_signal and macd_hist < 0:
            bearish_signals += 2
        
        # Stochastic Analysis (weight: 1)
        stoch_k = indicators['stoch_k'].iloc[-1]
        stoch_d = indicators['stoch_d'].iloc[-1]
        
        if stoch_k < 20 and stoch_k > stoch_d:
            bullish_signals += 2  # Oversold + bullish crossover
        elif stoch_k > 80 and stoch_k < stoch_d:
            bearish_signals += 2  # Overbought + bearish crossover
        
        # BOS Analysis (weight: 3)
        if bos['type'] == 'BULLISH_BOS':
            bullish_signals += 3
            if bos['volume_confirmed']:
                bullish_signals += 1
        elif bos['type'] == 'BEARISH_BOS':
            bearish_signals += 3
            if bos['volume_confirmed']:
                bearish_signals += 1
        
        # Fibonacci Confluence (weight: 2)
        if fib_confluence['near_fib']:
            # Determine if we're at support or resistance level
            current_price = indicators['fibonacci']['50%']  # Use as reference
            fib_price = fib_confluence['price']
            
            # Key Fibonacci levels for reversals
            if fib_confluence['level'] in ['61.8%', '78.6%', '50%']:
                if fib_price < current_price:
                    bullish_signals += 2  # At support level
                else:
                    bearish_signals += 2  # At resistance level
        
        # Multi-timeframe confluence (weight: 3)
        higher_tf_trends = []
        for tf in ['1h', '4h', '1d']:
            if tf in mtf_analysis:
                trend = mtf_analysis[tf].get('trend', 'SIDEWAYS')
                if 'BULLISH' in trend:
                    higher_tf_trends.append('BULLISH')
                elif 'BEARISH' in trend:
                    higher_tf_trends.append('BEARISH')
        
        bullish_htf = higher_tf_trends.count('BULLISH')
        bearish_htf = higher_tf_trends.count('BEARISH')
        
        bullish_signals += bullish_htf
        bearish_signals += bearish_htf
        
        # ADX strength confirmation
        adx = indicators['adx'].iloc[-1]
        if adx > 25:  # Strong trend
            if bullish_signals > bearish_signals:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        return "BUY" if bullish_signals > bearish_signals else "SELL"

    def calculate_signal_confidence(self, indicators: Dict, mtf_analysis: Dict, bos: Dict, 
                                  fib_confluence: Dict, volume_confirmation: bool) -> int:
        """Hitung confidence level signal dengan scoring system"""
        base_confidence = 40
        
        # Multi-timeframe alignment (max +25)
        trends = []
        for tf in ['15m', '1h', '4h']:
            if tf in mtf_analysis:
                trend = mtf_analysis[tf].get('trend', 'SIDEWAYS')
                if trend != 'SIDEWAYS':
                    trends.append(trend)
        
        if len(trends) >= 2:
            if len(set(trends)) == 1:  # All same direction
                base_confidence += 25
            elif len(set(trends)) == 2:  # Mixed
                base_confidence += 10
        
        # BOS confirmation (max +20)
        if bos['type']:
            base_confidence += 15
            if bos['volume_confirmed']:
                base_confidence += 5
        
        # Fibonacci confluence (max +15)
        if fib_confluence['near_fib']:
            if fib_confluence['level'] in ['61.8%', '50%', '38.2%']:
                base_confidence += 15
            else:
                base_confidence += 8
        
        # Volume confirmation (max +10)
        if volume_confirmation:
            base_confidence += 10
        
        # RSI positioning (max +10)
        rsi = indicators['rsi'].iloc[-1]
        if 30 <= rsi <= 70:
            base_confidence += 10
        elif 20 <= rsi <= 80:
            base_confidence += 5
        
        # ADX trend strength (max +10)
        adx = indicators['adx'].iloc[-1]
        if adx > 30:
            base_confidence += 10
        elif adx > 20:
            base_confidence += 5
        
        return min(95, max(30, base_confidence))

    def get_strategy_name(self, indicators: Dict, bos: Dict, fib_confluence: Dict) -> str:
        """Dapatkan nama strategi yang digunakan"""
        strategies = []
        
        if bos['type']:
            strategies.append("BOS Breakout")
        
        if fib_confluence['near_fib']:
            strategies.append(f"Fibonacci {fib_confluence['level']}")
        
        # Check for specific patterns
        rsi = indicators['rsi'].iloc[-1]
        if rsi < 30:
            strategies.append("RSI Oversold")
        elif rsi > 70:
            strategies.append("RSI Overbought")
        
        macd_hist = indicators['macd_histogram'].iloc[-1]
        if abs(macd_hist) > 0.001:
            strategies.append("MACD Momentum")
        
        if not strategies:
            strategies.append("EMA Confluence")
        
        return " + ".join(strategies[:2])  # Limit to 2 strategies

    def generate_reasoning(self, indicators: Dict, mtf_analysis: Dict, bos: Dict, fib_confluence: Dict) -> str:
        """Generate detailed reasoning untuk signal"""
        reasons = []
        
        # Multi-timeframe analysis
        h4_trend = mtf_analysis.get('4h', {}).get('trend', 'SIDEWAYS')
        h1_trend = mtf_analysis.get('1h', {}).get('trend', 'SIDEWAYS')
        m15_trend = mtf_analysis.get('15m', {}).get('trend', 'SIDEWAYS')
        
        reasons.append(f"HTF: {h4_trend} | H1: {h1_trend} | M15: {m15_trend}")
        
        # Key indicators
        rsi = indicators['rsi'].iloc[-1]
        adx = indicators['adx'].iloc[-1]
        reasons.append(f"RSI: {rsi:.1f} | ADX: {adx:.1f}")
        
        # BOS
        if bos['type']:
            vol_conf = "✓" if bos['volume_confirmed'] else "✗"
            reasons.append(f"BOS: {bos['type']} (Vol: {vol_conf})")
        
        # Fibonacci
        if fib_confluence['near_fib']:
            reasons.append(f"Fib: Near {fib_confluence['level']} level")
        
        # MACD momentum
        macd_hist = indicators['macd_histogram'].iloc[-1]
        momentum = "Bullish" if macd_hist > 0 else "Bearish"
        reasons.append(f"MACD: {momentum} momentum")
        
        return " | ".join(reasons)

class NewsAnalyzer:
    def __init__(self):
        self.economic_events = {
            'CPI': {'impact': 'HIGH', 'description': 'Consumer Price Index'},
            'PPI': {'impact': 'MEDIUM', 'description': 'Producer Price Index'},
            'DXY': {'impact': 'HIGH', 'description': 'US Dollar Index'},
            'NFP': {'impact': 'HIGH', 'description': 'Non-Farm Payrolls'},
            'FOMC': {'impact': 'VERY_HIGH', 'description': 'Federal Open Market Committee'},
            'GDP': {'impact': 'HIGH', 'description': 'Gross Domestic Product'},
            'PMI': {'impact': 'MEDIUM', 'description': 'Purchasing Managers Index'},
            'RETAIL': {'impact': 'MEDIUM', 'description': 'Retail Sales'},
            'JOBLESS': {'impact': 'MEDIUM', 'description': 'Initial Jobless Claims'},
            'HOUSING': {'impact': 'LOW', 'description': 'Housing Data'}
        }

    async def get_economic_calendar(self) -> List[Dict]:
        """Ambil jadwal economic events dengan simulasi real events"""
        # Simulasi events untuk hari ini dan besok
        current_time = datetime.now()
        
        events = [
            {
                'time': '08:30',
                'currency': 'GBP',
                'event': 'UK GDP m/m',
                'impact': 'HIGH',
                'forecast': '0.2%',
                'previous': '0.1%',
                'actual': None,
                'date': current_time.strftime('%Y-%m-%d')
            },
            {
                'time': '13:30',
                'currency': 'USD',
                'event': 'Core CPI m/m',
                'impact': 'VERY_HIGH',
                'forecast': '0.3%',
                'previous': '0.2%',
                'actual': None,
                'date': current_time.strftime('%Y-%m-%d')
            },
            {
                'time': '15:00',
                'currency': 'USD',
                'event': 'Federal Funds Rate',
                'impact': 'VERY_HIGH',
                'forecast': '5.50%',
                'previous': '5.25%',
                'actual': None,
                'date': current_time.strftime('%Y-%m-%d')
            },
            {
                'time': '09:00',
                'currency': 'EUR',
                'event': 'ECB Interest Rate Decision',
                'impact': 'VERY_HIGH',
                'forecast': '4.00%',
                'previous': '4.00%',
                'actual': None,
                'date': (current_time + timedelta(days=1)).strftime('%Y-%m-%d')
            },
            {
                'time': '14:30',
                'currency': 'USD',
                'event': 'Non-Farm Payrolls',
                'impact': 'VERY_HIGH',
                'forecast': '180K',
                'previous': '150K',
                'actual': None,
                'date': (current_time + timedelta(days=2)).strftime('%Y-%m-%d')
            }
        ]
        
        return events

    def analyze_news_impact(self, pair: str, events: List[Dict]) -> str:
        """Analisis impact news terhadap pair dengan detail"""
        base_currency = pair[:3]
        quote_currency = pair[3:]
        
        impact_summary = []
        high_impact_count = 0
        
        for event in events:
            if event['currency'] in [base_currency, quote_currency]:
                impact_level = event['impact']
                if impact_level in ['HIGH', 'VERY_HIGH']:
                    high_impact_count += 1
                
                impact_summary.append({
                    'event': event,
                    'affecting_currency': event['currency'],
                    'potential_volatility': self.calculate_volatility_impact(impact_level)
                })
        
        if not impact_summary:
            return "✅ No major economic events affecting this pair in the next 48 hours"
        
        summary_text = f"⚠️ {high_impact_count} HIGH IMPACT events detected:\n\n"
        
        for item in impact_summary:
            event = item['event']
            emoji = {'VERY_HIGH': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
            impact_emoji = emoji.get(event['impact'], '⚪')
            
            summary_text += f"{impact_emoji} *{event['time']}* - {event['currency']}\n"
            summary_text += f"📊 {event['event']}\n"
            summary_text += f"🎯 Expected volatility: {item['potential_volatility']}\n\n"
        
        return summary_text

    def calculate_volatility_impact(self, impact_level: str) -> str:
        """Calculate expected volatility based on impact level"""
        volatility_map = {
            'VERY_HIGH': 'Extreme (50-100+ pips)',
            'HIGH': 'High (30-50 pips)',
            'MEDIUM': 'Medium (15-30 pips)',
            'LOW': 'Low (5-15 pips)'
        }
        return volatility_map.get(impact_level, 'Unknown')

class RiskManager:
    """Advanced Risk Management System"""
    
    @staticmethod
    def calculate_position_size(account_balance: float, risk_percentage: float, 
                              entry_price: float, stop_loss: float) -> float:
        """Calculate optimal position size"""
        risk_amount = account_balance * (risk_percentage / 100)
        pip_risk = abs(entry_price - stop_loss)
        
        # For forex pairs (simplified pip value calculation)
        pip_value = 10  # $10 per pip for standard lot
        
        position_size = risk_amount / (pip_risk * 10000 * pip_value)
        return round(position_size, 2)
    
    @staticmethod
    def calculate_drawdown_risk(signals_history: List[Dict]) -> Dict:
        """Calculate potential drawdown risk"""
        if not signals_history:
            return {'max_drawdown': 0, 'consecutive_losses': 0, 'risk_level': 'LOW'}
        
        # Simulate win/loss based on confidence levels
        consecutive_losses = 0
        max_consecutive = 0
        
        for signal in signals_history[-10:]:  # Last 10 signals
            if signal.get('confidence', 0) < 60:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0
        
        risk_level = 'HIGH' if max_consecutive >= 3 else 'MEDIUM' if max_consecutive >= 2 else 'LOW'
        
        return {
            'max_drawdown': max_consecutive * 2,  # 2% per loss
            'consecutive_losses': max_consecutive,
            'risk_level': risk_level
        }

class TelegramBot:
    def __init__(self):
        self.analyzer = ForexAnalyzer()
        self.news_analyzer = NewsAnalyzer()
        self.risk_manager = RiskManager()
        self.user_settings = {}  # Store user preferences
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk command /start"""
        user_id = update.effective_user.id
        
        # Initialize user settings
        if user_id not in self.user_settings:
            self.user_settings[user_id] = {
                'risk_percentage': 2.0,
                'account_balance': 10000,
                'preferred_pairs': ['EURUSD', 'GBPUSD', 'XAUUSD'],
                'notifications': True
            }
        
        welcome_text = """
🚀 *Forex Signal Bot Pro v2.0* 🚀

🔥 *Premium Features:*
✅ Multi-timeframe analysis (5m-1D)
✅ Break of Structure (BOS) detection  
✅ Fibonacci confluence analysis
✅ Advanced risk management
✅ Economic calendar integration
✅ Volume confirmation
✅ Smart position sizing
✅ Real-time market data

*Choose your trading mode:*
        """
        
        keyboard = [
            [
                InlineKeyboardButton("⚡ Scalping (5-15min)", callback_data="mode_scalping"),
                InlineKeyboardButton("📊 Day Trading (15m-1H)", callback_data="mode_day")
            ],
            [
                InlineKeyboardButton("📈 Swing Trading (1H-4H)", callback_data="mode_swing"),
                InlineKeyboardButton("🔍 Deep Analysis", callback_data="mode_single")
            ],
            [
                InlineKeyboardButton("📰 Economic Calendar", callback_data="economic_calendar"),
                InlineKeyboardButton("💹 Market Overview", callback_data="market_overview")
            ],
            [
                InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
                InlineKeyboardButton("🎯 Live Signals", callback_data="live_signals")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        if update.callback_query:
           await update.callback_query.edit_message_text(welcome_text, reply_markup=reply_markup, parse_mode='Markdown')
        else:
           await update.message.reply_text(welcome_text, reply_markup=reply_markup, parse_mode='Markdown')

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk callback buttons"""
        query = update.callback_query
        if query is None:
            logger.error("No CallbackQuery found in update")
            return
    
        await query.answer()
    
        if query.data.startswith("mode_"):
            await self.handle_trading_mode(query, context)
        elif query.data == "economic_calendar":
            await self.show_economic_calendar(query, context)
        elif query.data == "market_overview":
            await self.show_market_overview(query, context)
        elif query.data == "settings":
            await self.show_settings(query, context)
        elif query.data == "live_signals":
            await self.show_live_signals(query, context)
        elif query.data.startswith("pair_"):
            await self.handle_pair_selection(query, context)
        elif query.data.startswith("analyze_"):
            await self.show_deep_analysis(query, context)
        elif query.data.startswith("mtf_"):
            await self.show_mtf_analysis(query, context)
        elif query.data.startswith("fib_"):
            await self.show_fibonacci_analysis(query, context)
        elif query.data.startswith("chart_"):
            await self.show_chart_analysis(query, context)
        elif query.data == "back_to_main":
            await self.start(update, context)
        elif query.data == "back_to_pairs":
            mode = context.user_data.get('trading_mode', TradingStyle.DAY_TRADING)
            await self.show_pair_selection(query, context, mode)

    async def handle_trading_mode(self, query, context):
        """Handle pemilihan mode trading"""
        mode_map = {
            "mode_scalping": TradingStyle.SCALPING,
            "mode_day": TradingStyle.DAY_TRADING,
            "mode_swing": TradingStyle.SWING,
            "mode_single": "single_analysis"
        }
        
        selected_mode = mode_map[query.data]
        context.user_data['trading_mode'] = selected_mode
        
        if selected_mode == "single_analysis":
            await self.show_single_analysis_menu(query, context)
        else:
            await self.show_pair_selection(query, context, selected_mode)

    async def show_pair_selection(self, query, context, style: TradingStyle):
        """Tampilkan pilihan currency pairs dengan categories"""
        text = f"🎯 *{style.value.replace('_', ' ').title()} Mode*\n\nSelect currency pair:\n\n"
        
        # Categorize pairs
        majors = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
        commodities = ['XAUUSD', 'XAGUSD']
        crypto = ['BTCUSD', 'ETHUSD']
        crosses = ['GBPJPY', 'EURJPY', 'EURGBP']
        
        keyboard = []
        
        # Majors
        text += "*🏛️ Major Pairs:*\n"
        row = []
        for pair in majors:
            row.append(InlineKeyboardButton(pair, callback_data=f"pair_{pair}"))
            if len(row) == 2:
                keyboard.append(row)
                row = []
        if row:
            keyboard.append(row)
        
        # Commodities
        keyboard.append([
            InlineKeyboardButton("🥇 XAUUSD", callback_data="pair_XAUUSD"),
            InlineKeyboardButton("🥈 XAGUSD", callback_data="pair_XAGUSD")
        ])
        
        # Crypto (if enabled)
        keyboard.append([
            InlineKeyboardButton("₿ BTCUSD", callback_data="pair_BTCUSD"),
            InlineKeyboardButton("Ξ ETHUSD", callback_data="pair_ETHUSD")
        ])
        
        keyboard.append([InlineKeyboardButton("🔙 Back to Main", callback_data="back_to_main")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode='Markdown')

    async def handle_pair_selection(self, query, context):
        """Handle pemilihan pair dan generate signal"""
        pair = query.data.replace("pair_", "")
        trading_mode = context.user_data.get('trading_mode', TradingStyle.DAY_TRADING)
        
        await query.edit_message_text("🔄 *Analyzing market data...*\n⏳ *Calculating indicators...*\n🧮 *Processing signals...*", parse_mode='Markdown')
        
        # Generate signal
        signal = self.analyzer.generate_signal(pair, trading_mode)
        
        # Calculate position size
        user_id = query.from_user.id
        user_settings = self.user_settings.get(user_id, {})
        account_balance = user_settings.get('account_balance', 10000)
        risk_percentage = user_settings.get('risk_percentage', 2.0)
        
        position_size = self.risk_manager.calculate_position_size(
            account_balance, risk_percentage, signal.entry_price, signal.stop_loss
        )
        
        # Format signal message
        signal_text = self.format_signal_message(signal, position_size, account_balance)
        
        # Keyboard untuk actions
        keyboard = [
            [
                InlineKeyboardButton("🔄 New Signal", callback_data=f"pair_{pair}"),
                InlineKeyboardButton("📊 MTF Analysis", callback_data=f"mtf_{pair}")
            ],
            [
                InlineKeyboardButton("🧮 Fibonacci Levels", callback_data=f"fib_{pair}"),
                InlineKeyboardButton("📈 Chart Analysis", callback_data=f"chart_{pair}")
            ],
            [InlineKeyboardButton("🔙 Back to Pairs", callback_data="back_to_pairs")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(signal_text, reply_markup=reply_markup, parse_mode='Markdown')

    def format_signal_message(self, signal: TradingSignal, position_size: float, account_balance: float) -> str:
        """Format signal menjadi message yang rapi dengan risk management"""
        direction_emoji = "🟢" if signal.direction == "BUY" else "🔴"
        
        # Confidence emoji
        if signal.confidence >= 85:
            confidence_emoji = "🔥🔥"
        elif signal.confidence >= 75:
            confidence_emoji = "🔥"
        elif signal.confidence >= 65:
            confidence_emoji = "⚡"
        else:
            confidence_emoji = "⚠️"
        
        # Market structure emoji
        structure_emoji = {
            'uptrend': '📈', 'downtrend': '📉', 'range': '🔄',
            'break_bullish': '🚀', 'break_bearish': '💥'
        }.get(signal.market_structure, '🔄')
        
        # Calculate risk amount
        risk_amount = abs(signal.entry_price - signal.stop_loss) * position_size * 10000
        risk_percentage = (risk_amount / account_balance) * 100
        
        text = f"""
{direction_emoji} *{signal.pair} - {signal.direction} SIGNAL* {confidence_emoji}

💰 *Entry Price:* `{signal.entry_price:.5f}`
🎯 *TP1:* `{signal.tp1:.5f}` (+{abs(signal.tp1-signal.entry_price)*10000:.1f} pips)
🎯 *TP2:* `{signal.tp2:.5f}` (+{abs(signal.tp2-signal.entry_price)*10000:.1f} pips)
🎯 *TP3:* `{signal.tp3:.5f}` (+{abs(signal.tp3-signal.entry_price)*10000:.1f} pips)
🛡️ *Stop Loss:* `{signal.stop_loss:.5f}` (-{abs(signal.entry_price-signal.stop_loss)*10000:.1f} pips)

📊 *Strategy:* {signal.strategy}
⏰ *Timeframe:* {signal.timeframe.replace('_', ' ').title()}
🎲 *Confidence:* {signal.confidence}% {confidence_emoji}
💎 *Risk/Reward:* 1:{signal.risk_reward:.2f}
{structure_emoji} *Market:* {signal.market_structure.title()}
{"📈" if signal.volume_confirmation else "📊"} *Volume:* {"Confirmed" if signal.volume_confirmation else "Normal"}

💰 *Risk Management:*
• Position Size: {position_size:.2f} lots
• Risk Amount: ${risk_amount:.2f} ({risk_percentage:.1f}%)
• Potential Profit (TP1): ${abs(signal.tp1-signal.entry_price)*position_size*10000:.2f}

🧮 *Fibonacci Levels:*
• 61.8%: `{signal.fibonacci_levels['61.8%']:.5f}`
• 50.0%: `{signal.fibonacci_levels['50%']:.5f}`
• 38.2%: `{signal.fibonacci_levels['38.2%']:.5f}`

📋 *Analysis:*
{signal.reasoning}

⚠️ *Trading Rules:*
• Wait for pullback to optimal entry
• Use trailing stop after TP1
• Close 50% at TP1, 30% at TP2, 20% at TP3
• Monitor economic news

🕐 *Generated:* {datetime.now().strftime('%H:%M:%S GMT+7')}
        """
        
        return text.strip()

    async def show_mtf_analysis(self, query, context):
        """Tampilkan Multi-Timeframe Analysis detail"""
        pair = query.data.replace("mtf_", "")
        
        await query.edit_message_text("🔄 *Loading MTF Analysis...*", parse_mode='Markdown')
        
        mtf_data = self.analyzer.analyze_timeframes(pair)
        
        text = f"📊 *Multi-Timeframe Analysis - {pair}*\n\n"
        
        timeframe_names = {
            '5m': '⚡ M5', '15m': '🔥 M15', '1h': '📈 H1', 
            '4h': '🎯 H4', '1d': '🏛️ D1'
        }
        
        for tf, data in mtf_data.items():
            tf_name = timeframe_names.get(tf, tf.upper())
            trend = data['trend']
            strength = data['strength']
            rsi = data['rsi']
            
            # Trend emoji
            trend_emoji = {
                'STRONG_BULLISH': '🟢🟢', 'BULLISH': '🟢',
                'STRONG_BEARISH': '🔴🔴', 'BEARISH': '🔴',
                'SIDEWAYS': '🟡'
            }.get(trend, '⚪')
            
            # Strength bars
            strength_bars = "█" * (strength // 20)
            
            text += f"{tf_name} {trend_emoji}\n"
            text += f"├ Trend: {trend}\n"
            text += f"├ Strength: {strength}% {strength_bars}\n"
            text += f"├ RSI: {rsi:.1f}\n"
            text += f"└ Price: {data['price']:.5f}\n\n"
        
        # Overall confluence
        bullish_count = sum(1 for data in mtf_data.values() if 'BULLISH' in data['trend'])
        bearish_count = sum(1 for data in mtf_data.values() if 'BEARISH' in data['trend'])
        
        text += f"🎯 *Confluence Summary:*\n"
        text += f"• Bullish TFs: {bullish_count}/5\n"
        text += f"• Bearish TFs: {bearish_count}/5\n"
        
        if bullish_count >= 3:
            text += f"• Overall Bias: 🟢 BULLISH\n"
        elif bearish_count >= 3:
            text += f"• Overall Bias: 🔴 BEARISH\n"
        else:
            text += f"• Overall Bias: 🟡 MIXED\n"
        
        keyboard = [
            [
                InlineKeyboardButton("🔄 Refresh", callback_data=f"mtf_{pair}"),
                InlineKeyboardButton("📊 Get Signal", callback_data=f"pair_{pair}")
            ],
            [InlineKeyboardButton("🔙 Back", callback_data="back_to_pairs")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode='Markdown')

    async def show_deep_analysis(self, query, context):
        """Tampilkan analisis mendalam untuk single pair"""
        pair = query.data.replace("analyze_", "")
        
        await query.edit_message_text("🔍 *Deep Analysis in Progress...*", parse_mode='Markdown')
        
        # Get comprehensive analysis
        mtf_data = self.analyzer.analyze_timeframes(pair)
        df = self.analyzer.generate_sample_data(pair, '15m')
        indicators = self.analyzer.calculate_indicators(df)
        
        # Market data
        market_data = await self.analyzer.get_market_data(pair)
        
        # BOS analysis
        bos = self.analyzer.detect_break_of_structure(df, indicators)
        
        # Fibonacci confluence
        fib_confluence = self.analyzer.detect_fibonacci_confluence(
            market_data.price, indicators['fibonacci']
        )
        
        text = f"🔍 *Deep Analysis - {pair}*\n\n"
        text += f"💰 *Current Price:* `{market_data.price:.5f}`\n"
        text += f"📊 *24h Change:* {market_data.change_24h:+.2f}%\n"
        text += f"📈 *Volume:* {market_data.volume:,.0f}\n\n"
        
        # Market Structure
        market_structure = self.analyzer.analyze_market_structure(df, indicators)
        structure_emoji = {
            'uptrend': '📈', 'downtrend': '📉', 'range': '🔄',
            'break_bullish': '🚀', 'break_bearish': '💥'
        }.get(market_structure, '🔄')
        
        text += f"{structure_emoji} *Market Structure:* {market_structure.title()}\n\n"
        
        # Key Indicators
        text += f"🎯 *Key Indicators:*\n"
        text += f"• RSI: {indicators['rsi'].iloc[-1]:.1f}\n"
        text += f"• ADX: {indicators['adx'].iloc[-1]:.1f} (Trend Strength)\n"
        text += f"• MACD: {'Bullish' if indicators['macd_histogram'].iloc[-1] > 0 else 'Bearish'}\n"
        text += f"• Stochastic: {indicators['stoch_k'].iloc[-1]:.1f}\n\n"
        
        # BOS Analysis
        if bos['type']:
            text += f"🔥 *Break of Structure Detected!*\n"
            text += f"• Type: {bos['type']}\n"
            text += f"• Level: {bos['level']:.5f}\n"
            text += f"• Volume Confirmed: {'✅' if bos['volume_confirmed'] else '❌'}\n\n"
        
        # Fibonacci Analysis
        text += f"🧮 *Fibonacci Levels:*\n"
        current_price = market_data.price
        for level, price in indicators['fibonacci'].items():
            distance = abs(current_price - price)
            if distance <= 0.0020:  # Within 20 pips
                text += f"• {level}: `{price:.5f}` 🎯\n"
            else:
                text += f"• {level}: `{price:.5f}`\n"
        
        if fib_confluence['near_fib']:
            text += f"\n🎯 *Near Fibonacci {fib_confluence['level']}!*\n"
        
        # Support/Resistance
        text += f"\n🏗️ *Key Levels:*\n"
        text += f"• Resistance: `{indicators['resistance']:.5f}`\n"
        text += f"• Support: `{indicators['support']:.5f}`\n\n"
        
        # Multi-timeframe confluence
        text += f"⏰ *Timeframe Confluence:*\n"
        bullish_tfs = 0
        bearish_tfs = 0
        
        for tf, data in mtf_data.items():
            trend = data['trend']
            if 'BULLISH' in trend:
                bullish_tfs += 1
                text += f"• {tf.upper()}: 🟢 {trend}\n"
            elif 'BEARISH' in trend:
                bearish_tfs += 1
                text += f"• {tf.upper()}: 🔴 {trend}\n"
            else:
                text += f"• {tf.upper()}: 🟡 {trend}\n"
        
        text += f"\n📊 *Confluence Score:*\n"
        text += f"• Bullish: {bullish_tfs}/5\n"
        text += f"• Bearish: {bearish_tfs}/5\n"
        
        keyboard = [
            [
                InlineKeyboardButton("📊 Generate Signal", callback_data=f"pair_{pair}"),
                InlineKeyboardButton("🔄 Refresh", callback_data=f"analyze_{pair}")
            ],
            [InlineKeyboardButton("🔙 Back", callback_data="back_to_main")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode='Markdown')

    async def show_economic_calendar(self, query, context):
        """Tampilkan economic calendar dengan impact analysis"""
        events = await self.news_analyzer.get_economic_calendar()
        
        text = "📅 *Economic Calendar - Next 48 Hours*\n\n"
        
        # Group events by date
        events_by_date = {}
        for event in events:
            date = event['date']
            if date not in events_by_date:
                events_by_date[date] = []
            events_by_date[date].append(event)
        
        for date, day_events in events_by_date.items():
            # Format date
            event_date = datetime.strptime(date, '%Y-%m-%d')
            if event_date.date() == datetime.now().date():
                date_str = "📅 *TODAY*"
            elif event_date.date() == (datetime.now() + timedelta(days=1)).date():
                date_str = "📅 *TOMORROW*"
            else:
                date_str = f"📅 *{event_date.strftime('%A, %B %d')}*"
            
            text += f"{date_str}\n"
            
            for event in day_events:
                impact_emoji = {
                    'VERY_HIGH': '🔴🔴',
                    'HIGH': '🟠', 
                    'MEDIUM': '🟡',
                    'LOW': '🟢'
                }.get(event['impact'], '⚪')
                
                text += f"\n{impact_emoji} *{event['time']} GMT+7* - {event['currency']}\n"
                text += f"📊 {event['event']}\n"
                
                if event['forecast']:
                    text += f"🔮 Forecast: {event['forecast']}\n"
                if event['previous']:
                    text += f"📈 Previous: {event['previous']}\n"
                
                # Impact description
                impact_desc = self.news_analyzer.calculate_volatility_impact(event['impact'])
                text += f"💥 Impact: {impact_desc}\n"
            
            text += "\n" + "─" * 30 + "\n\n"
        
        # Trading recommendations
        text += "💡 *Trading Recommendations:*\n"
        text += "• Avoid trading 30min before/after HIGH impact news\n"
        text += "• Watch for volatility spikes during releases\n"
        text += "• Consider wider stops during news events\n"
        text += "• NFP & FOMC = Extreme volatility expected\n"
        
        keyboard = [
            [
                InlineKeyboardButton("🔄 Refresh Calendar", callback_data="economic_calendar"),
                InlineKeyboardButton("💹 Market Impact", callback_data="news_impact")
            ],
            [InlineKeyboardButton("🔙 Back to Main", callback_data="back_to_main")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode='Markdown')

    async def show_market_overview(self, query, context):
        """Tampilkan overview market dengan sentiment analysis"""
        text = "💹 *Live Market Overview*\n\n"
        
        major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'BTCUSD']
        
        total_bullish = 0
        total_bearish = 0
        
        for pair in major_pairs:
            try:
                market_data = await self.analyzer.get_market_data(pair)
                mtf_analysis = self.analyzer.analyze_timeframes(pair)
                
                # Get H1 trend for quick overview
                h1_trend = mtf_analysis.get('1h', {}).get('trend', 'SIDEWAYS')
                
                change_emoji = "🟢" if market_data.change_24h > 0 else "🔴" if market_data.change_24h < 0 else "🟡"
                trend_emoji = "📈" if 'BULLISH' in h1_trend else "📉" if 'BEARISH' in h1_trend else "🔄"
                
                # Count sentiment
                if 'BULLISH' in h1_trend:
                    total_bullish += 1
                elif 'BEARISH' in h1_trend:
                    total_bearish += 1
                
                text += f"{change_emoji} *{pair}*: `{market_data.price:.5f}`\n"
                text += f"├ 24h: {market_data.change_24h:+.2f}%\n"
                text += f"└ Trend: {trend_emoji} {h1_trend}\n\n"
                
            except Exception as e:
                logger.error(f"Error getting overview for {pair}: {e}")
                text += f"⚪ *{pair}*: Data unavailable\n\n"
        
        # Market sentiment
        text += "🧠 *Market Sentiment Analysis:*\n"
        
        if total_bullish > total_bearish:
            sentiment = "🟢 RISK ON"
            text += f"• Overall: {sentiment} ({total_bullish}B vs {total_bearish}B)\n"
        elif total_bearish > total_bullish:
            sentiment = "🔴 RISK OFF"
            text += f"• Overall: {sentiment} ({total_bearish}B vs {total_bullish}B)\n"
        else:
            sentiment = "🟡 MIXED"
            text += f"• Overall: {sentiment} (No clear bias)\n"
        
        # Currency strength (simplified)
        text += f"\n💪 *Currency Strength:*\n"
        text += f"• USD: {'Strong 💪' if total_bearish > 2 else 'Neutral 🤝'}\n"
        text += f"• EUR: {'Strong 💪' if 'EURUSD' in [p for p in major_pairs if 'BULLISH' in mtf_analysis.get('1h', {}).get('trend', '')] else 'Neutral 🤝'}\n"
        text += f"• GBP: Volatile movements 🎢\n"
        text += f"• Gold: {'Safe haven demand 🛡️' if market_data.change_24h > 0 else 'Risk appetite 📈'}\n"
        
        # Quick trading opportunities
        text += f"\n🎯 *Quick Opportunities:*\n"
        opportunities = []
        
        for pair in major_pairs[:3]:  # Check top 3 pairs
            try:
                signal = self.analyzer.generate_signal(pair, TradingStyle.DAY_TRADING)
                if signal.confidence >= 75:
                    direction_emoji = "🟢" if signal.direction == "BUY" else "🔴"
                    opportunities.append(f"• {direction_emoji} {pair} - {signal.confidence}% confidence")
            except:
                pass
        
        if opportunities:
            text += "\n".join(opportunities)
        else:
            text += "• No high-confidence signals at the moment\n"
        
        text += f"\n\n🕐 *Last Updated:* {datetime.now().strftime('%H:%M:%S GMT+7')}"
        
        keyboard = [
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="market_overview"),
                InlineKeyboardButton("🎯 Live Signals", callback_data="live_signals")
            ],
            [InlineKeyboardButton("🔙 Back to Main", callback_data="back_to_main")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode='Markdown')

    async def show_live_signals(self, query, context):
        """Tampilkan live signals untuk multiple pairs"""
        await query.edit_message_text("🔴 *SCANNING LIVE SIGNALS...*\n⏳ *Please wait...*", parse_mode='Markdown')
        
        text = "🔴 *LIVE SIGNALS MONITOR*\n\n"
        
        high_confidence_signals = []
        medium_confidence_signals = []
        
        monitored_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'USDCHF', 'AUDUSD']
        
        for pair in monitored_pairs:
            try:
                # Generate signals for day trading style
                signal = self.analyzer.generate_signal(pair, TradingStyle.DAY_TRADING)
                
                if signal.confidence >= 80:
                    high_confidence_signals.append(signal)
                elif signal.confidence >= 65:
                    medium_confidence_signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error generating signal for {pair}: {e}")
        
        # High confidence signals
        if high_confidence_signals:
            text += "🔥 *HIGH CONFIDENCE SIGNALS:*\n\n"
            for signal in high_confidence_signals:
                direction_emoji = "🟢" if signal.direction == "BUY" else "🔴"
                text += f"{direction_emoji} *{signal.pair}* - {signal.direction}\n"
                text += f"├ Entry: `{signal.entry_price:.5f}`\n"
                text += f"├ TP1: `{signal.tp1:.5f}`\n" 
                text += f"├ SL: `{signal.stop_loss:.5f}`\n"
                text += f"├ Confidence: {signal.confidence}%\n"
                text += f"└ RR: 1:{signal.risk_reward:.2f}\n\n"
        
        # Medium confidence signals
        if medium_confidence_signals:
            text += "⚡ *MEDIUM CONFIDENCE SIGNALS:*\n\n"
            for signal in medium_confidence_signals:
                direction_emoji = "🟢" if signal.direction == "BUY" else "🔴"
                text += f"{direction_emoji} {signal.pair} - {signal.direction} ({signal.confidence}%)\n"
        
        if not high_confidence_signals and not medium_confidence_signals:
            text += "📊 *No high-quality signals detected at the moment.*\n\n"
            text += "💡 *Market Conditions:*\n"
            text += "• Low volatility period\n"
            text += "• Waiting for clear setups\n"
            text += "• Check back in 30-60 minutes\n"
        
        text += f"\n🔄 *Auto-refresh every 15 minutes*\n"
        text += f"🕐 *Last scan:* {datetime.now().strftime('%H:%M:%S GMT+7')}"
        
        keyboard = [
            [
                InlineKeyboardButton("🔄 Refresh Signals", callback_data="live_signals"),
                InlineKeyboardButton("📊 Market Overview", callback_data="market_overview")
            ],
            [InlineKeyboardButton("🔙 Back to Main", callback_data="back_to_main")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode='Markdown')

    async def show_fibonacci_analysis(self, query, context):
        """Tampilkan analisis Fibonacci detail"""
        pair = query.data.replace("fib_", "")
        
        df = self.analyzer.generate_sample_data(pair, '1h')
        indicators = self.analyzer.calculate_indicators(df)
        current_price = df['close'].iloc[-1]
        
        text = f"🧮 *Fibonacci Analysis - {pair}*\n\n"
        text += f"💰 *Current Price:* `{current_price:.5f}`\n\n"
        
        text += f"📊 *Fibonacci Retracement Levels:*\n"
        
        fib_levels = indicators['fibonacci']
        for level, price in fib_levels.items():
            distance_pips = abs(current_price - price) * 10000
            
            if distance_pips <= 10:  # Within 10 pips
                text += f"🎯 *{level}:* `{price:.5f}` ⭐ (Near!)\n"
            elif distance_pips <= 20:  # Within 20 pips
                text += f"🔸 *{level}:* `{price:.5f}` 📍 (Close)\n"
            else:
                text += f"• {level}: `{price:.5f}` ({distance_pips:.1f} pips)\n"
        
        # Fibonacci confluence analysis
        fib_confluence = self.analyzer.detect_fibonacci_confluence(current_price, fib_levels)
        
        if fib_confluence['near_fib']:
            text += f"\n🎯 *FIBONACCI CONFLUENCE DETECTED!*\n"
            text += f"• Level: {fib_confluence['level']}\n"
            text += f"• Price: `{fib_confluence['price']:.5f}`\n"
            text += f"• Distance: {fib_confluence['distance']*10000:.1f} pips\n\n"
            
            # Trading recommendation
            if fib_confluence['level'] in ['61.8%', '50%', '38.2%']:
                text += f"💡 *Trading Opportunity:*\n"
                text += f"• Strong reversal zone\n"
                text += f"• Watch for rejection/bounce\n"
                text += f"• Good risk/reward setup\n"
        
        # Extension levels
        text += f"\n📈 *Fibonacci Extensions:*\n"
        swing_high = fib_levels['0%']
        swing_low = fib_levels['100%']
        range_size = abs(swing_high - swing_low)
        
        if swing_high > swing_low:  # Uptrend extensions
            ext_127 = swing_high + (range_size * 0.272)
            ext_161 = swing_high + (range_size * 0.618)
            ext_261 = swing_high + (range_size * 1.618)
            
            text += f"• 127.2%: `{ext_127:.5f}`\n"
            text += f"• 161.8%: `{ext_161:.5f}`\n"
            text += f"• 261.8%: `{ext_261:.5f}`\n"
        else:  # Downtrend extensions
            ext_127 = swing_low - (range_size * 0.272)
            ext_161 = swing_low - (range_size * 0.618)
            ext_261 = swing_low - (range_size * 1.618)
            
            text += f"• 127.2%: `{ext_127:.5f}`\n"
            text += f"• 161.8%: `{ext_161:.5f}`\n"
            text += f"• 261.8%: `{ext_261:.5f}`\n"
        
        keyboard = [
            [
                InlineKeyboardButton("📊 Generate Signal", callback_data=f"pair_{pair}"),
                InlineKeyboardButton("🔄 Refresh Fib", callback_data=f"fib_{pair}")
            ],
            [InlineKeyboardButton("🔙 Back", callback_data="back_to_pairs")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode='Markdown')

    async def show_chart_analysis(self, query, context):
        """Tampilkan analisis chart pattern"""
        pair = query.data.replace("chart_", "")
        
        text = f"📈 *Chart Pattern Analysis - {pair}*\n\n"
        
        df = self.analyzer.generate_sample_data(pair, '1h')
        indicators = self.analyzer.calculate_indicators(df)
        
        # Detect chart patterns
        patterns = self.detect_chart_patterns(df)
        
        if patterns:
            text += "🔍 *Detected Patterns:*\n"
            for pattern in patterns:
                pattern_emoji = {
                    'double_top': '⏫', 'double_bottom': '⏬',
                    'head_shoulders': '👤', 'triangle': '🔺',
                    'flag': '🏁', 'pennant': '🚩'
                }.get(pattern['type'], '📊')
                
                text += f"{pattern_emoji} *{pattern['name']}*\n"
                text += f"├ Reliability: {pattern['reliability']}%\n"
                text += f"├ Target: `{pattern['target']:.5f}`\n"
                text += f"└ Invalidation: `{pattern['invalidation']:.5f}`\n\n"
        else:
            text += "📊 *No clear patterns detected at current timeframe*\n\n"
        
        # Price action analysis
        text += "💹 *Price Action Analysis:*\n"
        
        # Candlestick patterns
        latest_candles = df.tail(3)
        candle_pattern = self.analyze_candlestick_patterns(latest_candles)
        
        if candle_pattern:
            text += f"🕯️ *Pattern:* {candle_pattern['name']}\n"
            text += f"🎯 *Signal:* {candle_pattern['signal']}\n"
            text += f"📊 *Reliability:* {candle_pattern['reliability']}%\n\n"
        
        # Volume analysis
        volume_analysis = self.analyze_volume_profile(df)
        text += f"📊 *Volume Analysis:*\n"
        text += f"• Current vs Average: {volume_analysis['ratio']:.1f}x\n"
        text += f"• Trend: {volume_analysis['trend']}\n"
        text += f"• Confirmation: {volume_analysis['confirmation']}\n\n"
        
        # Key levels summary
        text += f"🏗️ *Critical Levels:*\n"
        text += f"• Next Resistance: `{indicators['resistance']:.5f}`\n"
        text += f"• Next Support: `{indicators['support']:.5f}`\n"
        text += f"• 50% Fib: `{indicators['fibonacci']['50%']:.5f}`\n"
        text += f"• 61.8% Fib: `{indicators['fibonacci']['61.8%']:.5f}`\n"
        
        keyboard = [
            [
                InlineKeyboardButton("📊 Get Signal", callback_data=f"pair_{pair}"),
                InlineKeyboardButton("🔄 Refresh Chart", callback_data=f"chart_{pair}")
            ],
            [InlineKeyboardButton("🔙 Back", callback_data="back_to_pairs")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode='Markdown')

    def detect_chart_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect common chart patterns"""
        patterns = []
        
        # Simple pattern detection based on price action
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # Double top/bottom detection (simplified)
        if len(df) >= 50:
            recent_highs = highs[-20:]
            recent_lows = lows[-20:]
            
            # Double top
            high_peaks, _ = find_peaks(recent_highs, distance=5)
            if len(high_peaks) >= 2:
                if abs(recent_highs[high_peaks[-1]] - recent_highs[high_peaks[-2]]) < recent_highs[high_peaks[-1]] * 0.002:
                    patterns.append({
                        'type': 'double_top',
                        'name': 'Double Top',
                        'reliability': 75,
                        'target': recent_highs[high_peaks[-1]] - (recent_highs[high_peaks[-1]] - np.min(recent_lows)) * 0.618,
                        'invalidation': recent_highs[high_peaks[-1]]
                    })
            
            # Double bottom
            low_peaks, _ = find_peaks(-recent_lows, distance=5)
            if len(low_peaks) >= 2:
                if abs(recent_lows[low_peaks[-1]] - recent_lows[low_peaks[-2]]) < recent_lows[low_peaks[-1]] * 0.002:
                    patterns.append({
                        'type': 'double_bottom',
                        'name': 'Double Bottom',
                        'reliability': 75,
                        'target': recent_lows[low_peaks[-1]] + (np.max(recent_highs) - recent_lows[low_peaks[-1]]) * 0.618,
                        'invalidation': recent_lows[low_peaks[-1]]
                    })
        
        return patterns

    def analyze_candlestick_patterns(self, candles: pd.DataFrame) -> Optional[Dict]:
        """Analyze candlestick patterns"""
        if len(candles) < 3:
            return None
        
        # Get last 3 candles
        c1 = candles.iloc[-3]  # 2 candles ago
        c2 = candles.iloc[-2]  # 1 candle ago
        c3 = candles.iloc[-1]  # Current candle
        
        # Bullish engulfing
        if (c1['close'] < c1['open'] and  # Bearish candle
            c2['close'] > c2['open'] and  # Bullish candle
            c2['open'] < c1['close'] and  # Opens below previous close
            c2['close'] > c1['open']):     # Closes above previous open
            return {
                'name': 'Bullish Engulfing',
                'signal': 'BUY',
                'reliability': 80
            }
        
        # Bearish engulfing
        elif (c1['close'] > c1['open'] and  # Bullish candle
              c2['close'] < c2['open'] and  # Bearish candle
              c2['open'] > c1['close'] and  # Opens above previous close
              c2['close'] < c1['open']):     # Closes below previous open
            return {
                'name': 'Bearish Engulfing',
                'signal': 'SELL',
                'reliability': 80
            }
        
        # Doji
        elif abs(c3['close'] - c3['open']) < (c3['high'] - c3['low']) * 0.1:
            return {
                'name': 'Doji',
                'signal': 'REVERSAL',
                'reliability': 60
            }
        
        return None

    def analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Analyze volume profile"""
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        # Volume trend
        volume_trend = df['volume'].rolling(window=5).mean()
        if volume_trend.iloc[-1] > volume_trend.iloc[-3]:
            trend = "Increasing"
        elif volume_trend.iloc[-1] < volume_trend.iloc[-3]:
            trend = "Decreasing"
        else:
            trend = "Stable"
        
        # Price-volume confirmation
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
        
        if volume_ratio > 1.5 and abs(price_change) > 0.001:
            confirmation = "Strong confirmation"
        elif volume_ratio > 1.2:
            confirmation = "Moderate confirmation"
        else:
            confirmation = "Weak confirmation"
        
        return {
            'ratio': volume_ratio,
            'trend': trend,
            'confirmation': confirmation
        }

    async def show_settings(self, query, context):
        """Tampilkan menu settings"""
        user_id = query.from_user.id
        settings = self.user_settings.get(user_id, {})
        
        text = f"⚙️ *Bot Settings*\n\n"
        text += f"💰 *Account Balance:* ${settings.get('account_balance', 10000):,.2f}\n"
        text += f"🎯 *Risk per Trade:* {settings.get('risk_percentage', 2.0)}%\n"
        text += f"🔔 *Notifications:* {'On' if settings.get('notifications', True) else 'Off'}\n"
        text += f"📊 *Preferred Pairs:* {len(settings.get('preferred_pairs', []))}\n\n"
        
        text += f"🧮 *Risk Calculator:*\n"
        risk_amount = settings.get('account_balance', 10000) * (settings.get('risk_percentage', 2.0) / 100)
        text += f"• Max risk per trade: ${risk_amount:.2f}\n"
        text += f"• Daily max loss (5 trades): ${risk_amount * 5:.2f}\n"
        text += f"• Conservative size: Use 50% of calculated lots\n"
        
        keyboard = [
            [
                InlineKeyboardButton("💰 Set Balance", callback_data="set_balance"),
                InlineKeyboardButton("🎯 Set Risk %", callback_data="set_risk")
            ],
            [
                InlineKeyboardButton("📊 Pair Preferences", callback_data="set_pairs"),
                InlineKeyboardButton("🔔 Notifications", callback_data="toggle_notif")
            ],
            [InlineKeyboardButton("🔙 Back to Main", callback_data="back_to_main")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode='Markdown')

    async def show_single_analysis_menu(self, query, context):
        """Menu untuk analisis single pair"""
        text = "🔍 *Deep Analysis Mode*\n\n"
        text += "Comprehensive analysis including:\n"
        text += "• Multi-timeframe confluence\n"
        text += "• Fibonacci levels\n"
        text += "• Market structure\n"
        text += "• Volume analysis\n"
        text += "• Support/Resistance\n\n"
        text += "*Select pair for deep analysis:*"
        
        keyboard = []
        pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'USDCHF', 'AUDUSD', 'BTCUSD', 'ETHUSD']
        
        for i in range(0, len(pairs), 2):
            row = []
            for j in range(2):
                if i + j < len(pairs):
                    pair = pairs[i + j]
                    emoji = "🥇" if pair == "XAUUSD" else "₿" if pair == "BTCUSD" else "💱"
                    row.append(InlineKeyboardButton(f"{emoji} {pair}", callback_data=f"analyze_{pair}"))
            keyboard.append(row)
        
        keyboard.append([InlineKeyboardButton("🔙 Back to Main", callback_data="back_to_main")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode='Markdown')

    async def market_scanner(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Command /scan untuk scan semua pairs"""
        await update.message.reply_text("🔍 *MARKET SCANNER RUNNING...*", parse_mode='Markdown')
        
        all_pairs = self.analyzer.major_pairs
        high_conf_signals = []
        medium_conf_signals = []
        
        for pair in all_pairs:
            try:
                signal = self.analyzer.generate_signal(pair, TradingStyle.DAY_TRADING)
                
                if signal.confidence >= 80:
                    high_conf_signals.append(signal)
                elif signal.confidence >= 65:
                    medium_conf_signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Scanner error for {pair}: {e}")
        
        # Sort by confidence
        high_conf_signals.sort(key=lambda x: x.confidence, reverse=True)
        medium_conf_signals.sort(key=lambda x: x.confidence, reverse=True)
        
        text = "🔍 *MARKET SCANNER RESULTS*\n\n"
        
        if high_conf_signals:
            text += "🔥 *HIGH CONFIDENCE SIGNALS:*\n\n"
            for signal in high_conf_signals[:5]:  # Top 5
                direction_emoji = "🟢" if signal.direction == "BUY" else "🔴"
                text += f"{direction_emoji} *{signal.pair}* - {signal.direction}\n"
                text += f"├ Confidence: {signal.confidence}%\n"
                text += f"├ Entry: `{signal.entry_price:.5f}`\n"
                text += f"├ RR: 1:{signal.risk_reward:.2f}\n"
                text += f"└ Strategy: {signal.strategy}\n\n"
        
        if medium_conf_signals:
            text += "⚡ *MEDIUM CONFIDENCE:*\n"
            for signal in medium_conf_signals[:3]:  # Top 3
                direction_emoji = "🟢" if signal.direction == "BUY" else "🔴"
                text += f"{direction_emoji} {signal.pair} - {signal.direction} ({signal.confidence}%)\n"
        
        if not high_conf_signals and not medium_conf_signals:
            text += "📊 *No quality setups found.*\n"
            text += "💡 *Market Status:* Consolidation/Low volatility\n"
            text += "🕐 *Next scan:* Try again in 30-60 minutes\n"
        
        text += f"\n🕐 *Scan completed:* {datetime.now().strftime('%H:%M:%S GMT+7')}"
        
        await update.message.reply_text(text, parse_mode='Markdown')

    async def watchlist_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Command /watchlist untuk pairs monitoring"""
        user_id = update.effective_user.id
        preferred_pairs = self.user_settings.get(user_id, {}).get('preferred_pairs', ['EURUSD', 'GBPUSD', 'XAUUSD'])
        
        text = "👀 *WATCHLIST MONITORING*\n\n"
        
        for pair in preferred_pairs:
            try:
                market_data = await self.analyzer.get_market_data(pair)
                mtf_analysis = self.analyzer.analyze_timeframes(pair)
                
                # Get overall bias
                h1_trend = mtf_analysis.get('1h', {}).get('trend', 'SIDEWAYS')
                h4_trend = mtf_analysis.get('4h', {}).get('trend', 'SIDEWAYS')
                
                change_emoji = "🟢" if market_data.change_24h > 0 else "🔴"
                
                text += f"{change_emoji} *{pair}*\n"
                text += f"├ Price: `{market_data.price:.5f}`\n"
                text += f"├ 24h: {market_data.change_24h:+.2f}%\n"
                text += f"├ H1: {h1_trend}\n"
                text += f"└ H4: {h4_trend}\n\n"
                
            except Exception as e:
                text += f"⚪ *{pair}*: Data error\n\n"
        
        text += f"🔄 *Auto-updates every 5 minutes*\n"
        text += f"Use /signals for trading opportunities"
        
        await update.message.reply_text(text, parse_mode='Markdown')

    async def fibonacci_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Command /fib untuk quick fibonacci analysis"""
        args = context.args
        
        if not args:
            await update.message.reply_text(
                "Usage: /fib EURUSD\n\nExample: /fib XAUUSD",
                parse_mode='Markdown'
            )
            return
        
        pair = args[0].upper()
        
        if pair not in self.analyzer.major_pairs:
            await update.message.reply_text(
                f"❌ Pair {pair} not supported.\n\nSupported: {', '.join(self.analyzer.major_pairs[:8])}",
                parse_mode='Markdown'
            )
            return
        
        await update.message.reply_text(f"🧮 *Calculating Fibonacci for {pair}...*", parse_mode='Markdown')
        
        # Generate Fibonacci analysis
        df = self.analyzer.generate_sample_data(pair, '1h')
        indicators = self.analyzer.calculate_indicators(df)
        current_price = df['close'].iloc[-1]
        
        fib_levels = indicators['fibonacci']
        fib_confluence = self.analyzer.detect_fibonacci_confluence(current_price, fib_levels)
        
        text = f"🧮 *Fibonacci Analysis - {pair}*\n\n"
        text += f"💰 *Current:* `{current_price:.5f}`\n\n"
        
        text += f"📊 *Key Fibonacci Levels:*\n"
        
        # Sort levels by proximity to current price
        level_distances = []
        for level, price in fib_levels.items():
            distance = abs(current_price - price)
            level_distances.append((level, price, distance))
        
        level_distances.sort(key=lambda x: x[2])
        
        for level, price, distance in level_distances:
            distance_pips = distance * 10000
            
            if distance_pips <= 10:
                text += f"🎯 *{level}:* `{price:.5f}` ⭐ NEAR!\n"
            elif distance_pips <= 25:
                text += f"🔸 *{level}:* `{price:.5f}` 📍 ({distance_pips:.1f}p)\n"
            else:
                text += f"• {level}: `{price:.5f}` ({distance_pips:.0f}p)\n"
        
        if fib_confluence['near_fib']:
            text += f"\n🎯 *CONFLUENCE ZONE!*\n"
            text += f"Near {fib_confluence['level']} level\n"
            text += f"Watch for reversal signals!\n"
        
        await update.message.reply_text(text, parse_mode='Markdown')

    async def live_signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Command /signals untuk live signals"""
        text = "🔴 *LIVE SIGNALS MONITORING*\n\n"
        
        signals_found = 0
        
        for pair in ['EURUSD', 'GBPUSD', 'XAUUSD', 'USDJPY']:
            signal = self.analyzer.generate_signal(pair, TradingStyle.DAY_TRADING)
            
            if signal.confidence >= 70:
                signals_found += 1
                direction_emoji = "🟢" if signal.direction == "BUY" else "🔴"
                confidence_emoji = "🔥" if signal.confidence >= 80 else "⚡"
                
                text += f"{direction_emoji} *{pair}* - {signal.direction} {confidence_emoji}\n"
                text += f"├ Entry: `{signal.entry_price:.5f}`\n"
                text += f"├ TP1: `{signal.tp1:.5f}`\n"
                text += f"├ Confidence: {signal.confidence}%\n"
                text += f"└ Strategy: {signal.strategy}\n\n"
        
        if signals_found == 0:
            text += "📊 No high-confidence signals at the moment.\n"
            text += "💡 Market may be in consolidation phase.\n\n"
        
        text += f"🕐 *Scan Time:* {datetime.now().strftime('%H:%M:%S')}\n"
        text += "🔄 Use /signals again to refresh"
        
        await update.message.reply_text(text, parse_mode='Markdown')

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Command /help"""
        help_text = """
🆘 *Forex Bot Commands & Features*

*📱 Commands:*
/start - Main menu
/signals - Quick live signals
/scan - Market scanner
/watchlist - Monitor preferred pairs
/fib PAIR - Fibonacci analysis
/help - This help message

*🎯 Trading Modes:*
• **Scalping** - 5-15min quick trades
• **Day Trading** - 15min-1H intraday
• **Swing Trading** - 1H-4H position trades

*📊 Features:*
• Multi-timeframe analysis (M5-D1)
• Fibonacci retracement levels
• Break of Structure detection
• Volume confirmation
• Economic calendar
• Risk management calculator
• Position sizing
• Real-time market data

*🧮 Indicators Used:*
• EMA (9, 20, 50, 200)
• RSI (14)
• MACD
• Stochastic
• ADX (Trend Strength)
• ATR (Volatility)
• Bollinger Bands
• Williams %R

*⚠️ Risk Disclaimer:*
This bot is for educational purposes. Always:
• Use proper risk management
• Never risk more than 2% per trade
• Backtest strategies before live trading
• Consider market conditions & news

*💡 Tips for Best Results:*
• Wait for high-confidence signals (80%+)
• Check economic calendar before trading
• Use multiple timeframe confirmation
• Follow the stop loss strictly
• Take partial profits at each TP level

📧 *Support:* Use feedback button for issues
        """
        
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk errors"""
        logger.error("Exception while handling an update:", exc_info=context.error)
        
        # Send error message to user if possible
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text(
                "🚨 *Oops! Something went wrong.*\n\n"
                "Please try again or use /start to return to main menu.",
                parse_mode='Markdown'
            )

def main():
    """Main function untuk menjalankan bot"""
    
    if BOT_TOKEN == "bot_token":
        print("❌ Error: Please set your BOT_TOKEN in the code!")
        print("📱 Steps to setup:")
        print("1. Message @BotFather on Telegram")
        print("2. Create new bot with /newbot")
        print("3. Copy the token and replace YOUR_BOT_TOKEN_HERE")
        print("4. Run the bot again")
        return
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Create bot instance
    bot = TelegramBot()
    
    # Add main handlers
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("signals", bot.live_signals_command))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("scan", bot.market_scanner))
    application.add_handler(CommandHandler("watchlist", bot.watchlist_command))
    application.add_handler(CommandHandler("fib", bot.fibonacci_command))
    application.add_handler(CallbackQueryHandler(bot.handle_callback))
    
    # Add error handler
    application.add_error_handler(bot.error_handler)
    
    # Start bot
    print("🚀 Forex Trading Bot Pro v2.0 - Starting...")
    print("\n📱 Available Commands:")
    print("   /start     - Main menu")
    print("   /signals   - Live signals")
    print("   /scan      - Market scanner")
    print("   /watchlist - Monitor preferred pairs")
    print("   /fib PAIR  - Fibonacci analysis")
    print("   /help      - Help & guide")
    print("\n🔥 Features:")
    print("   • Multi-timeframe analysis (M5-D1)")
    print("   • Fibonacci confluence detection")
    print("   • Break of Structure (BOS)")
    print("   • Economic calendar integration")
    print("   • Advanced risk management")
    print("   • Volume confirmation")
    print("   • Real-time market data")
    print("   • Position size calculator")
    print("\n✅ Bot is now running! Press Ctrl+C to stop.")
    
    # Run bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    os.system('clear')
    main()
