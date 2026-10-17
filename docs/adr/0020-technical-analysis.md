# ADR 0020: Technical analysis: a full indicator toolkit with plain readings

- Status: Accepted
- Date: 2026-10-17

## Context

Until now the analyst saw raw daily bars and a one-line summary, and the trade score used
only a 20-day average, a 5-day move, and ATR. The operator asked for the bot to "know
everything about technical knowledge". A language model asked to compute RSI or MACD from
a table of numbers in its head gets them wrong often. It is also inconsistent: the same
bars can give different answers on different runs.

## Decisions

### 1. Indicators are computed in code, not by the model
`analysis/indicators.py` is a pure-Python library with no new dependencies. It covers:
- moving averages: SMA and EMA;
- momentum: RSI with Wilder smoothing, MACD, stochastic, Williams %R, CCI, rate of change,
  and money flow index;
- volatility: true range, ATR, Bollinger, Keltner, and Donchian channels;
- volume: OBV, VWAP, and Chaikin money flow;
- trend strength: ADX with +DI and −DI;
- levels: classic pivot points and swing supports and resistances;
- candlestick patterns: doji, hammer, shooting star, marubozu, engulfing, inside bar, and
  morning and evening star.

Every function returns a series aligned with its input, with `None` during warm-up. Tests
check each one against values worked out by hand from the textbook definition, plus
properties over random markets: oscillators stay bounded, bands stay ordered, and lengths
stay aligned.

### 2. A report with readings, not just numbers
`analysis/technicals.py` turns a year of daily bars, today's 5-minute bars, and SPY/QQQ
into a `TechnicalReport` with these sections: trend, momentum, volatility, volume, levels,
patterns, intraday (VWAP, the 30-minute opening range, and the 5-minute EMA trend), and
relative strength.

Each signal gets a fixed reading (bullish, bearish, or neutral) with a short note, such as
"RSI 72: overbought, pullback risk" or "closed above the 20-day high on heavy volume". The
overall bias is "bullish" or "bearish" only when one side leads the other by 3 readings or
more; otherwise it is "mixed". The same bars always give the same report.

### 3. One new read-only research tool
`technical_analysis(symbol)` returns the report and counts as one budgeted call. It fetches
daily bars for the stock and its benchmarks in one request, plus today's 5-minute bars. If
intraday data fails, the report still comes back with a warning. The critic's evidence
check takes only the report's live price as an observed price: indicator levels are not
quotes. The analyst prompt (`analyst-2026-10-17.4`) tells the analyst to cite readings
instead of recomputing them. It also lists the agreements to favor and the warnings to
respect, and treats a mixed bias as a reason for `no_trade`.

### 4. A seventh part in the trade score
"Technicals" (weight 20) averages five checks. Each check scores 0, ½, or 1:
1. RSI zone: 50–70 is healthy; 80 and above, or below 40, scores 0.
2. MACD: above zero, and above its signal.
3. ADX direction: buyers lead, and ADX is 20 or more.
4. Bollinger position: the upper half of the bands.
5. On-balance volume: rising over 20 days.

Checks the history cannot support are left out. With no checks at all, the part is neutral
(½). There is no new hard fail; the existing extension rule already rejects overextended
prices. The risk engine now fetches 60 daily bars so that every check can run.

### 5. A command for people
`trading-agent technicals AAPL` (`--json`) prints the same report the AI sees. It costs
nothing: market data only, no LLM, no orders.

## Consequences

- The analyst gets correct, consistent indicator values, and the critic can check its
  claims against them.
- Adding a score part changes totals. A clean 30-bar uptrend scored 88.6 and now scores
  86.3, because RSI 100 counts as overbought. Real stocks usually show pullbacks.
- Indicators describe the past. They make research more thorough but do not predict
  prices. The go/no-go evaluation (ADR 0012), not this ADR, judges whether results improve.
