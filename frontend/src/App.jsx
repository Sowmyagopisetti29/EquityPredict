
import { useRef, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import './App.css';

const API_URL =
  import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// Detect currency based on stock exchange suffix
function getCurrency(stockSymbol) {
  const normalizedSymbol = (stockSymbol || '').trim().toUpperCase();

  if (
    normalizedSymbol.endsWith('.NS') ||
    normalizedSymbol.endsWith('.BO')
  ) {
    return {
      code: 'INR',
      symbol: '₹',
    };
  }

  return {
    code: 'USD',
    symbol: '$',
  };
}

// Format stock prices using the correct currency
function formatPrice(price, stockSymbol) {
  const currency = getCurrency(stockSymbol);
  const numericPrice = Number(price);

  if (!Number.isFinite(numericPrice)) {
    return currency.symbol + '0.00';
  }

  return currency.symbol + numericPrice.toFixed(2);
}

function getSignal(returnPercent) {
  if (returnPercent > 1) return 'BUY';
  if (returnPercent < -1) return 'SELL';
  return 'HOLD';
}

// Format a calendar date without shifting it across timezones
function formatStockDate(date, includeYear = false) {
  const datePart = String(date).split('T')[0];
  const [year, month, day] = datePart.split('-').map(Number);

  if (
    !year ||
    !month ||
    !day ||
    month < 1 ||
    month > 12 ||
    day < 1 ||
    day > 31
  ) {
    return String(date);
  }

  const calendarDate = new Date(
    Date.UTC(year, month - 1, day)
  );

  return calendarDate.toLocaleDateString('en-US', {
    month: 'short',
    day: '2-digit',
    ...(includeYear ? { year: 'numeric' } : {}),
    timeZone: 'UTC',
  });
}

function App() {
  // References for stock input fields
  const symbolInputRef = useRef(null);
  const multipleSymbolsInputRef = useRef(null);

  const [symbol, setSymbol] = useState('AAPL');
  const [analyzedSymbol, setAnalyzedSymbol] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [multipleSymbols, setMultipleSymbols] = useState(
    'AAPL, GOOGL, MSFT'
  );
  const [comparison, setComparison] = useState([]);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState('');

  const [evaluation, setEvaluation] = useState(null);

  // Predict a single stock
  const handlePredict = async () => {
    const enteredSymbol =
      symbolInputRef.current?.value ?? 'AAPL';

    // Remove accidental spaces and convert to uppercase
    const selectedSymbol = enteredSymbol
      .replace(/\s+/g, '')
      .toUpperCase();

    if (!selectedSymbol) {
      setError('Please enter a stock symbol.');
      return;
    }

    // Update state only after the user submits
    setSymbol(enteredSymbol);

    setLoading(true);
    setError('');
    setPrediction(null);
    setHistory([]);
    setEvaluation(null);
    setAnalyzedSymbol('');

    try {
      const predictionResponse = await fetch(
        `${API_URL}/predict/${encodeURIComponent(selectedSymbol)}`
      );

      const predictionData = await predictionResponse.json();

      if (!predictionResponse.ok || predictionData.error) {
        throw new Error(
          `We couldn't find stock data for ${selectedSymbol}. Please check the ticker symbol and try again.`
        );
      }

      setPrediction(predictionData);
      setEvaluation(predictionData.model_evaluation || null);
      setAnalyzedSymbol(selectedSymbol);

      const historyResponse = await fetch(
        `${API_URL}/stock/${encodeURIComponent(selectedSymbol)}`
      );

      const historyData = await historyResponse.json();

      if (!historyResponse.ok || historyData.error) {
        throw new Error(
          historyData.error || 'Unable to fetch stock history.'
        );
      }

      setHistory(historyData.history || []);
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.');
      setPrediction(null);
      setHistory([]);
      setEvaluation(null);
      setAnalyzedSymbol('');
    } finally {
      setLoading(false);
    }
  };

  // Compare multiple stocks
  const handleCompare = async () => {
    const enteredSymbols =
      multipleSymbolsInputRef.current?.value ??
      'AAPL, GOOGL, MSFT';

    // Update state after submission, not on every keystroke
    setMultipleSymbols(enteredSymbols);

    const symbols = enteredSymbols
      .split(',')
      .map((item) =>
        item.replace(/\s+/g, '').toUpperCase()
      )
      .filter(Boolean);

    if (symbols.length === 0) {
      setCompareError('Please enter at least one stock symbol.');
      return;
    }

    if (symbols.length > 5) {
      setCompareError('You can compare a maximum of 5 stocks.');
      return;
    }

    if (new Set(symbols).size !== symbols.length) {
      setCompareError('Please remove duplicate stock symbols.');
      return;
    }

    setCompareLoading(true);
    setCompareError('');
    setComparison([]);

    try {
      const response = await fetch(
        `${API_URL}/predict-multiple?symbols=${encodeURIComponent(
          symbols.join(',')
        )}`
      );

      const data = await response.json();

      if (!response.ok || data.error) {
        throw new Error(data.error || 'Unable to compare stocks.');
      }

      const sortedResults = (data.results || [])
        .filter((stock) => !stock.error)
        .sort(
          (a, b) =>
            b.predicted_return_percent -
            a.predicted_return_percent
        );

      setComparison(sortedResults);

      if (sortedResults.length === 0) {
        setCompareError(
          'No valid stock predictions were returned.'
        );
      }
    } catch (err) {
      setCompareError(
        err.message || 'Unable to compare stocks.'
      );
    } finally {
      setCompareLoading(false);
    }
  };

  // Download comparison results as CSV
  const downloadCSV = () => {
    if (comparison.length === 0) return;

    const headers = [
      'Rank',
      'Stock',
      'Currency',
      'Latest Price',
      'Predicted Return (%)',
      'Predicted Price',
      'Signal',
    ];

    const rows = comparison.map((stock, index) => [
      index + 1,
      stock.stock,
      getCurrency(stock.stock).code,
      Number(stock.latest_close).toFixed(2),
      Number(stock.predicted_return_percent).toFixed(3),
      Number(stock.predicted_next_close).toFixed(2),
      getSignal(stock.predicted_return_percent),
    ]);

    const csvContent = [headers, ...rows]
      .map((row) => row.join(','))
      .join('\n');

    const blob = new Blob([csvContent], {
      type: 'text/csv;charset=utf-8;',
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');

    link.href = url;
    link.download = 'EquityPredict_Report.csv';
    link.click();

    URL.revokeObjectURL(url);
  };

  const bestStock =
    comparison.length > 0 ? comparison[0] : null;

  const selectedCurrency = getCurrency(
    analyzedSymbol || symbol
  );

  // Calculate chart boundaries
  const chartPrices = history
    .map((item) => Number(item.Close))
    .filter((price) => Number.isFinite(price));

  const minPrice = chartPrices.length
    ? Math.min(...chartPrices)
    : 0;

  const maxPrice = chartPrices.length
    ? Math.max(...chartPrices)
    : 0;

  const chartMargin = Math.max(
    (maxPrice - minPrice) * 0.1,
    2
  );

  // About section styling
  const aboutStyle = {
    background: '#111b2e',
    border: '1px solid #26334a',
    borderRadius: '10px',
    padding: '16px 20px',
    marginBottom: '12px',
    color: '#cbd5e1',
  };

  const summaryStyle = {
    color: '#38bdf8',
    fontWeight: '600',
    cursor: 'pointer',
    fontSize: '16px',
  };

  return (
    <div className="app">

      {/* Navigation Bar */}
      <nav className="navbar">
        <h2>EquityPredict</h2>
        <p>AI-Powered Stock Market Insights</p>
      </nav>

      <main className="dashboard">

        {/* Welcome Section */}
        <section className="welcome">
          <h1>Stock Market Prediction Dashboard</h1>
          <p>
            Explore stock prices and generate machine learning-based
            predictions using XGBoost.
          </p>
        </section>

        {/* Single Stock Analysis */}
        <section className="search-section">
          <h2>Analyze a Stock</h2>

          <p>
            Enter a stock symbol to view its prediction and historical prices.
          </p>

          <div className="search-box">
            <input
              ref={symbolInputRef}
              type="text"
              defaultValue="AAPL"
              placeholder="Enter stock symbol (e.g. AAPL or RELIANCE.NS)"
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="off"
              spellCheck={false}
              inputMode="text"
            />

            <button onClick={handlePredict} disabled={loading}>
              {loading ? 'Analyzing...' : 'Predict Stock'}
            </button>
          </div>

          <p className="hint">
            Examples: AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA,
            RELIANCE.NS, TCS.NS
          </p>

          {error && <p className="error">{error}</p>}
        </section>

        {/* Prediction Results */}
        {prediction && (
          <section className="results">
            <h2>Prediction Results</h2>

            <div className="result-grid">

              <div className="result-card">
                <p>Stock Symbol</p>
                <h3>{analyzedSymbol}</h3>
              </div>

              <div className="result-card">
                <p>Latest Closing Price ({selectedCurrency.code})</p>
                <h3>
                  {formatPrice(
                    prediction.latest_close,
                    analyzedSymbol
                  )}
                </h3>
              </div>

              <div className="result-card">
                <p>Predicted Return</p>
                <h3
                  className={
                    prediction.predicted_return_percent >= 0
                      ? 'positive'
                      : 'negative'
                  }
                >
                  {Number(
                    prediction.predicted_return_percent
                  ).toFixed(3)}%
                </h3>
              </div>

              <div className="result-card">
                <p>
                  Predicted Next Closing Price ({selectedCurrency.code})
                </p>
                <h3>
                  {formatPrice(
                    prediction.predicted_next_close,
                    analyzedSymbol
                  )}
                </h3>
              </div>

              <div className="result-card">
                <p>Buy / Hold / Sell Signal</p>
                <h3>
                  {getSignal(
                    prediction.predicted_return_percent
                  )}
                </h3>
              </div>

            </div>
          </section>
        )}

        {/* Historical Stock Chart */}
        {history.length > 0 && (
          <section className="chart-section">
            <h2>
              Historical Stock Price ({selectedCurrency.code})
            </h2>

            <div className="chart-container">
              <ResponsiveContainer width="100%" height={350}>
                <LineChart
                  data={history}
                  margin={{
                    top: 15,
                    right: 20,
                    left: 10,
                    bottom: 5,
                  }}
                >

                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#26334a"
                  />

                  <XAxis
                    dataKey="Date"
                    stroke="#94a3b8"
                    tick={{ fontSize: 11 }}
                    minTickGap={25}
                    tickFormatter={(date) =>
                      formatStockDate(date)
                    }
                  />

                  <YAxis
                    stroke="#94a3b8"
                    domain={[
                      minPrice - chartMargin,
                      maxPrice + chartMargin,
                    ]}
                    tickFormatter={(value) =>
                      selectedCurrency.symbol +
                      value.toFixed(0)
                    }
                    tick={{ fontSize: 11 }}
                    width={65}
                  />

                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#111b2e',
                      border: '1px solid #26334a',
                      color: '#f1f5f9',
                    }}
                    labelFormatter={(date) =>
                      formatStockDate(date, true)
                    }
                    formatter={(value) => [
                      formatPrice(value, analyzedSymbol),
                      'Closing Price',
                    ]}
                  />

                  <Line
                    type="monotone"
                    dataKey="Close"
                    stroke="#38bdf8"
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 5 }}
                  />

                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>
        )}

        {/* Multiple Stock Comparison */}
        <section className="multiple-section">
          <h2>Compare Multiple Stocks</h2>

          <p>
            Compare up to 5 stocks and identify the highest predicted return.
          </p>

          <div className="search-box">
            <input
              ref={multipleSymbolsInputRef}
              type="text"
              defaultValue="AAPL, GOOGL, MSFT"
              placeholder="Example: AAPL, GOOGL, RELIANCE.NS"
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="off"
              spellCheck={false}
              inputMode="text"
            />

            <button
              onClick={handleCompare}
              disabled={compareLoading}
            >
              {compareLoading ? 'Comparing...' : 'Compare Stocks'}
            </button>
          </div>

          {compareError && (
            <p className="error">{compareError}</p>
          )}

          {bestStock && (
            <div className="best-stock-card">
              <h3>Top Ranked Stock: {bestStock.stock}</h3>

              <p>
                Predicted Return:{' '}
                <strong>
                  {Number(
                    bestStock.predicted_return_percent
                  ).toFixed(3)}%
                </strong>
              </p>

              <p>
                Predicted Price:{' '}
                {formatPrice(
                  bestStock.predicted_next_close,
                  bestStock.stock
                )}
              </p>
            </div>
          )}

          {comparison.length > 0 && (
            <>
              <div className="table-container">
                <table className="stock-table">

                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Stock</th>
                      <th>Latest Price</th>
                      <th>Predicted Return</th>
                      <th>Predicted Price</th>
                      <th>Signal</th>
                    </tr>
                  </thead>

                  <tbody>
                    {comparison.map((stock, index) => (
                      <tr key={stock.stock}>
                        <td>#{index + 1}</td>
                        <td>{stock.stock}</td>
                        <td>
                          {formatPrice(
                            stock.latest_close,
                            stock.stock
                          )}
                        </td>
                        <td>
                          {Number(
                            stock.predicted_return_percent
                          ).toFixed(3)}%
                        </td>
                        <td>
                          {formatPrice(
                            stock.predicted_next_close,
                            stock.stock
                          )}
                        </td>
                        <td>
                          {getSignal(
                            stock.predicted_return_percent
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>

                </table>
              </div>

              <button
                onClick={downloadCSV}
                className="download-button"
                style={{
                  marginTop: '18px',
                  padding: '11px 18px',
                  background: '#0284c7',
                  color: 'white',
                  border: 'none',
                  borderRadius: '7px',
                  cursor: 'pointer',
                  fontWeight: '600',
                }}
              >
                Download CSV Report
              </button>
            </>
          )}
        </section>

        {/* Model Evaluation */}
        {evaluation && (
          <section className="results">
            <h2>Model Performance</h2>

            <div className="result-grid">

              <div className="result-card">
                <p>Mean Absolute Error (MAE %)</p>
                <h3>
                  {Number(evaluation.mae_percent).toFixed(4)}%
                </h3>
              </div>

              <div className="result-card">
                <p>Root Mean Squared Error (RMSE %)</p>
                <h3>
                  {Number(evaluation.rmse_percent).toFixed(4)}%
                </h3>
              </div>

              <div className="result-card">
                <p>R² Score</p>
                <h3>
                  {Number(evaluation.r2_score).toFixed(4)}
                </h3>
              </div>

              <div className="result-card">
                <p>Training Samples</p>
                <h3>{evaluation.training_samples}</h3>
              </div>

              <div className="result-card">
                <p>Testing Samples</p>
                <h3>{evaluation.testing_samples}</h3>
              </div>

            </div>
          </section>
        )}

        {/* About EquityPredict */}
        <section
          style={{
            marginTop: '35px',
            marginBottom: '35px',
          }}
        >

          <h2
            style={{
              marginBottom: '8px',
              color: '#f1f5f9',
            }}
          >
            About EquityPredict
          </h2>

          <p
            style={{
              color: '#94a3b8',
              marginBottom: '18px',
            }}
          >
            Learn more about the project and its technology.
          </p>

          <details style={aboutStyle}>
            <summary style={summaryStyle}>
              Project Overview
            </summary>

            <p style={{ marginTop: '12px', lineHeight: '1.7' }}>
              EquityPredict is a stock market analysis and prediction
              application that uses historical stock data and machine
              learning to estimate the next closing price and predicted return.
            </p>
          </details>

          <details style={aboutStyle}>
            <summary style={summaryStyle}>
              How EquityPredict Works
            </summary>

            <p style={{ marginTop: '12px', lineHeight: '1.8' }}>
              1. Enter a stock symbol.
              <br />
              2. Fetch historical stock data.
              <br />
              3. Calculate technical indicators.
              <br />
              4. Train and use the XGBoost model.
              <br />
              5. Generate the predicted return and closing price.
              <br />
              6. Display results, charts and comparison rankings.
            </p>
          </details>

          <details style={aboutStyle}>
            <summary style={summaryStyle}>
              Technologies and Indicators
            </summary>

            <p style={{ marginTop: '12px', lineHeight: '1.8' }}>
              <strong>Technologies:</strong> Python, React, FastAPI,
              XGBoost, Pandas, NumPy, Scikit-learn, Recharts and yfinance.
              <br />

              <strong>Indicators:</strong> Daily Return, Intraday Return,
              Moving Averages, High-Low Range and RSI.
            </p>
          </details>

          <details style={aboutStyle}>
            <summary style={summaryStyle}>
              Project Architecture
            </summary>

            <p style={{ marginTop: '12px', lineHeight: '1.8' }}>
              <strong>Frontend:</strong> React dashboard
              <br />
              ↓
              <br />
              <strong>Backend:</strong> FastAPI REST API
              <br />
              ↓
              <br />
              <strong>Data Processing:</strong> Python, Pandas and NumPy
              <br />
              ↓
              <br />
              <strong>Machine Learning:</strong> XGBoost prediction model
              <br />
              ↓
              <br />
              <strong>Output:</strong> Predictions, charts and stock rankings
            </p>
          </details>

        </section>

      </main>

      {/* Footer */}
      <footer>
        <p>
          EquityPredict | Educational Project | Predictions are estimates,
          not financial advice.
        </p>
      </footer>

    </div>
  );
}

export default App;
