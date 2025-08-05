# Yearly Statistics 
SELECT 
    YEAR(TO_DATE(`Date`)) AS Year,
    ROUND(AVG(Close), 2) AS AvgClose,
    ROUND(AVG(High - Low), 4) AS AvgVolatility,
    SUM(Volume) AS TotalVolume,
    ROUND(MAX(High), 2) AS YearHigh,
    ROUND(MIN(Low), 2) AS YearLow
FROM stock_data
GROUP BY YEAR(TO_DATE(`Date`))
ORDER BY Year;

# Most Volatile Days 
SELECT 
    `Date`,
    (High - Low) AS Volatility,
    Open,
    Close
FROM stock_data
ORDER BY Volatility DESC
LIMIT 5;

# Query to find days with highest trading volume
SELECT 
    `Date`,
    Volume
FROM stock_data
ORDER BY Volume DESC
LIMIT 10;

# Calculate variance and standard deviation of closing prices
SELECT 
    ROUND(VARIANCE(Close), 4) AS PriceVariance,
    ROUND(STDDEV(Close), 4) AS PriceStdDev,
    ROUND(AVG(Close), 2) AS AvgClosePrice
FROM stock_data;
