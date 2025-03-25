-- Schema for Keno validation database

-- Table for storing predictions
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    method TEXT NOT NULL,
    predicted_numbers TEXT NOT NULL,  -- JSON array of numbers
    draw_date DATE,
    draw_numbers TEXT,  -- JSON array of numbers, NULL until validated
    accuracy REAL,  -- NULL until validated
    matches INTEGER,  -- NULL until validated
    metadata TEXT  -- JSON object for additional info
);

-- Table for storing method performance statistics
CREATE TABLE IF NOT EXISTS method_stats (
    method TEXT PRIMARY KEY,
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    average_accuracy REAL DEFAULT 0.0,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing validation metrics
CREATE TABLE IF NOT EXISTS validation_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    method TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    window_size INTEGER,  -- For rolling metrics
    metadata TEXT  -- JSON object for additional info
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_predictions_method ON predictions(method);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(draw_date);
CREATE INDEX IF NOT EXISTS idx_validation_metrics_method ON validation_metrics(method); 