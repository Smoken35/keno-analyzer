# Keno Prediction System

A user-friendly web interface for the Keno prediction system, built with Streamlit.

## Features

- 📂 CSV file upload for historical Keno draw data
- 🧠 Multiple prediction strategies (Pattern, Rule, Cluster)
- 🎯 Configurable number of picks
- 📈 Interactive visualizations
- 📊 Confidence scores and hit rates
- 📥 Download predictions as CSV

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

1. Navigate to the streamlit_app directory:
```bash
cd streamlit_app
```

2. Start the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and go to the URL shown in the terminal (typically http://localhost:8501)

## Usage

1. Upload your Keno draw data as a CSV file
2. Select a prediction strategy
3. Choose the number of picks (5-20)
4. Click "Predict" to generate predictions
5. View the results and download them as CSV

## Data Format

The CSV file should contain historical Keno draw data with the following columns:
- `draw_date`: Date of the draw
- `numbers`: Comma-separated list of drawn numbers
- `draw_number`: Unique identifier for the draw

Example:
```csv
draw_date,numbers,draw_number
2023-01-01,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,DRAW001
```

## Development

The app is structured as follows:
```
streamlit_app/
├── app.py              # Main Streamlit application
├── static/             # Static files (CSS, images)
│   └── style.css      # Custom CSS styles
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
