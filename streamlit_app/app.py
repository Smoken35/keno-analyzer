#!/usr/bin/env python3
"""
Streamlit app for Keno prediction system.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure the page
st.set_page_config(
    page_title="Keno Prediction System",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
    .stSlider {
        margin-bottom: 1rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)


def load_css() -> None:
    """Load custom CSS styles."""
    css_file = Path(__file__).parent / "static" / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def create_confidence_gauge(confidence: float) -> go.Figure:
    """Create a gauge chart for confidence score.

    Args:
        confidence: Confidence score between 0 and 1

    Returns:
        Plotly figure object
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 33], "color": "lightgray"},
                    {"range": [33, 66], "color": "gray"},
                    {"range": [66, 100], "color": "darkgray"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": confidence * 100,
                },
            },
            title={"text": "Confidence Score"},
        )
    )
    fig.update_layout(height=250, margin={"t": 25, "b": 25, "l": 25, "r": 25})
    return fig


def create_hit_rate_chart(hit_rates: Dict[str, float]) -> go.Figure:
    """Create a bar chart for hit rates by strategy.

    Args:
        hit_rates: Dictionary of strategy names and their hit rates

    Returns:
        Plotly figure object
    """
    fig = px.bar(
        x=list(hit_rates.keys()),
        y=list(hit_rates.values()),
        title="Hit Rates by Strategy",
        labels={"x": "Strategy", "y": "Hit Rate (%)"},
    )
    fig.update_layout(height=300)
    return fig


def main() -> None:
    """Main function to run the Streamlit app."""
    # Load custom CSS
    load_css()

    # Sidebar
    with st.sidebar:
        st.title("🎲 Keno Predictor")
        st.markdown("---")
        st.markdown(
            """
            ### Instructions
            1. Upload your Keno draw data (CSV)
            2. Select prediction strategy
            3. Choose number of picks
            4. Click Predict
        """
        )
        st.markdown("---")
        st.markdown(
            """
            ### About
            This app uses advanced algorithms to predict Keno numbers based on historical data.
            Choose from different strategies to optimize your predictions.
        """
        )

    # Main content
    st.title("🔮 Keno Prediction System")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Keno draw data (CSV)",
        type=["csv"],
        help="Upload a CSV file containing historical Keno draw data",
    )

    # Create two columns for strategy and picks
    col1, col2 = st.columns(2)

    with col1:
        # Strategy selection
        strategy = st.selectbox(
            "Choose prediction strategy",
            ["Pattern-Based", "Rule-Based", "Cluster-Based"],
            help="Select the prediction algorithm to use",
        )

    with col2:
        # Pick size
        pick_size = st.slider(
            "Number of Picks",
            min_value=5,
            max_value=20,
            value=10,
            help="Select how many numbers to predict",
        )

    # Predict button
    if st.button("Predict", type="primary"):
        if uploaded_file:
            try:
                # Placeholder for prediction logic
                st.success("Running prediction... (backend integration coming next)")

                # Example visualizations (to be replaced with real data)
                st.subheader("Prediction Results")

                # Create two columns for visualizations
                viz_col1, viz_col2 = st.columns(2)

                with viz_col1:
                    st.plotly_chart(create_confidence_gauge(0.75), use_container_width=True)

                with viz_col2:
                    st.plotly_chart(
                        create_hit_rate_chart({"Pattern": 65, "Rule": 58, "Cluster": 62}),
                        use_container_width=True,
                    )

                # Results table (placeholder)
                st.subheader("Predicted Numbers")
                results_df = pd.DataFrame({"Number": range(1, 81), "Confidence": [0.8] * 80})
                st.dataframe(results_df.nlargest(pick_size, "Confidence"), use_container_width=True)

                # Download button
                st.download_button(
                    label="Download Results",
                    data=results_df.to_csv(index=False),
                    file_name="keno_predictions.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            st.warning("Please upload a CSV file first.")


if __name__ == "__main__":
    main()
