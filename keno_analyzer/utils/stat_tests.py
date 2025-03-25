"""
Statistical tests for Keno randomness analysis.
"""

import numpy as np
from scipy import stats
from scipy.stats import entropy
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class KenoRandomnessTester:
    """Implements statistical tests for Keno randomness analysis."""
    
    def __init__(self):
        """Initialize the randomness tester."""
        self.expected_entropy = 86.4  # Expected entropy for 20 draws from 80 numbers
        self.expected_frequency = 0.25  # Expected frequency for each number (20/80)
    
    def calculate_entropy(self, numbers: List[List[int]]) -> Dict:
        """
        Calculate Shannon entropy for a sequence of Keno draws.
        
        Args:
            numbers: List of lists, each containing 20 numbers from 1-80
            
        Returns:
            Dictionary containing entropy metrics
        """
        # Flatten all numbers into a single sequence
        all_numbers = [n for draw in numbers for n in draw]
        
        # Calculate frequency distribution
        freq_dist = np.zeros(80)
        for n in all_numbers:
            freq_dist[n-1] += 1
        freq_dist = freq_dist / len(all_numbers)
        
        # Calculate entropy
        observed_entropy = entropy(freq_dist, base=2) * 20  # Multiply by 20 for total entropy
        
        # Calculate deviation from expected
        entropy_deviation = abs(observed_entropy - self.expected_entropy)
        
        # Identify most over/underrepresented numbers
        number_freq = [(i+1, freq) for i, freq in enumerate(freq_dist)]
        sorted_freq = sorted(number_freq, key=lambda x: x[1], reverse=True)
        
        return {
            'observed_entropy': observed_entropy,
            'expected_entropy': self.expected_entropy,
            'entropy_deviation': entropy_deviation,
            'most_overrepresented': [n for n, _ in sorted_freq[:5]],
            'most_underrepresented': [n for n, _ in sorted_freq[-5:]]
        }
    
    def chi_square_test(self, numbers: List[List[int]]) -> Dict:
        """
        Perform chi-square test for uniformity.
        
        Args:
            numbers: List of lists, each containing 20 numbers from 1-80
            
        Returns:
            Dictionary containing chi-square test results
        """
        # Flatten all numbers into a single sequence
        all_numbers = [n for draw in numbers for n in draw]
        
        # Calculate observed frequencies
        observed = np.zeros(80)
        for n in all_numbers:
            observed[n-1] += 1
        
        # Calculate expected frequencies
        expected = np.full(80, len(all_numbers) / 80)
        
        # Perform chi-square test
        chi2, p_value = stats.chisquare(observed, expected)
        
        # Determine result interpretation
        if p_value < 0.01:
            result = "Non-uniform distribution (p < 0.01)"
        elif p_value < 0.05:
            result = "Possible non-uniform distribution (p < 0.05)"
        else:
            result = "Uniform distribution (p >= 0.05)"
        
        return {
            'chi_squared_statistic': chi2,
            'p_value': p_value,
            'result': result,
            'observed_frequencies': observed.tolist(),
            'expected_frequencies': expected.tolist()
        }
    
    def detect_temporal_drift(self, numbers: List[List[int]], dates: List[str]) -> List[Dict]:
        """
        Detect temporal drift in number frequencies.
        
        Args:
            numbers: List of lists, each containing 20 numbers from 1-80
            dates: List of date strings in 'YYYY-MM' format
            
        Returns:
            List of dictionaries containing drift information per month
        """
        # Group numbers by month
        monthly_numbers = {}
        for draw, date in zip(numbers, dates):
            month = date[:7]  # Extract YYYY-MM
            if month not in monthly_numbers:
                monthly_numbers[month] = []
            monthly_numbers[month].extend(draw)
        
        drift_results = []
        for month, month_numbers in monthly_numbers.items():
            # Calculate frequencies for this month
            freq = np.zeros(80)
            for n in month_numbers:
                freq[n-1] += 1
            freq = freq / len(month_numbers)
            
            # Calculate drift score (deviation from expected frequency)
            drift = np.abs(freq - self.expected_frequency)
            drift_score = np.mean(drift)
            
            # Identify drifting numbers (deviation > 2 standard deviations)
            std_dev = np.sqrt(self.expected_frequency * (1 - self.expected_frequency) / len(month_numbers))
            drifting_numbers = [i+1 for i, d in enumerate(drift) if d > 2*std_dev]
            
            drift_results.append({
                'month': month,
                'drifting_numbers': drifting_numbers,
                'drift_score': float(drift_score),
                'frequency_deviation': drift.tolist()
            })
        
        return drift_results
    
    def detect_autocorrelation(self, numbers: List[List[int]], max_lag: int = 100) -> Dict:
        """
        Detect autocorrelation in number sequences.
        
        Args:
            numbers: List of lists, each containing 20 numbers from 1-80
            max_lag: Maximum lag to check for correlation
            
        Returns:
            Dictionary containing autocorrelation analysis results
        """
        # Flatten numbers into a single sequence
        all_numbers = [n for draw in numbers for n in draw]
        
        # Calculate autocorrelation for different lags
        autocorr = []
        for lag in range(1, min(max_lag, len(all_numbers))):
            corr = np.corrcoef(all_numbers[:-lag], all_numbers[lag:])[0,1]
            autocorr.append((lag, corr))
        
        # Find significant correlations
        significant_correlations = [
            {'lag': lag, 'correlation': corr}
            for lag, corr in autocorr
            if abs(corr) > 0.1  # Threshold for significance
        ]
        
        # Check for repeating patterns
        pattern_lengths = []
        for lag, corr in autocorr:
            if corr > 0.5:  # Strong correlation threshold
                pattern_lengths.append(lag)
        
        return {
            'autocorrelation': autocorr,
            'significant_correlations': significant_correlations,
            'pattern_lengths': pattern_lengths,
            'has_repeating_patterns': len(pattern_lengths) > 0
        }
    
    def calculate_randomness_score(self, numbers: List[List[int]], dates: List[str]) -> Dict:
        """
        Calculate overall randomness score based on multiple metrics.
        
        Args:
            numbers: List of lists, each containing 20 numbers from 1-80
            dates: List of date strings in 'YYYY-MM' format
            
        Returns:
            Dictionary containing overall randomness score and component scores
        """
        # Calculate component scores
        entropy_results = self.calculate_entropy(numbers)
        chi2_results = self.chi_square_test(numbers)
        drift_results = self.detect_temporal_drift(numbers, dates)
        autocorr_results = self.detect_autocorrelation(numbers)
        
        # Calculate component scores (0-100)
        entropy_score = max(0, 100 - (entropy_results['entropy_deviation'] * 10))
        chi2_score = 100 * chi2_results['p_value']
        drift_score = 100 * (1 - np.mean([r['drift_score'] for r in drift_results]))
        pattern_score = 100 if not autocorr_results['has_repeating_patterns'] else 50
        
        # Calculate weighted average
        weights = {'entropy': 0.3, 'chi2': 0.3, 'drift': 0.2, 'patterns': 0.2}
        overall_score = (
            weights['entropy'] * entropy_score +
            weights['chi2'] * chi2_score +
            weights['drift'] * drift_score +
            weights['patterns'] * pattern_score
        )
        
        # Determine confidence level
        if overall_score >= 90:
            confidence = "Likely random"
        elif overall_score >= 70:
            confidence = "Slight imbalance"
        elif overall_score >= 50:
            confidence = "Possible bias"
        else:
            confidence = "Likely exploitable"
        
        return {
            'overall_score': overall_score,
            'confidence_level': confidence,
            'component_scores': {
                'entropy': entropy_score,
                'chi_square': chi2_score,
                'drift': drift_score,
                'patterns': pattern_score
            },
            'entropy_results': entropy_results,
            'chi_square_results': chi2_results,
            'drift_results': drift_results,
            'autocorrelation_results': autocorr_results
        } 