"""Keno analysis module for analyzing historical data and making predictions."""

import glob
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from keno.analysis.combination_analyzer import CombinationAnalyzer
from keno.analysis.gap_analyzer import KenoGapAnalyzer
from keno.analysis.payout_analyzer import PayoutAnalyzer


class KenoAnalyzer:
    """Analyzes Keno historical data and provides various analysis methods."""

    def __init__(self, data_source: Union[str, List[List[int]]]):
        """Initialize the analyzer with historical data.

        Args:
            data_source: Either a path to data file or list of historical draws
        """
        self.data = self._load_data(data_source)
        self.payout_table = None
        self.combination_analyzer = CombinationAnalyzer()
        self.gap_analyzer = KenoGapAnalyzer()
        self.payout_analyzer = PayoutAnalyzer()

    def _load_data(self, data_source: Union[str, List[List[int]]]) -> List[List[int]]:
        """Load historical data from source.

        Args:
            data_source: Either a path to data file or list of historical draws

        Returns:
            List of historical draws
        """
        if isinstance(data_source, list):
            return data_source
        # Add file loading logic here if needed
        return []

    def analyze_frequency(self) -> Dict[int, int]:
        """Analyze frequency of each number in historical data.

        Returns:
            Dictionary mapping numbers to their frequencies
        """
        if not self.data:
            return {i: 0 for i in range(1, 81)}

        freq = {}
        for draw in self.data:
            for num in draw:
                freq[num] = freq.get(num, 0) + 1

        # Fill in missing numbers with 0
        for i in range(1, 81):
            if i not in freq:
                freq[i] = 0

        return freq

    def analyze_patterns(self, window: int = 10) -> Dict[str, List[int]]:
        """Analyze patterns in recent draws.

        Args:
            window: Number of recent draws to analyze

        Returns:
            Dictionary containing hot and cold numbers (20 each)
        """
        if not self.data:
            return {"hot_numbers": [], "cold_numbers": []}

        recent_draws = self.data[-window:]
        freq = {}
        for draw in recent_draws:
            for num in draw:
                freq[num] = freq.get(num, 0) + 1

        # Fill in missing numbers with 0
        for i in range(1, 81):
            if i not in freq:
                freq[i] = 0

        # Sort numbers by frequency
        sorted_nums = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        # Split into hot and cold numbers (20 each)
        hot_nums = [num for num, _ in sorted_nums[:20]]
        cold_nums = [num for num, _ in sorted_nums[-20:]]  # Take 20 least frequent numbers

        return {"hot_numbers": sorted(hot_nums), "cold_numbers": sorted(cold_nums)}

    def analyze_due_numbers(self) -> List[Tuple[int, float]]:
        """Analyze numbers that are due to appear based on historical frequency.

        Returns:
            List of tuples (number, due_score) sorted by due score
        """
        if not self.data:
            return [(i, 0.0) for i in range(1, 81)]

        freq = self.analyze_frequency()
        total_draws = len(self.data)
        expected_freq = total_draws / 80  # Expected frequency for each number

        due_scores = []
        for num in range(1, 81):
            actual_freq = freq.get(num, 0)
            due_score = max(0.0, expected_freq - actual_freq)
            due_scores.append((num, due_score))

        return sorted(due_scores, key=lambda x: x[1], reverse=True)

    def analyze_cyclic_patterns(self) -> Dict[str, Union[int, float]]:
        """
        Analyze cyclic patterns in the data.

        Returns:
            Dictionary containing cycle analysis results with normalized probabilities
        """
        if not self.data:
            return {"cycle_length": 0, "confidence": 0.0, "probabilities": {}}

        cycle_length = self._find_cycle_length()
        confidence = self._calculate_cycle_confidence(cycle_length)

        # Calculate probabilities for each possible cycle length
        probabilities = {}
        max_length = min(20, len(self.data) // 2)
        total_score = 0.0

        for length in range(2, max_length + 1):
            score = self._calculate_cycle_score(length)
            probabilities[length] = score
            total_score += score

        # Normalize probabilities to sum to 1.0
        if total_score > 0:
            probabilities = {k: v / total_score for k, v in probabilities.items()}

        # Normalize confidence to [0, 1]
        normalized_confidence = min(1.0, confidence)

        return {
            "cycle_length": int(cycle_length),
            "confidence": float(normalized_confidence),
            "probabilities": probabilities,
        }

    def _find_cycle_length(self) -> int:
        """Find the most likely cycle length in the data."""
        if len(self.data) < 2:
            return 0

        # Try different cycle lengths
        max_length = min(20, len(self.data) // 2)
        best_length = 0
        best_score = 0.0

        for length in range(2, max_length + 1):
            score = self._calculate_cycle_score(length)
            if score > best_score:
                best_score = score
                best_length = length

        return best_length

    def _calculate_cycle_score(self, length: int) -> float:
        """Calculate how well a cycle length fits the data."""
        if length >= len(self.data):
            return 0.0

        score = 0.0
        for i in range(len(self.data) - length):
            current = set(self.data[i])
            next_draw = set(self.data[i + length])
            overlap = len(current.intersection(next_draw))
            score += overlap / 20.0  # Normalize by maximum possible overlap

        # Normalize by the number of comparisons made
        return score / (len(self.data) - length) if len(self.data) > length else 0.0

    def _calculate_cycle_confidence(self, cycle_length: int) -> float:
        """Calculate confidence in the cycle length."""
        if cycle_length == 0:
            return 0.0

        score = self._calculate_cycle_score(cycle_length)
        # Normalize confidence to [0, 1]
        return min(1.0, score)

    def build_transition_matrix(self) -> np.ndarray:
        """
        Build a Markov transition matrix from the data.

        Returns:
            80x80 transition probability matrix
        """
        matrix = np.zeros((80, 80))

        for i in range(len(self.data) - 1):
            current = self.data[i]
            next_draw = self.data[i + 1]

            for num1 in current:
                for num2 in next_draw:
                    matrix[num1 - 1][num2 - 1] += 1

        # Normalize rows
        row_sums = matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix = matrix / row_sums[:, np.newaxis]

        return matrix

    def analyze_skip_and_hit_patterns(self) -> Dict[int, Dict[str, Union[float, List[int]]]]:
        """
        Analyze skip and hit patterns for each number.

        Returns:
            Dictionary mapping numbers to their skip and hit pattern statistics
        """
        if not self.data:
            return {
                i: {
                    "avg_skip": 0.0,
                    "avg_hit": 0.0,
                    "max_skip": 0,
                    "skip_history": [],
                    "hit_history": [],
                    "hit_rate": 0.0,
                }
                for i in range(1, 81)
            }

        patterns = {i: {"skip_history": [], "hit_history": []} for i in range(1, 81)}

        # Track the last appearance and current sequences
        last_seen = {i: -1 for i in range(1, 81)}
        current_hit = {i: 0 for i in range(1, 81)}
        total_hits = {i: 0 for i in range(1, 81)}

        # Analyze patterns
        for idx, draw in enumerate(self.data):
            draw_set = set(draw)
            for num in range(1, 81):
                if num in draw_set:
                    # Record skip length if not first appearance
                    if last_seen[num] != -1:
                        skip_length = idx - last_seen[num] - 1
                        if skip_length > 0:
                            patterns[num]["skip_history"].append(skip_length)
                    current_hit[num] += 1
                    total_hits[num] += 1
                    last_seen[num] = idx
                else:
                    # Record hit length if sequence ends
                    if current_hit[num] > 0:
                        patterns[num]["hit_history"].append(current_hit[num])
                        current_hit[num] = 0

        # Add final sequences
        last_draw_idx = len(self.data) - 1
        for num in range(1, 81):
            if current_hit[num] > 0:
                patterns[num]["hit_history"].append(current_hit[num])
            if last_seen[num] != -1 and last_seen[num] < last_draw_idx:
                skip_length = last_draw_idx - last_seen[num]
                patterns[num]["skip_history"].append(skip_length)

        # Calculate statistics
        for num in range(1, 81):
            skip_history = patterns[num]["skip_history"]
            hit_history = patterns[num]["hit_history"]
            patterns[num]["avg_skip"] = float(np.mean(skip_history)) if skip_history else 0.0
            patterns[num]["avg_hit"] = float(np.mean(hit_history)) if hit_history else 0.0
            patterns[num]["max_skip"] = int(np.max(skip_history)) if skip_history else 0
            patterns[num]["hit_rate"] = float(total_hits[num] / len(self.data))

        return patterns

    def simulate_strategy(
        self, method: str, pick_size: int, bet_size: float, num_simulations: int = 1000
    ) -> Dict[str, Union[float, int, List[List[int]], Dict[int, int]]]:
        """
        Simulate a betting strategy.

        Args:
            method: Prediction method to use
            pick_size: Number of numbers to pick
            bet_size: Size of each bet
            num_simulations: Number of simulations to run

        Returns:
            Dictionary containing simulation results including:
            - total_return: Total return across all simulations
            - average_return: Average return per simulation
            - win_rate: Proportion of winning simulations
            - roi_percent: Return on investment percentage
            - num_simulations: Number of simulations run
            - predictions: List of predictions used
            - match_distribution: Distribution of number matches
        """
        if pick_size not in self.payout_table:
            raise ValueError(f"No payout table for pick size {pick_size}")

        total_return = 0.0
        wins = 0
        match_dist = defaultdict(int)
        predictions = []

        # Get prediction for this method
        prediction = self.predict_next_draw(method=method, picks=pick_size)
        predictions.extend([prediction] * num_simulations)

        # Run simulations
        for _ in range(num_simulations):
            # Generate random draw
            actual = sorted(np.random.choice(range(1, 81), size=20, replace=False))

            # Count matches
            matches = len(set(prediction) & set(actual))
            match_dist[matches] += 1

            # Calculate return
            payout = self.payout_table[pick_size].get(matches, 0.0)
            total_return += payout - bet_size
            if payout > bet_size:
                wins += 1

        # Calculate statistics
        average_return = total_return / num_simulations
        win_rate = wins / num_simulations
        roi_percent = (total_return / (bet_size * num_simulations)) * 100

        return {
            "total_return": float(total_return),
            "average_return": float(average_return),
            "win_rate": float(win_rate),
            "roi_percent": float(roi_percent),
            "num_simulations": num_simulations,
            "predictions": predictions,
            "match_distribution": dict(match_dist),
        }

    def predict_next_draw(self, method: str = "frequency", picks: int = 20) -> List[int]:
        """Predict numbers for the next draw using specified method.

        Args:
            method: Prediction method ("frequency", "patterns", "due", "markov")
            picks: Number of numbers to predict (1-20)

        Returns:
            List of predicted numbers

        Raises:
            ValueError: If method is invalid or picks is out of range
        """
        if not isinstance(picks, int) or picks < 1 or picks > 20:
            raise ValueError("picks must be an integer between 1 and 20")

        if method not in ["frequency", "patterns", "due", "markov"]:
            raise ValueError("Invalid prediction method")

        if not self.data:
            return sorted(np.random.choice(range(1, 81), size=picks, replace=False))

        if method == "frequency":
            freq = self.analyze_frequency()
            sorted_nums = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            return [num for num, _ in sorted_nums[:picks]]

        elif method == "patterns":
            patterns = self.analyze_patterns()
            return patterns["hot_numbers"][:picks]

        elif method == "due":
            due_nums = self.analyze_due_numbers()
            return [num for num, _ in due_nums[:picks]]

        else:  # markov
            return self._predict_markov(picks)

    def _predict_markov(self, picks: int) -> List[int]:
        """Predict numbers using Markov chain analysis.

        Args:
            picks: Number of numbers to predict

        Returns:
            List of predicted numbers
        """
        if len(self.data) < 2:
            return sorted(np.random.choice(range(1, 81), size=picks, replace=False))

        # Create transition matrix
        transitions = np.zeros((80, 80))
        for i in range(len(self.data) - 1):
            prev_draw = set(self.data[i])
            curr_draw = set(self.data[i + 1])
            for prev in prev_draw:
                for curr in curr_draw:
                    transitions[prev - 1][curr - 1] += 1

        # Normalize transition matrix
        row_sums = transitions.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transitions = transitions / row_sums[:, np.newaxis]

        # Get last draw
        last_draw = set(self.data[-1])

        # Calculate probabilities for next draw
        probs = np.zeros(80)
        for num in last_draw:
            probs += transitions[num - 1]

        # Select top picks
        top_indices = np.argsort(probs)[-picks:]
        return sorted(top_indices + 1)

    def calculate_expected_value(self, pick_size: int, method: str = "frequency") -> float:
        """
        Calculate expected value for a given pick size and prediction method.

        Args:
            pick_size: Number of numbers to pick
            method: Prediction method to use

        Returns:
            Expected value of the bet

        Raises:
            ValueError: If pick_size is not in the payout table
        """
        if not isinstance(pick_size, int):
            raise ValueError("Pick size must be an integer")
        if pick_size < 1 or pick_size > 20:
            raise ValueError("Pick size must be between 1 and 20")
        if pick_size not in self.payout_table:
            raise ValueError(f"No payout table for pick size {pick_size}")

        # Get predictions using the specified method
        predictions = self.predict_next_draw(method=method, picks=pick_size)

        # Calculate probabilities for each possible number of matches
        total_numbers = 80
        total_drawn = 20
        ev = 0.0

        for matches in range(pick_size + 1):
            # Calculate probability of getting exactly 'matches' numbers correct
            prob = stats.hypergeom.pmf(matches, total_numbers, total_drawn, pick_size)
            payout = self.payout_table[pick_size].get(matches, 0.0)
            ev += prob * payout

        return ev

    def set_payout_table(self, table: Dict[int, Dict[int, float]]):
        """
        Set the payout table.

        Args:
            table: Dictionary mapping pick sizes to payout dictionaries
        """
        self.payout_table = table
