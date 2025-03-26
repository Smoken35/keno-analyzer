"""
Payout analysis module for Keno game analysis.
Handles expected value calculations and payout comparisons.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class PayoutAnalyzer:
    def __init__(self):
        self.payout_tables: Dict[str, Dict[int, Dict[int, float]]] = {}

    def set_payout_table(self, variant_name: str, payout_data: Dict[int, Dict[int, float]]) -> None:
        """
        Set payout table for a specific Keno variant

        Args:
            variant_name: Name of the Keno variant
            payout_data: Dictionary with payout data structured as:
                        {pick_size: {matches: payout, ...}, ...}
                        Example: {4: {4: 100, 3: 5, 2: 1, 1: 0, 0: 0}, ...}
        """
        self.payout_tables[variant_name] = payout_data

    def calculate_expected_value(
        self, variant_name: str, pick_size: int, bet_amount: float = 1.0
    ) -> Optional[Dict]:
        """
        Calculate the expected value of a specific Keno play type

        Args:
            variant_name: Name of the Keno variant
            pick_size: Number of spots picked (e.g., 4 for Pick 4)
            bet_amount: Amount bet per draw

        Returns:
            Dictionary containing expected value and other statistics
        """
        if variant_name not in self.payout_tables:
            print(f"No payout table found for {variant_name}")
            return None

        if pick_size not in self.payout_tables[variant_name]:
            print(f"No payout data for Pick {pick_size} in {variant_name}")
            return None

        payout_table = self.payout_tables[variant_name][pick_size]
        total_numbers = 80
        drawn_numbers = 20  # Standard Keno draws 20 numbers

        # Calculate probability and expected value for each possible outcome
        probabilities = {}
        expected_value = 0.0

        for matches in range(min(pick_size, drawn_numbers) + 1):
            probability = self._calculate_probability(
                total_numbers, drawn_numbers, pick_size, matches
            )
            payout = payout_table.get(matches, 0) * bet_amount

            expected_value += probability * payout
            probabilities[matches] = probability

        return {
            "variant": variant_name,
            "pick_size": pick_size,
            "bet_amount": bet_amount,
            "expected_value": expected_value,
            "return_percentage": expected_value / bet_amount * 100,
            "probabilities": probabilities,
        }

    def _calculate_probability(self, total: int, drawn: int, picked: int, matches: int) -> float:
        """
        Calculate probability using hypergeometric distribution
        P(X = matches) = [C(picked, matches) * C(total - picked, drawn - matches)] / C(total, drawn)

        Args:
            total: Total numbers in the pool (usually 80)
            drawn: Numbers drawn each game (usually 20)
            picked: Numbers picked by player
            matches: Number of matches to calculate probability for

        Returns:
            Probability as a float
        """
        if matches > min(picked, drawn) or matches < max(0, picked + drawn - total):
            return 0.0

        try:
            numerator = self._combinations(picked, matches) * self._combinations(
                total - picked, drawn - matches
            )
            denominator = self._combinations(total, drawn)
            return numerator / denominator
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _combinations(self, n: int, k: int) -> int:
        """
        Calculate combinations (n choose k)

        Args:
            n: Total number of items
            k: Number of items to choose

        Returns:
            Number of possible combinations
        """
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1

        k = min(k, n - k)
        c = 1
        for i in range(k):
            c = c * (n - i) // (i + 1)
        return c

    def compare_play_types(self, variant_name: str, bet_amount: float = 1.0) -> Optional[Dict]:
        """
        Compare expected values and other metrics for different play types

        Args:
            variant_name: Name of the Keno variant
            bet_amount: Amount bet per draw

        Returns:
            Dictionary containing comparison metrics for each play type
        """
        if variant_name not in self.payout_tables:
            print(f"No payout table found for {variant_name}")
            return None

        comparison = {}

        for pick_size in self.payout_tables[variant_name].keys():
            ev_data = self.calculate_expected_value(variant_name, pick_size, bet_amount)
            if ev_data:
                max_payout = max(self.payout_tables[variant_name][pick_size].values()) * bet_amount
                comparison[pick_size] = {
                    "expected_value": ev_data["expected_value"],
                    "return_percentage": ev_data["return_percentage"],
                    "max_payout": max_payout,
                    "probabilities": ev_data["probabilities"],
                }

        return comparison

    def get_best_play_type(
        self, variant_name: str, bet_amount: float = 1.0
    ) -> Optional[Tuple[int, Dict]]:
        """
        Determine the best play type based on expected value

        Args:
            variant_name: Name of the Keno variant
            bet_amount: Amount bet per draw

        Returns:
            Tuple of (best pick size, metrics) or None if no data available
        """
        comparison = self.compare_play_types(variant_name, bet_amount)
        if not comparison:
            return None

        best_pick = max(comparison.items(), key=lambda x: x[1]["return_percentage"])

        return best_pick
