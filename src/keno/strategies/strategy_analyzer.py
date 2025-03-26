"""
Strategy analysis module for Keno game analysis.
Handles different number selection strategies and their performance analysis.
"""

import random
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from keno.analysis.combination_analyzer import CombinationAnalyzer


class StrategyAnalyzer:
    def __init__(
        self,
        historical_data: Union[pd.DataFrame, np.ndarray],
        payout_table: Optional[Dict[int, Dict[int, float]]] = None,
    ):
        """
        Initialize strategy analyzer with historical data

        Args:
            historical_data: DataFrame or array containing historical Keno draws
            payout_table: Optional payout table for different pick sizes
        """
        if isinstance(historical_data, np.ndarray):
            self.data = pd.DataFrame({"numbers": [list(row) for row in historical_data]})
        else:
            self.data = historical_data
        self.strategies = {
            "frequency": self._frequency_strategy,
            "pattern": self._pattern_strategy,
            "random": self._random_strategy,
        }
        self.payout_table = payout_table or {}
        self.combination_analyzer = CombinationAnalyzer()

    def analyze_frequency(self) -> pd.DataFrame:
        """
        Analyze the frequency of each number in historical data

        Returns:
            DataFrame with frequency analysis for each number
        """
        if self.data.empty:
            print("No historical data available")
            return pd.DataFrame()

        # Flatten all numbers from all draws
        all_numbers = []
        for nums in self.data["numbers"]:
            all_numbers.extend(nums)

        # Count frequency of each number
        frequency = Counter(all_numbers)

        # Convert to DataFrame for easier analysis
        freq_df = pd.DataFrame(
            {
                "number": list(range(1, 81)),
                "frequency": [frequency.get(i, 0) for i in range(1, 81)],
                "percentage": [frequency.get(i, 0) / len(self.data) * 5 for i in range(1, 81)],
            }
        )

        return freq_df.sort_values("frequency", ascending=False)

    def _frequency_strategy(self, pick_size: int) -> List[int]:
        """
        Select numbers based on historical frequency

        Args:
            pick_size: Number of spots to pick

        Returns:
            List of selected numbers
        """
        freq_df = self.analyze_frequency()
        return freq_df.head(pick_size)["number"].tolist()

    def _pattern_strategy(self, pick_size: int) -> List[int]:
        """
        Select numbers based on identified patterns

        Args:
            pick_size: Number of spots to pick

        Returns:
            List of selected numbers
        """
        if self.data.empty:
            return self._random_strategy(pick_size)

        patterns = self._analyze_patterns()

        # Use pattern analysis to guide selection
        target_even = round(pick_size * patterns["avg_even_ratio"])
        numbers = []

        # Select even numbers
        even_candidates = [n for n in range(1, 81) if n % 2 == 0]
        numbers.extend(random.sample(even_candidates, target_even))

        # Select odd numbers
        odd_candidates = [n for n in range(1, 81) if n % 2 != 0 and n not in numbers]
        numbers.extend(random.sample(odd_candidates, pick_size - target_even))

        return sorted(numbers)

    def _random_strategy(self, pick_size: int) -> List[int]:
        """
        Select numbers randomly

        Args:
            pick_size: Number of spots to pick

        Returns:
            List of selected numbers
        """
        return sorted(random.sample(range(1, 81), pick_size))

    def _analyze_patterns(self) -> Dict:
        """
        Analyze patterns in historical data

        Returns:
            Dictionary containing pattern statistics
        """
        if self.data.empty:
            return {"avg_even_ratio": 0.5, "avg_high_ratio": 0.5}

        even_ratios = []
        high_ratios = []

        for nums in self.data["numbers"]:
            even_count = sum(1 for n in nums if n % 2 == 0)
            high_count = sum(1 for n in nums if n > 40)

            even_ratios.append(even_count / len(nums))
            high_ratios.append(high_count / len(nums))

        return {
            "avg_even_ratio": sum(even_ratios) / len(even_ratios),
            "avg_high_ratio": sum(high_ratios) / len(high_ratios),
        }

    def simulate_strategy(
        self,
        strategy: str,
        pick_size: int,
        payout_table: Dict[int, float],
        num_draws: int = 100,
        bet_amount: float = 1.0,
    ) -> Dict:
        """
        Simulate a strategy's performance

        Args:
            strategy: Name of strategy to use ('frequency', 'pattern', 'random')
            pick_size: Number of spots to pick
            payout_table: Dictionary mapping number of matches to payout multiplier
            num_draws: Number of draws to simulate
            bet_amount: Amount bet per draw

        Returns:
            Dictionary containing simulation results
        """
        if strategy not in self.strategies:
            print(f"Unknown strategy: {strategy}")
            return {}

        if len(self.data) < 100:
            print("Insufficient historical data for meaningful simulation")
            return {}

        # Select numbers using the specified strategy
        selected_numbers = self.strategies[strategy](pick_size)

        # Simulate draws
        total_spent = num_draws * bet_amount
        total_won = 0.0
        matches_dist = {i: 0 for i in range(pick_size + 1)}

        for i in range(num_draws):
            draw_idx = i % len(self.data)  # Cycle through historical draws
            draw_numbers = self.data.iloc[draw_idx]["numbers"]

            # Count matches
            matches = len(set(selected_numbers).intersection(set(draw_numbers)))

            # Calculate winnings
            winnings = payout_table.get(matches, 0) * bet_amount
            total_won += winnings

            # Update matches distribution
            matches_dist[matches] += 1

        return {
            "strategy": strategy,
            "selected_numbers": selected_numbers,
            "total_spent": total_spent,
            "total_won": total_won,
            "net_profit": total_won - total_spent,
            "roi": ((total_won - total_spent) / total_spent) * 100,
            "matches_distribution": matches_dist,
        }

    def compare_strategies(
        self,
        pick_size: int,
        payout_table: Dict[int, float],
        num_draws: List[int] = [1, 5, 10, 20],
        bet_amount: float = 1.0,
    ) -> pd.DataFrame:
        """
        Compare performance of different strategies

        Args:
            pick_size: Number of spots to pick
            payout_table: Dictionary mapping number of matches to payout multiplier
            num_draws: List of number of draws to simulate
            bet_amount: Amount bet per draw

        Returns:
            DataFrame comparing strategy performances
        """
        results = []

        for strategy in self.strategies.keys():
            for draws in num_draws:
                sim_result = self.simulate_strategy(
                    strategy, pick_size, payout_table, draws, bet_amount
                )

                if sim_result:
                    results.append(
                        {
                            "Strategy": strategy,
                            "Draws": draws,
                            "Selected Numbers": sim_result["selected_numbers"],
                            "Total Spent": sim_result["total_spent"],
                            "Total Won": sim_result["total_won"],
                            "Net Profit": sim_result["net_profit"],
                            "ROI %": sim_result["roi"],
                        }
                    )

        return pd.DataFrame(results).sort_values("ROI %", ascending=False)

    def get_best_strategy(
        self,
        pick_size: int,
        payout_table: Dict[int, float],
        num_draws: List[int] = [1, 5, 10, 20],
        bet_amount: float = 1.0,
    ) -> Tuple[str, int, List[int], float]:
        """
        Determine the best strategy and number of draws

        Args:
            pick_size: Number of spots to pick
            payout_table: Dictionary mapping number of matches to payout multiplier
            num_draws: List of number of draws to simulate
            bet_amount: Amount bet per draw

        Returns:
            Tuple of (best strategy, optimal draws, recommended numbers, expected ROI)
        """
        comparison = self.compare_strategies(pick_size, payout_table, num_draws, bet_amount)
        if comparison.empty:
            return ("random", 1, self._random_strategy(pick_size), 0.0)

        best_result = comparison.iloc[0]
        return (
            best_result["Strategy"],
            best_result["Draws"],
            best_result["Selected Numbers"],
            best_result["ROI %"],
        )

    def analyze_strategy(
        self,
        predictions: Union[List[List[int]], np.ndarray],
        actual_results: Optional[Union[List[List[int]], np.ndarray]] = None,
        pick_size: int = 20,
    ) -> Dict[str, Union[float, List[int]]]:
        """
        Analyze the performance of a strategy by comparing predictions to actual results.

        Args:
            predictions: List or array of predicted number sets
            actual_results: List or array of actual drawn numbers (optional)
            pick_size: Number of spots picked (default: 20)

        Returns:
            Dictionary containing analysis results
        """
        # Convert numpy arrays to lists if needed
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
        if isinstance(actual_results, np.ndarray):
            actual_results = actual_results.tolist()

        # If no actual results provided, use the last len(predictions) draws from historical data
        if actual_results is None:
            if len(self.data) == 0:
                return {"accuracy": 0.0, "matches": [], "hit_rate": 0.0}
            actual_results = self.data["numbers"].tail(len(predictions)).tolist()

        if not predictions or not actual_results:
            return {"accuracy": 0.0, "matches": [], "hit_rate": 0.0}

        total_matches = []
        total_hits = 0

        for pred, actual in zip(predictions, actual_results):
            matches = len(set(pred).intersection(set(actual)))
            total_matches.append(matches)
            if matches > 0:
                total_hits += 1

        accuracy = sum(total_matches) / (len(predictions) * pick_size) if predictions else 0.0
        hit_rate = total_hits / len(predictions) if predictions else 0.0

        return {"accuracy": float(accuracy), "matches": total_matches, "hit_rate": float(hit_rate)}

    def evaluate_performance(
        self,
        predictions: Union[List[int], np.ndarray],
        actual_results: Union[List[int], np.ndarray],
    ) -> float:
        """
        Evaluate the performance of predictions against actual results.

        Args:
            predictions: List or array of predicted numbers
            actual_results: List or array of actual drawn numbers

        Returns:
            Performance score between 0 and 1
        """
        # Convert numpy arrays to lists if needed
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
        if isinstance(actual_results, np.ndarray):
            actual_results = actual_results.tolist()

        if not len(predictions) or not len(actual_results):
            return 0.0

        # Convert inputs to sets for intersection
        pred_set = set(predictions)
        actual_set = set(actual_results)

        # Calculate matches
        matches = len(pred_set.intersection(actual_set))

        # Calculate score based on matches relative to prediction size
        score = matches / len(predictions) if predictions else 0.0

        return float(score)  # Ensure we return a native float

    def simulate_strategy(
        self, strategy_func: callable, num_simulations: int = 1000, pick_size: int = 5
    ) -> Dict[str, Union[float, List[List[int]]]]:
        """
        Simulate a strategy over multiple draws.

        Args:
            strategy_func: Function that generates predictions
            num_simulations: Number of simulations to run
            pick_size: Number of numbers to pick

        Returns:
            Dictionary containing simulation results
        """
        predictions = []
        actual_results = []
        total_matches = 0

        for _ in range(num_simulations):
            # Generate prediction
            pred = strategy_func(pick_size)
            predictions.append(pred)

            # Simulate actual draw
            actual = sorted(np.random.choice(range(1, 81), size=20, replace=False))
            actual_results.append(actual)

            # Calculate matches
            matches = len(set(pred).intersection(set(actual)))
            total_matches += matches

        # Calculate metrics
        avg_matches = total_matches / num_simulations
        expected_matches = self._calculate_expected_matches(pick_size)
        p_value = self._calculate_significance(total_matches, num_simulations)

        return {
            "total_simulations": num_simulations,
            "total_matches": total_matches,
            "average_matches": avg_matches,
            "expected_matches": expected_matches,
            "p_value": p_value,
            "predictions": predictions,
            "actual_results": actual_results,
        }

    def _calculate_expected_matches(self, pick_size: int) -> float:
        """
        Calculate expected number of matches for a pick size.

        Args:
            pick_size: Number of numbers to pick

        Returns:
            Expected number of matches
        """
        return (pick_size * 20) / 80  # Simple probability calculation

    def _calculate_significance(self, matches: int, n: int) -> float:
        """
        Calculate statistical significance of matches.

        Args:
            matches: Number of matches
            n: Number of trials

        Returns:
            p-value
        """
        return stats.binomtest(matches, n=n, p=20 / 80).pvalue

    def identify_hot_numbers(self, results: List[List[int]], window: int = 30) -> List[int]:
        """
        Identify hot numbers based on recent results.

        Args:
            results: List of draw results
            window: Number of recent draws to analyze

        Returns:
            List of hot numbers
        """
        if not results:
            raise ValueError("No results provided")

        # Use last N draws
        recent_draws = results[-window:]

        # Count frequency of each number
        frequency = {}
        for draw in recent_draws:
            for num in draw:
                frequency[num] = frequency.get(num, 0) + 1

        # Sort by frequency and return top 20
        sorted_nums = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:20]]

    def identify_cold_numbers(self, results: List[List[int]], window: int = 30) -> List[int]:
        """
        Identify cold numbers based on recent results.

        Args:
            results: List of draw results
            window: Number of recent draws to analyze

        Returns:
            List of cold numbers
        """
        if not results:
            raise ValueError("No results provided")

        # Use last N draws
        recent_draws = results[-window:]

        # Count frequency of each number
        frequency = {}
        for num in range(1, 81):
            frequency[num] = 0
            for draw in recent_draws:
                if num in draw:
                    frequency[num] += 1

        # Sort by frequency and return bottom 20
        sorted_nums = sorted(frequency.items(), key=lambda x: x[1])
        return [num for num, _ in sorted_nums[:20]]

    def analyze_patterns(self, results: List[List[int]], window: int = 30) -> Dict[str, List[int]]:
        """
        Analyze patterns in recent results.

        Args:
            results: List of draw results
            window: Number of recent draws to analyze

        Returns:
            Dictionary containing pattern analysis results
        """
        if not results:
            raise ValueError("No results provided")

        # Use last N draws
        recent_draws = results[-window:]

        patterns = {"consecutive": [], "gaps": [], "repeats": []}

        for i in range(len(recent_draws) - 1):
            current = set(recent_draws[i])
            next_draw = set(recent_draws[i + 1])

            # Check for consecutive numbers
            for num in current:
                if num + 1 in current:
                    patterns["consecutive"].append(num)

            # Check for gaps
            for num in range(1, 81):
                if num not in current and num not in next_draw:
                    patterns["gaps"].append(num)

            # Check for repeats
            repeats = current.intersection(next_draw)
            patterns["repeats"].extend(list(repeats))

        # Remove duplicates and sort
        for key in patterns:
            patterns[key] = sorted(list(set(patterns[key])))

        return patterns
