#!/usr/bin/env python3
"""
Association Rule Engine - Generates and ranks association rules from frequent patterns.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AssociationRule:
    """Data class for storing association rule information."""

    antecedent: List[int]
    consequent: List[int]
    support: float
    confidence: float
    lift: float
    timestamp: str


class AssociationEngine:
    """Implements association rule generation and ranking."""

    def __init__(self, min_confidence: float = 0.4, min_lift: float = 1.2):
        """
        Initialize the association rule engine.

        Args:
            min_confidence: Minimum confidence threshold for rules
            min_lift: Minimum lift threshold for rules
        """
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.rules: List[AssociationRule] = []
        self.transaction_count = 0

    def _calculate_rule_metrics(
        self, antecedent: Set[int], consequent: Set[int], transactions: List[Set[int]]
    ) -> Tuple[float, float, float]:
        """
        Calculate support, confidence, and lift for a rule.

        Args:
            antecedent: Set of antecedent items
            consequent: Set of consequent items
            transactions: List of transaction sets

        Returns:
            Tuple of (support, confidence, lift)
        """
        # Count transactions containing antecedent and consequent
        antecedent_count = 0
        both_count = 0
        consequent_count = 0

        for transaction in transactions:
            if antecedent.issubset(transaction):
                antecedent_count += 1
            if consequent.issubset(transaction):
                consequent_count += 1
            if antecedent.issubset(transaction) and consequent.issubset(transaction):
                both_count += 1

        # Calculate metrics
        support = both_count / self.transaction_count
        confidence = both_count / antecedent_count if antecedent_count > 0 else 0
        expected_support = (antecedent_count / self.transaction_count) * (
            consequent_count / self.transaction_count
        )
        lift = support / expected_support if expected_support > 0 else 0

        return support, confidence, lift

    def _generate_rules(
        self, frequent_set: Set[int], transactions: List[Set[int]]
    ) -> List[AssociationRule]:
        """
        Generate all possible rules from a frequent itemset.

        Args:
            frequent_set: Set of frequent items
            transactions: List of transaction sets

        Returns:
            List of association rules
        """
        rules = []

        # Generate all possible antecedent-consequent combinations
        for size in range(1, len(frequent_set)):
            for antecedent in combinations(frequent_set, size):
                antecedent = set(antecedent)
                consequent = frequent_set - antecedent

                # Calculate rule metrics
                support, confidence, lift = self._calculate_rule_metrics(
                    antecedent, consequent, transactions
                )

                # Check if rule meets thresholds
                if confidence >= self.min_confidence and lift >= self.min_lift:
                    rules.append(
                        AssociationRule(
                            antecedent=sorted(list(antecedent)),
                            consequent=sorted(list(consequent)),
                            support=support,
                            confidence=confidence,
                            lift=lift,
                            timestamp=datetime.now().isoformat(),
                        )
                    )

        return rules

    def generate_rules(
        self, frequent_sets: List[Dict], transactions: List[Set[int]]
    ) -> List[AssociationRule]:
        """
        Generate association rules from frequent itemsets.

        Args:
            frequent_sets: List of frequent itemsets
            transactions: List of transaction sets

        Returns:
            List of association rules
        """
        logger.info("Starting association rule generation...")

        self.transaction_count = len(transactions)
        self.rules = []

        # Generate rules for each frequent set
        for fs in frequent_sets:
            itemset = set(fs["items"])
            rules = self._generate_rules(itemset, transactions)
            self.rules.extend(rules)

        # Sort rules by lift and confidence
        self.rules.sort(key=lambda x: (x.lift, x.confidence), reverse=True)

        logger.info(f"Generated {len(self.rules)} association rules")
        return self.rules

    def _calculate_rule_quality(self, rule: AssociationRule) -> float:
        """
        Calculate a quality score for a rule.

        Args:
            rule: Association rule

        Returns:
            Quality score (0-1)
        """
        # Weighted combination of metrics
        weights = {"support": 0.3, "confidence": 0.4, "lift": 0.3}

        # Normalize lift (assuming max lift of 5)
        normalized_lift = min(rule.lift / 5.0, 1.0)

        return (
            weights["support"] * rule.support
            + weights["confidence"] * rule.confidence
            + weights["lift"] * normalized_lift
        )

    def rank_rules(self) -> List[Dict]:
        """
        Rank rules by quality score.

        Returns:
            List of ranked rules with quality scores
        """
        ranked_rules = []

        for rule in self.rules:
            quality_score = self._calculate_rule_quality(rule)
            ranked_rules.append(
                {
                    "antecedent": rule.antecedent,
                    "consequent": rule.consequent,
                    "support": rule.support,
                    "confidence": rule.confidence,
                    "lift": rule.lift,
                    "quality_score": quality_score,
                    "timestamp": rule.timestamp,
                }
            )

        # Sort by quality score
        ranked_rules.sort(key=lambda x: x["quality_score"], reverse=True)

        return ranked_rules

    def save_results(self, output_file: str):
        """
        Save association rules to JSON file.

        Args:
            output_file: Path to save results
        """
        import json
        from pathlib import Path

        # Rank rules and prepare output
        ranked_rules = self.rank_rules()
        results = {
            "association_rules": ranked_rules,
            "metadata": {
                "total_rules": len(ranked_rules),
                "min_confidence": self.min_confidence,
                "min_lift": self.min_lift,
                "timestamp": datetime.now().isoformat(),
            },
        }

        # Create output directory if needed
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to file
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_file}")

    def get_top_rules(self, n: int = 10) -> List[Dict]:
        """
        Get top N rules by quality score.

        Args:
            n: Number of rules to return

        Returns:
            List of top N rules
        """
        ranked_rules = self.rank_rules()
        return ranked_rules[:n]

    def analyze_rule_impact(self, rule: AssociationRule, transactions: List[Set[int]]) -> Dict:
        """
        Analyze the impact of a rule on historical data.

        Args:
            rule: Association rule to analyze
            transactions: List of transaction sets

        Returns:
            Dictionary containing impact analysis
        """
        antecedent = set(rule.antecedent)
        consequent = set(rule.consequent)

        # Count rule matches and misses
        matches = 0
        misses = 0
        for transaction in transactions:
            if antecedent.issubset(transaction):
                if consequent.issubset(transaction):
                    matches += 1
                else:
                    misses += 1

        # Calculate accuracy
        total = matches + misses
        accuracy = matches / total if total > 0 else 0

        return {
            "matches": matches,
            "misses": misses,
            "accuracy": accuracy,
            "total_opportunities": total,
        }
