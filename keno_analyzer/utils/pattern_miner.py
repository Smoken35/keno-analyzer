#!/usr/bin/env python3
"""
Pattern Mining Module - Implements frequent itemset mining algorithms for Keno analysis.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrequentSet:
    """Data class for storing frequent itemset information."""

    items: List[int]
    support: float
    count: int
    timestamp: str


class PatternMiner:
    """Implements pattern mining algorithms for Keno number analysis."""

    def __init__(self, min_support: float = 0.02):
        """
        Initialize the pattern miner.

        Args:
            min_support: Minimum support threshold for frequent itemsets
        """
        self.min_support = min_support
        self.frequent_sets: List[FrequentSet] = []
        self.transaction_count = 0

    def _create_transaction_database(self, numbers: List[List[int]]) -> List[Set[int]]:
        """
        Convert Keno draws into a transaction database.

        Args:
            numbers: List of Keno draws

        Returns:
            List of sets representing transactions
        """
        return [set(draw) for draw in numbers]

    def _get_single_item_support(self, transactions: List[Set[int]]) -> Dict[int, int]:
        """
        Calculate support for single items.

        Args:
            transactions: List of transaction sets

        Returns:
            Dictionary mapping items to their support counts
        """
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        return dict(item_counts)

    def _filter_frequent_items(self, item_counts: Dict[int, int]) -> Set[int]:
        """
        Filter items that meet the minimum support threshold.

        Args:
            item_counts: Dictionary of item support counts

        Returns:
            Set of frequent items
        """
        return {
            item
            for item, count in item_counts.items()
            if count >= self.min_support * self.transaction_count
        }

    def _create_fp_tree(self, transactions: List[Set[int]], frequent_items: Set[int]) -> Dict:
        """
        Create an FP-Tree from transactions.

        Args:
            transactions: List of transaction sets
            frequent_items: Set of frequent items

        Returns:
            Dictionary representing the FP-Tree
        """
        tree = {"root": {}, "header": {}}

        for transaction in transactions:
            # Sort items by frequency
            sorted_items = sorted(
                [item for item in transaction if item in frequent_items],
                key=lambda x: item_counts[x],
                reverse=True,
            )

            # Add to tree
            current = tree["root"]
            for item in sorted_items:
                if item not in current:
                    current[item] = {"count": 1, "children": {}}
                    if item not in tree["header"]:
                        tree["header"][item] = []
                    tree["header"][item].append(current[item])
                else:
                    current[item]["count"] += 1
                current = current[item]["children"]

        return tree

    def _mine_fp_tree(self, tree: Dict, prefix: List[int] = None) -> List[FrequentSet]:
        """
        Mine frequent patterns from FP-Tree.

        Args:
            tree: FP-Tree dictionary
            prefix: Current prefix path

        Returns:
            List of frequent sets
        """
        if prefix is None:
            prefix = []

        patterns = []

        # Mine each item in header table
        for item, nodes in tree["header"].items():
            # Calculate support
            support = sum(node["count"] for node in nodes)
            if support >= self.min_support * self.transaction_count:
                # Create new pattern
                new_pattern = prefix + [item]
                patterns.append(
                    FrequentSet(
                        items=new_pattern,
                        support=support / self.transaction_count,
                        count=support,
                        timestamp=datetime.now().isoformat(),
                    )
                )

                # Create conditional tree
                cond_tree = self._create_conditional_tree(nodes)
                if cond_tree["root"]:
                    # Recursively mine conditional tree
                    patterns.extend(self._mine_fp_tree(cond_tree, new_pattern))

        return patterns

    def _create_conditional_tree(self, nodes: List[Dict]) -> Dict:
        """
        Create conditional FP-Tree for given nodes.

        Args:
            nodes: List of nodes from header table

        Returns:
            Dictionary representing conditional FP-Tree
        """
        cond_tree = {"root": {}, "header": {}}

        # Collect prefix paths
        prefix_paths = []
        for node in nodes:
            path = []
            current = node
            while "parent" in current:
                path.append(current["item"])
                current = current["parent"]
            if path:
                prefix_paths.append((path, node["count"]))

        # Build conditional tree
        for path, count in prefix_paths:
            current = cond_tree["root"]
            for item in path:
                if item not in current:
                    current[item] = {"count": count, "children": {}}
                    if item not in cond_tree["header"]:
                        cond_tree["header"][item] = []
                    cond_tree["header"][item].append(current[item])
                else:
                    current[item]["count"] += count
                current = current[item]["children"]

        return cond_tree

    def mine_patterns(
        self, numbers: List[List[int]], algorithm: str = "fp-growth"
    ) -> List[FrequentSet]:
        """
        Mine frequent patterns from Keno draws.

        Args:
            numbers: List of Keno draws
            algorithm: Mining algorithm to use ('fp-growth' or 'apriori')

        Returns:
            List of frequent sets
        """
        logger.info(f"Starting pattern mining with {len(numbers)} draws...")

        # Create transaction database
        transactions = self._create_transaction_database(numbers)
        self.transaction_count = len(transactions)

        if algorithm == "fp-growth":
            # Calculate single item support
            item_counts = self._get_single_item_support(transactions)
            frequent_items = self._filter_frequent_items(item_counts)

            # Create and mine FP-Tree
            tree = self._create_fp_tree(transactions, frequent_items)
            self.frequent_sets = self._mine_fp_tree(tree)

        else:  # Apriori algorithm
            self.frequent_sets = self._mine_apriori(transactions)

        # Sort by support
        self.frequent_sets.sort(key=lambda x: x.support, reverse=True)

        logger.info(f"Found {len(self.frequent_sets)} frequent patterns")
        return self.frequent_sets

    def _mine_apriori(self, transactions: List[Set[int]]) -> List[FrequentSet]:
        """
        Mine frequent patterns using Apriori algorithm.

        Args:
            transactions: List of transaction sets

        Returns:
            List of frequent sets
        """
        # Initialize with single items
        k = 1
        current_sets = [{item} for item in range(1, 81)]
        frequent_sets = []

        while current_sets:
            # Count support for current sets
            set_counts = defaultdict(int)
            for transaction in transactions:
                for itemset in current_sets:
                    if itemset.issubset(transaction):
                        set_counts[frozenset(itemset)] += 1

            # Filter frequent sets
            frequent_k_sets = []
            for itemset, count in set_counts.items():
                support = count / self.transaction_count
                if support >= self.min_support:
                    frequent_k_sets.append(
                        FrequentSet(
                            items=sorted(list(itemset)),
                            support=support,
                            count=count,
                            timestamp=datetime.now().isoformat(),
                        )
                    )

            frequent_sets.extend(frequent_k_sets)

            # Generate next level candidates
            k += 1
            current_sets = self._generate_candidates(frequent_k_sets, k)

        return frequent_sets

    def _generate_candidates(self, frequent_sets: List[FrequentSet], k: int) -> List[Set[int]]:
        """
        Generate candidate sets for next level.

        Args:
            frequent_sets: List of frequent sets from current level
            k: Current level number

        Returns:
            List of candidate sets
        """
        candidates = []
        for i, set1 in enumerate(frequent_sets):
            for set2 in frequent_sets[i + 1 :]:
                # Join sets if they share k-2 items
                union = set(set1.items) | set(set2.items)
                if len(union) == k:
                    candidates.append(union)

        return candidates

    def save_results(self, output_file: str):
        """
        Save mining results to JSON file.

        Args:
            output_file: Path to save results
        """
        import json
        from pathlib import Path

        # Convert to serializable format
        results = {
            "frequent_sets": [
                {
                    "items": fs.items,
                    "support": fs.support,
                    "count": fs.count,
                    "timestamp": fs.timestamp,
                }
                for fs in self.frequent_sets
            ],
            "metadata": {
                "total_transactions": self.transaction_count,
                "min_support": self.min_support,
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
