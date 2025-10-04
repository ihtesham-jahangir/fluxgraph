# File: fluxgraph/analytics/metrics_collector.py
import logging
from collections import defaultdict
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects custom metrics such as counters, gauges, and summaries (histograms).
    Supports labeled metrics for more granular tracking.
    This can be used to collect application-specific metrics beyond agent performance,
    such as API hit counts, error types, or resource usage.
    """

    def __init__(self):
        # Simple counters (no labels)
        self.counters: Dict[str, int] = defaultdict(int)
        
        # Labeled counters (e.g., counters with tags like status=200)
        self.labeled_counters: Dict[str, Dict[frozenset, int]] = defaultdict(lambda: defaultdict(int))
        
        # Gauges (current values, like memory usage)
        self.gauges: Dict[str, float] = {}
        
        # Labeled gauges
        self.labeled_gauges: Dict[str, Dict[frozenset, float]] = defaultdict(lambda: defaultdict(float))
        
        # Summaries (for observations like response times, to compute avg/min/max)
        self.summaries: Dict[str, List[float]] = defaultdict(list)
        
        # Labeled summaries
        self.labeled_summaries: Dict[str, Dict[frozenset, List[float]]] = defaultdict(lambda: defaultdict(list))

    def increment(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """
        Increment a counter metric.
        
        :param name: Name of the counter
        :param value: Amount to increment by (default 1)
        :param labels: Optional dictionary of labels (e.g., {'status': '200'})
        """
        if labels:
            label_key = frozenset(labels.items())
            self.labeled_counters[name][label_key] += value
        else:
            self.counters[name] += value
        logger.debug(f"Incremented counter '{name}' by {value} with labels {labels}")

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Set a gauge metric to a specific value.
        
        :param name: Name of the gauge
        :param value: The value to set
        :param labels: Optional dictionary of labels
        """
        if labels:
            label_key = frozenset(labels.items())
            self.labeled_gauges[name][label_key] = value
        else:
            self.gauges[name] = value
        logger.debug(f"Set gauge '{name}' to {value} with labels {labels}")

    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Observe a value for a summary metric (e.g., response time).
        
        :param name: Name of the summary
        :param value: The value to observe
        :param labels: Optional dictionary of labels
        """
        if labels:
            label_key = frozenset(labels.items())
            self.labeled_summaries[name][label_key].append(value)
        else:
            self.summaries[name].append(value)
        logger.debug(f"Observed '{name}' with value {value} and labels {labels}")

    def _summarize(self, values: List[float]) -> Dict[str, Any]:
        """Compute summary statistics for a list of values."""
        if not values:
            return {
                'count': 0,
                'sum': 0.0,
                'average': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        return {
            'count': len(values),
            'sum': sum(values),
            'average': sum(values) / len(values),
            'min': min(values),
            'max': max(values)
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all collected metrics in a serializable dictionary format.
        """
        metrics = {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'summaries': {name: self._summarize(values) for name, values in self.summaries.items()},
            'labeled_counters': {
                name: {str(dict(label_items)): value for label_items, value in labels_dict.items()}
                for name, labels_dict in self.labeled_counters.items()
            },
            'labeled_gauges': {
                name: {str(dict(label_items)): value for label_items, value in labels_dict.items()}
                for name, labels_dict in self.labeled_gauges.items()
            },
            'labeled_summaries': {
                name: {str(dict(label_items)): self._summarize(values) for label_items, values in labels_dict.items()}
                for name, labels_dict in self.labeled_summaries.items()
            }
        }
        return metrics

    def clear_metrics(self):
        """Clear all collected metrics."""
        self.counters.clear()
        self.labeled_counters.clear()
        self.gauges.clear()
        self.labeled_gauges.clear()
        self.summaries.clear()
        self.labeled_summaries.clear()
        logger.info("Cleared all custom metrics")