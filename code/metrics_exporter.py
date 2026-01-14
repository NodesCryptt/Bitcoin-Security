"""
CP1 Prometheus Metrics Exporter
===============================
Exposes CP1 metrics for Prometheus scraping.

Metrics:
- cp1_tx_ingested_total: Total transactions processed
- cp1_accept_count: ACCEPT decisions
- cp1_flag_count: FLAG decisions  
- cp1_reject_count: REJECT decisions
- cp1_consensus_reject_count: Consensus validation failures
- cp1_score_distribution: ML score histogram
- cp1_infer_latency_seconds: Inference latency histogram
- cp1_model_version: Current model version
"""

import logging
import os
import time
from typing import Dict, Any, Optional
from functools import wraps
import threading

logger = logging.getLogger(__name__)

# Try to import prometheus_client
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info,
        start_http_server, REGISTRY, generate_latest
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed. Metrics disabled.")


class MetricsExporter:
    """
    Prometheus metrics exporter for CP1.
    
    Usage:
        metrics = MetricsExporter(port=8000)
        metrics.start_server()
        
        # Record metrics
        metrics.increment("cp1_tx_ingested_total")
        metrics.observe("cp1_score_distribution", 0.72)
        
        # Use as decorator
        @metrics.latency("cp1_infer_latency_seconds")
        def inference():
            ...
    """
    
    def __init__(self, port: int = 8000, model_version: str = "v1"):
        """
        Initialize metrics exporter.
        
        Args:
            port: Port to expose /metrics endpoint
            model_version: Current model version string
        """
        self.port = port
        self.model_version = model_version
        self._server_started = False
        
        if PROMETHEUS_AVAILABLE:
            self._init_metrics()
        else:
            self._metrics = {}
            self._counters = {}
    
    def _init_metrics(self):
        """Initialize Prometheus metric objects."""
        
        # Counters
        self.tx_ingested = Counter(
            "cp1_tx_ingested_total",
            "Total transactions ingested by CP1"
        )
        
        self.accept_count = Counter(
            "cp1_accept_count",
            "Number of ACCEPT decisions"
        )
        
        self.flag_count = Counter(
            "cp1_flag_count",
            "Number of FLAG decisions"
        )
        
        self.reject_count = Counter(
            "cp1_reject_count",
            "Number of REJECT decisions",
            ["reason"]  # reason: ml, consensus
        )
        
        self.consensus_reject_count = Counter(
            "cp1_consensus_reject_count",
            "Number of consensus validation failures",
            ["reason"]  # reason: decode_error, invalid_structure, invalid_script
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            "cp1_cache_hits_total",
            "Total cache hits",
            ["cache_type"]  # utxo, addr, decoded
        )
        
        self.cache_misses = Counter(
            "cp1_cache_misses_total",
            "Total cache misses",
            ["cache_type"]
        )
        
        # SHAP metrics
        self.shap_explanations = Counter(
            "cp1_shap_explanations_total",
            "Total SHAP explanations generated"
        )
        
        self.shap_latency = Histogram(
            "cp1_shap_latency_seconds",
            "SHAP explanation generation latency",
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
        )
        
        # Double-spend detection
        self.double_spend_detected = Counter(
            "cp1_double_spend_detected_total",
            "Total double-spend attempts detected"
        )
        
        # Histograms
        self.score_histogram = Histogram(
            "cp1_score_distribution",
            "Distribution of ML scores",
            buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
        )
        
        self.latency_histogram = Histogram(
            "cp1_infer_latency_seconds",
            "Inference latency in seconds",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        )
        
        self.feature_extraction_latency = Histogram(
            "cp1_feature_extraction_seconds",
            "Feature extraction latency in seconds",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
        )
        
        # Gauges
        self.model_version_gauge = Gauge(
            "cp1_model_version_info",
            "Current model version",
            ["version"]
        )
        self.model_version_gauge.labels(version=self.model_version).set(1)
        
        self.queue_size = Gauge(
            "cp1_queue_size",
            "Current processing queue size"
        )
        
        self.cache_size = Gauge(
            "cp1_cache_size",
            "Current cache size",
            ["cache_type"]  # addr, tx, cluster
        )
        
        # Info
        self.build_info = Info(
            "cp1_build",
            "CP1 build information"
        )
        self.build_info.info({
            "version": self.model_version,
            "component": "cp1_runtime"
        })
    
    def start_server(self):
        """Start the HTTP server for Prometheus scraping."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Cannot start metrics server - prometheus_client not installed")
            return
        
        if self._server_started:
            logger.warning("Metrics server already started")
            return
        
        try:
            start_http_server(self.port)
            self._server_started = True
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    # =========================================================================
    # METRIC RECORDING METHODS
    # =========================================================================
    
    def increment(self, metric_name: str, labels: Dict[str, str] = None, value: float = 1):
        """
        Increment a counter metric.
        
        Args:
            metric_name: Name of the metric
            labels: Label values (optional)
            value: Amount to increment (default 1)
        """
        if not PROMETHEUS_AVAILABLE:
            self._counters[metric_name] = self._counters.get(metric_name, 0) + value
            return
        
        metric = self._get_counter(metric_name)
        if metric:
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)
    
    def observe(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """
        Observe a histogram value.
        
        Args:
            metric_name: Name of the metric
            value: Value to observe
            labels: Label values (optional)
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        histogram = self._get_histogram(metric_name)
        if histogram:
            if labels:
                histogram.labels(**labels).observe(value)
            else:
                histogram.observe(value)
    
    def set_gauge(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """
        Set a gauge value.
        
        Args:
            metric_name: Name of the metric
            value: Value to set
            labels: Label values (optional)
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        gauge = self._get_gauge(metric_name)
        if gauge:
            if labels:
                gauge.labels(**labels).set(value)
            else:
                gauge.set(value)
    
    def _get_counter(self, name: str):
        """Get counter by name."""
        mapping = {
            "cp1_tx_ingested_total": self.tx_ingested,
            "cp1_accept_count": self.accept_count,
            "cp1_flag_count": self.flag_count,
            "cp1_reject_count": self.reject_count,
            "cp1_consensus_reject_count": self.consensus_reject_count,
            "cp1_cache_hits_total": self.cache_hits,
            "cp1_cache_misses_total": self.cache_misses,
            "cp1_shap_explanations_total": self.shap_explanations,
            "cp1_double_spend_detected_total": self.double_spend_detected,
        }
        return mapping.get(name)
    
    def _get_histogram(self, name: str):
        """Get histogram by name."""
        mapping = {
            "cp1_score_distribution": self.score_histogram,
            "cp1_infer_latency_seconds": self.latency_histogram,
            "cp1_feature_extraction_seconds": self.feature_extraction_latency,
            "cp1_shap_latency_seconds": self.shap_latency,
        }
        return mapping.get(name)
    
    def _get_gauge(self, name: str):
        """Get gauge by name."""
        mapping = {
            "cp1_queue_size": self.queue_size,
            "cp1_cache_size": self.cache_size,
        }
        return mapping.get(name)
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    def record_tx_ingested(self):
        """Record a transaction ingestion."""
        self.increment("cp1_tx_ingested_total")
    
    def record_decision(self, decision: str, reason: str = "ml"):
        """
        Record a decision.
        
        Args:
            decision: ACCEPT, FLAG, or REJECT (case-insensitive)
            reason: Reason for reject (ml, consensus)
        """
        decision_upper = decision.upper()
        if decision_upper == "ACCEPT":
            self.increment("cp1_accept_count")
        elif decision_upper == "FLAG":
            self.increment("cp1_flag_count")
        elif decision_upper == "REJECT":
            self.increment("cp1_reject_count", {"reason": reason})
    
    def record_consensus_failure(self, reason: str):
        """
        Record a consensus validation failure.
        
        Args:
            reason: Failure reason (decode_error, invalid_structure, invalid_script)
        """
        self.increment("cp1_consensus_reject_count", {"reason": reason})
    
    def record_score(self, score: float):
        """Record an ML score."""
        self.observe("cp1_score_distribution", score)
    
    def record_latency(self, latency_seconds: float):
        """Record inference latency."""
        self.observe("cp1_infer_latency_seconds", latency_seconds)
    
    def record_cache_hit(self, cache_type: str = "utxo"):
        """Record a cache hit."""
        self.increment("cp1_cache_hits_total", {"cache_type": cache_type})
    
    def record_cache_miss(self, cache_type: str = "utxo"):
        """Record a cache miss."""
        self.increment("cp1_cache_misses_total", {"cache_type": cache_type})
    
    def record_shap_explanation(self, latency_seconds: float = None):
        """Record SHAP explanation generation."""
        self.increment("cp1_shap_explanations_total")
        if latency_seconds is not None:
            self.observe("cp1_shap_latency_seconds", latency_seconds)
    
    def record_double_spend(self):
        """Record double-spend detection."""
        self.increment("cp1_double_spend_detected_total")
    
    # =========================================================================
    # DECORATORS
    # =========================================================================
    
    def latency(self, metric_name: str):
        """
        Decorator to measure function latency.
        
        Usage:
            @metrics.latency("cp1_infer_latency_seconds")
            def inference():
                ...
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    elapsed = time.perf_counter() - start
                    self.observe(metric_name, elapsed)
            return wrapper
        return decorator
    
    def count(self, metric_name: str, labels: Dict[str, str] = None):
        """
        Decorator to count function calls.
        
        Usage:
            @metrics.count("cp1_tx_ingested_total")
            def process():
                ...
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.increment(metric_name, labels)
                return func(*args, **kwargs)
            return wrapper
        return decorator


# Default metrics instance
_default_metrics: Optional[MetricsExporter] = None


def get_metrics(port: int = 8000) -> MetricsExporter:
    """Get or create default metrics exporter."""
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = MetricsExporter(port=port)
    return _default_metrics


def start_metrics_server(port: int = 8000):
    """Start the default metrics server."""
    metrics = get_metrics(port)
    metrics.start_server()
    return metrics
