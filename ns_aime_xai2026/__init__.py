"""NS-AIME: Rule-Guided Re-Constraint of Approximate Inverse Explanations."""

from .core import NSAIME
from .utils import induce_rules, compute_metrics, calculate_metrics
from .visualization import plot_logic_graph, plot_dual_report
from .interactive import render_logic_graph_html

__version__ = "0.2.0"
__all__ = [
    "NSAIME",
    "induce_rules",
    "compute_metrics",
    "calculate_metrics",
    "plot_logic_graph",
    "plot_dual_report",
    "render_logic_graph_html",
]
