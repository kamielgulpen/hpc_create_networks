"""
ASNU - Aggregated Social Network Unfolder
==========================================

A Python package for generating large-scale population-based networks with
realistic community structure, preferential attachment, reciprocity, and
transitivity.

Main Functions
--------------
generate : Generate a complete network from population and interaction data

Classes
-------
NetworkXGraph : Graph wrapper with metadata for network generation
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from asnu.core.generate import generate
from asnu.core.graph import NetworkXGraph
from asnu.core.community import create_communities, create_hierarchical_community_file
from asnu.core.utils import check_group_interactions, plot_group_interactions

__all__ = ['generate', 'NetworkXGraph', 'create_communities', 'create_hierarchical_community_file',
           'check_group_interactions', 'plot_group_interactions']
