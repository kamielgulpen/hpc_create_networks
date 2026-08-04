"""
ASNU - Aggregated Social Network Unfolder
==========================================

A Python package for generating large-scale population-based networks with
realistic community structure, reciprocity, and transitivity.

Main Functions
--------------
generate          : Generate a complete network from population and interaction data
create_communities: Derive a community partition from population + interaction data
clone_communities : Scale an existing community partition up to a larger scale

Classes
-------
NetworkXGraph : Graph wrapper with metadata for network generation
"""

__version__ = "0.1.0"
__author__ = "Kamiel Gulpen"

from asnu.core.generate import generate
from asnu.core.graph import NetworkXGraph
from asnu.core.community import create_communities
from asnu.core.clone_communities import clone_communities

__all__ = ['generate', 'NetworkXGraph', 'create_communities', 'clone_communities']