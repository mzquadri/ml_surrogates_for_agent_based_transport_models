"""
Run only fig3_feature_distributions and fig10_node_vs_graph from generate_all_thesis_figures.py.
Usage: conda run -n thesis-env python scripts/run_fig31_fig32_redesign.py
"""

import sys, os

# Add figures directory to path
fig_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, fig_dir)

# Import and run the two functions
from generate_all_thesis_figures import fig3_feature_distributions, fig10_node_vs_graph

print("=" * 60)
print("Generating Fig 3.1 (feature distributions) ...")
print("=" * 60)
fig3_feature_distributions()

print()
print("=" * 60)
print("Generating Fig 3.2 (node vs graph) ...")
print("=" * 60)
fig10_node_vs_graph()

print()
print("DONE. Both figures regenerated.")
