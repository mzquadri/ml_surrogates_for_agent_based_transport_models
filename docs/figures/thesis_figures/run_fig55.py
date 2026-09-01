"""Generate only Fig 5.5 (fig3_conformal_coverage)."""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_all_thesis_figures import fig3_conformal_coverage

fig3_conformal_coverage()
print("Done generating fig3_conformal_coverage")
