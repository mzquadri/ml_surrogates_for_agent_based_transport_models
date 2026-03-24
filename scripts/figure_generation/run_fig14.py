"""Generate only Fig 5.4 (fig14_conformal_workflow)."""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_new_figures import fig14_conformal_workflow

fig14_conformal_workflow()
print("Done generating fig14_conformal_workflow")
