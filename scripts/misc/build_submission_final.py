#!/usr/bin/env python3
"""
Build the final submission.zip with everything needed for thesis defense and cross-checking.
"""

import os
import zipfile
import glob

REPO = r"C:\Users\zamin\OneDrive\Desktop\Nazim_thesis\ml_surrogates_for_agent_based_transport_models"
OUT_ZIP = os.path.join(REPO, "submission_final.zip")

print("Building final submission package...")

added = set()

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:

    def add(src, arcname):
        if arcname not in added and os.path.exists(src):
            zf.write(src, arcname)
            added.add(arcname)
            return True
        return False

    # 1. THESIS PDF
    main_pdf = os.path.join(REPO, "thesis", "latex_tum_official", "main.pdf")
    if add(main_pdf, "thesis/main.pdf"):
        print("  + thesis/main.pdf")

    # 2. PRESENTATION
    pptx = os.path.join(REPO, "thesis_presentation_final.pptx")
    if add(pptx, "presentation/thesis_presentation_final.pptx"):
        print("  + presentation/thesis_presentation_final.pptx")

    # 3. HD PLOTS (PNG + PDF)
    plots_dir = os.path.join(REPO, "docs", "hd_plots")
    for f in sorted(glob.glob(os.path.join(plots_dir, "*"))):
        bn = os.path.basename(f)
        if add(f, f"hd_plots/{bn}"):
            print(f"  + hd_plots/{bn}")

    # 4. JSON ARTIFACTS (phase3_results)
    phase3 = os.path.join(REPO, "docs", "verified", "phase3_results")
    for f in sorted(glob.glob(os.path.join(phase3, "*.json"))):
        bn = os.path.basename(f)
        if add(f, f"verified_data/phase3_results/{bn}"):
            print(f"  + verified_data/phase3_results/{bn}")

    # 5. UQ RESULTS (original JSON + NPZ)
    uq_dir = os.path.join(
        REPO,
        "data",
        "TR-C_Benchmarks",
        "point_net_transf_gat_8th_trial_lower_dropout",
        "uq_results",
    )
    for f in sorted(glob.glob(os.path.join(uq_dir, "*.json"))):
        bn = os.path.basename(f)
        if add(f, f"data/uq_results/{bn}"):
            print(f"  + data/uq_results/{bn}")

    # NPZ files (large but essential for reproducibility)
    for f in sorted(glob.glob(os.path.join(uq_dir, "*.npz"))):
        bn = os.path.basename(f)
        if add(f, f"data/uq_results/{bn}"):
            print(f"  + data/uq_results/{bn}")

    # 6. Test evaluation
    tec = os.path.join(
        REPO,
        "data",
        "TR-C_Benchmarks",
        "point_net_transf_gat_8th_trial_lower_dropout",
        "test_evaluation_complete.json",
    )
    if add(tec, "data/test_evaluation_complete.json"):
        print("  + data/test_evaluation_complete.json")

    # 7. KEY SCRIPTS
    scripts_to_include = [
        "scripts/generate_all_hd_plots.py",
        "scripts/generate_presentation_final.py",
        "scripts/compute_crps.py",
        "scripts/compute_pit.py",
        "scripts/compute_pit_after_tempscaling.py",
        "scripts/compute_winkler.py",
        "scripts/compute_nll.py",
        "scripts/compute_bootstrap_ci.py",
        "scripts/regenerate_fig58_s30.py",
        "scripts/run_s_convergence.py",
        "scripts/verify_all_numbers_final.py",
    ]
    for s in scripts_to_include:
        src = os.path.join(REPO, s)
        if add(src, s):
            print(f"  + {s}")

    # Model code
    model_src = os.path.join(
        REPO, "scripts", "gnn", "models", "point_net_transf_gat.py"
    )
    if add(model_src, "scripts/gnn/models/point_net_transf_gat.py"):
        print("  + scripts/gnn/models/point_net_transf_gat.py")

    mc_predict = os.path.join(REPO, "scripts", "gnn", "help_functions.py")
    if add(mc_predict, "scripts/gnn/help_functions.py"):
        print("  + scripts/gnn/help_functions.py")

    # 8. LaTeX source
    latex_dir = os.path.join(REPO, "thesis", "latex_tum_official")
    for f in sorted(glob.glob(os.path.join(latex_dir, "*.tex"))):
        bn = os.path.basename(f)
        if add(f, f"latex_source/{bn}"):
            print(f"  + latex_source/{bn}")

    for f in sorted(glob.glob(os.path.join(latex_dir, "*.bib"))):
        bn = os.path.basename(f)
        if add(f, f"latex_source/{bn}"):
            print(f"  + latex_source/{bn}")

    # Chapter files
    ch_dir = os.path.join(latex_dir, "chapters")
    for f in sorted(glob.glob(os.path.join(ch_dir, "*.tex"))):
        bn = os.path.basename(f)
        if add(f, f"latex_source/chapters/{bn}"):
            print(f"  + latex_source/chapters/{bn}")

    pages_dir = os.path.join(latex_dir, "pages")
    for f in sorted(glob.glob(os.path.join(pages_dir, "*.tex"))):
        bn = os.path.basename(f)
        if add(f, f"latex_source/pages/{bn}"):
            print(f"  + latex_source/pages/{bn}")

    # 9. Meeting prep
    meeting = os.path.join(REPO, "docs", "meeting_prep_answer_cards.md")
    if add(meeting, "docs/meeting_prep_answer_cards.md"):
        print("  + docs/meeting_prep_answer_cards.md")

    # MANIFEST
    manifest = os.path.join(latex_dir, "MANIFEST.md")
    if add(manifest, "MANIFEST.md"):
        print("  + MANIFEST.md")

print(f"\nTotal files: {len(added)}")
print(f"Output: {OUT_ZIP}")
print(f"Size: {os.path.getsize(OUT_ZIP) / (1024 * 1024):.1f} MB")
