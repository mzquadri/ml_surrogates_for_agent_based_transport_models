"""Immutable source and submission contracts for the thesis audit."""

AUDIT_SOURCE_COMMIT = "fdb4ef0c9c736576ae34d5e331d8b66a7a6d877a"
SUBMITTED_ARTIFACT_COMMIT = "4b95a3d8aca5929bb88b84bb7f7ae86c48e2f428"
SUBMITTED_PDF_BYTES = 674_395
SUBMITTED_PDF_SHA256 = (
    "0ac5309d060cda53d82a05cc837136fe853e7f9dcbabd2f4fb4b4282a39bc97e"
)
SUBMITTED_DOCUMENT_FILE_COUNT = 40
SUBMITTED_DOCUMENT_GIT_TREE = "f104db730eb1c8d228d913fde6545599da7795d5"
LOCAL_TEST_LOADER_PATH = (
    "data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/"
    "data_created_during_training/test_dl.pt"
)
LOCAL_TEST_LOADER_BYTES = 311_059_407
LOCAL_TEST_LOADER_SHA256 = (
    "0850b5ccf0331590d7cab43293d2c5816ed451f44e507b05b0649fd79f003ebd"
)

# name: (repository-relative path, bytes, SHA-256)
SOURCE_ARTIFACTS = {
    "t8_mc": (
        (
            "results/predictions/point_net_transf_gat_8th_trial_lower_dropout/"
            "uq_results/mc_dropout_full_100graphs_mc30.npz"
        ),
        27_989_231,
        "ae9460fb5707a93728f441db3480668db2c3ab8e59da2da78e749146d245853a",
    ),
    "t7_mc": (
        (
            "results/predictions/point_net_transf_gat_7th_trial_80_10_10_split/"
            "uq_results/mc_dropout_full_100graphs_mc30.npz"
        ),
        27_855_316,
        "55bac7f932c9f513ce81e234a4e2bc8596ed02f8b578078354dfec2f8302611e",
    ),
    "t8_deterministic": (
        (
            "results/predictions/point_net_transf_gat_8th_trial_lower_dropout/"
            "test_predictions.npz"
        ),
        25_308_522,
        "ee4c61327c521c150dcce6416474054b08529246797f64c42fe1600bd65ef327",
    ),
    "deep_ensemble": (
        "results/predictions/deep_ensemble_results/deep_ensemble_predictions.npz",
        101_233_050,
        "0159257da5f2dcc5a97e90b52467656c0c4eb8f5462c94f4fded4e1fcbbc7d3e",
    ),
    "t10_cqr": (
        "results/predictions/point_net_transf_gat_10th_trial_cqr/val_predictions.npz",
        37_962_748,
        "088e6fc37374b447f9f5430c93e7a4f793e1d4bb1b130c91bf449913cde75366",
    ),
    "t11_cqr": (
        (
            "results/predictions/point_net_transf_gat_11th_trial_cqr_frozen/"
            "val_predictions.npz"
        ),
        37_962_748,
        "eb5caf4a944be45c40b3bb7d92896e5bf15abb9dc65ffc0a55f5ac7b61c6788e",
    ),
    "temperature": (
        "results/temperature_scaling_t8.json",
        2_635,
        "e13287ca5e92c602b5ee832b51d35e40e92e1342fc6673f6b8f8fa20d0b18af1",
    ),
    "conditional_conformal": (
        "results/conformal_conditional_coverage_t8.json",
        4_609,
        "6f0ae1576ae5dc061c6a0febb742408813b039f73d115bce0e975a6442e1d157",
    ),
    "t7_point_metrics": (
        (
            "results/trials/point_net_transf_gat_7th_trial_80_10_10_split/"
            "test_evaluation_complete.json"
        ),
        855,
        "87eecd8240579771ef85c05537706e95fc63a43fd708d9291e90a13072cfb0d5",
    ),
    "t8_point_metrics": (
        (
            "results/trials/point_net_transf_gat_8th_trial_lower_dropout/"
            "test_evaluation_complete.json"
        ),
        1_040,
        "b58041004f05a0e6c08ddb9d08243485fad19a36d54045fa8db2c91bf8ac4903",
    ),
    "deep_ensemble_metrics": (
        "results/trials/deep_ensemble_results/ensemble_metrics.json",
        2_102,
        "7455bea8c687054217f59425f498a1c0d06fa876942a88c74134247dcd3eff48",
    ),
    "t8_verified_metrics": (
        "results/trials/uq_verification_run/mc_dropout_verified_metrics.json",
        1_102,
        "1e69d53847b684a4cf74aac368526c4967584ddfd5e1fb54ae9d8aded7cf27a3",
    ),
    "t8_verified_predictions": (
        "results/predictions/uq_verification_run/mc_dropout_verified.npz",
        37_962_780,
        "d76282e9106032c458703e6a50bd334394244876afe6e77219cd704460d37901",
    ),
    "t10_metrics": (
        (
            "results/trials/point_net_transf_gat_10th_trial_cqr/cqr_results/"
            "cqr_metrics.json"
        ),
        1_454,
        "ac18d888228ac59f335d6b119a69f5556a93f66624faeb8b45f3c38e0f327e8d",
    ),
    "t11_metrics": (
        (
            "results/trials/point_net_transf_gat_11th_trial_cqr_frozen/cqr_results/"
            "cqr_metrics.json"
        ),
        1_450,
        "0d2190f509518781d581047b2b0f9482a111b17c6f89c5669646d08a943ff717",
    ),
}
