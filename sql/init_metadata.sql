CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source_dataset TEXT NOT NULL,
    category TEXT NOT NULL,
    sample_type TEXT,
    input_text TEXT,
    reference_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id TEXT PRIMARY KEY,
    run_type TEXT NOT NULL,
    agent_framework TEXT NOT NULL,
    model_name TEXT,
    dataset_path TEXT NOT NULL,
    ontology_version TEXT,
    started_at TEXT,
    finished_at TEXT,
    status TEXT NOT NULL,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS profile_decisions (
    run_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    selected_profile TEXT NOT NULL,
    candidate_profiles_json TEXT,
    confidence REAL,
    mapping_policy TEXT,
    extension_policy TEXT,
    rationale_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, doc_id),
    FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

CREATE TABLE IF NOT EXISTS graph_ingestion (
    run_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    node_count INTEGER,
    edge_count INTEGER,
    graph_namespace TEXT,
    graph_uri TEXT,
    quality_summary_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, doc_id),
    FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

CREATE TABLE IF NOT EXISTS question_answers (
    run_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    doc_id TEXT,
    category TEXT,
    query_template TEXT,
    selected_profile TEXT,
    answer_text TEXT,
    answer_confidence REAL,
    supporting_edges_json TEXT,
    evaluation_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, question_id),
    FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    doc_id TEXT,
    question_id TEXT,
    artifact_type TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    metadata_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id)
);

