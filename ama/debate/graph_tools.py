"""
Graph query tools for the debate agents.

Two sets of tools — one per logical database inside one DozerDB instance:

  RDF tools:  query_rdf_cypher, search_rdf_entities, get_rdf_entity_context
  LPG tools:  query_lpg_cypher, search_lpg_nodes, get_lpg_node_neighbors

Each tool is created via OpenAI Agents SDK's @function_tool decorator.
The underlying Neo4j driver is initialized explicitly via init_connections()
before any agent runs.

Graph schemas:

  RDF database:
    (:RDFNode {uri, _experiment, _example_id})
    -[:RDF_PREDICATE {predicate, _experiment, _example_id}]->
    (:RDFNode)

  LPG database:
    (:<Label> {_node_id, _experiment, _example_id, ...properties})
    -[:<TYPE> {_experiment, _example_id, ...properties}]->
    (:<Label>)
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from neo4j import GraphDatabase

from agents import function_tool  # OpenAI Agents SDK

# ---------------------------------------------------------------------------
# Module-level Neo4j connection state — initialized by init_connections()
# ---------------------------------------------------------------------------

_driver: Any = None
_rdf_database: str = "finderrdf"
_lpg_database: str = "finderlpg"
_active_experiment: str = ""
_active_example_id: str = ""


def get_database_bindings() -> dict[str, str]:
    """Return the current logical database names for RDF and LPG."""
    return {
        "rdf": _rdf_database,
        "lpg": _lpg_database,
    }


def get_active_scope() -> dict[str, str]:
    """Return the currently active debate scope for graph reads."""
    return {
        "experiment": _active_experiment,
        "example_id": _active_example_id,
    }


def set_active_scope(*, experiment_name: str = "", example_id: str = "") -> None:
    """Set the current graph-read scope for debate tools."""
    global _active_experiment, _active_example_id
    _active_experiment = experiment_name
    _active_example_id = example_id


def clear_active_scope() -> None:
    """Clear the current graph-read scope."""
    set_active_scope()


def init_connections(
    *,
    uri: Optional[str] = None,
    user: str = "neo4j",
    password: Optional[str] = None,
    rdf_database: Optional[str] = None,
    lpg_database: Optional[str] = None,
) -> None:
    """Initialize one Neo4j driver and remember which databases map to RDF/LPG."""
    global _driver, _rdf_database, _lpg_database

    _driver = GraphDatabase.driver(
        uri or os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(user, password or os.getenv("NEO4J_PASSWORD", "ama-experiment-2026")),
    )
    _rdf_database = rdf_database or os.getenv("NEO4J_DATABASE_RDF", "finderrdf")
    _lpg_database = lpg_database or os.getenv("NEO4J_DATABASE_LPG", "finderlpg")


def close_connections() -> None:
    """Close the shared Neo4j driver."""
    global _driver
    if _driver:
        _driver.close()
        _driver = None


def _run_cypher(driver: Any, database: str, cypher: str, params: Optional[dict] = None) -> str:
    """Execute Cypher and return JSON string of results."""
    with driver.session(database=database) as session:
        result = session.run(cypher, parameters=params or {})
        records = [dict(record) for record in result]
        return json.dumps(records, default=str, ensure_ascii=False)


def _scope_params(params: Optional[dict] = None) -> dict[str, Any]:
    scoped = dict(params or {})
    scoped["active_experiment"] = _active_experiment
    scoped["active_example_id"] = _active_example_id
    return scoped


def _scope_clause(alias: str = "n") -> str:
    return (
        f"($active_experiment = '' OR coalesce({alias}._experiment, '') = $active_experiment)"
        f" AND ($active_example_id = '' OR coalesce({alias}._example_id, '') = $active_example_id)"
    )


def _scoped_freeform_guard(cypher: str) -> Optional[str]:
    if not (_active_experiment or _active_example_id):
        return None
    lowered = cypher.lower()
    mentions_scope = (
        "_example_id" in lowered
        or "_experiment" in lowered
        or "uri" in lowered
        or "_node_id" in lowered
    )


def _run_records(driver: Any, database: str, cypher: str, params: Optional[dict] = None) -> list[dict[str, Any]]:
    with driver.session(database=database) as session:
        result = session.run(cypher, parameters=params or {})
        return [dict(record) for record in result]


def get_rdf_schema_card() -> dict[str, Any]:
    """Summarize the observed RDF graph shape for the current scope."""
    if not _driver:
        return {"labels": [], "predicates": [], "scope": get_active_scope()}
    labels = _run_records(
        _driver,
        _rdf_database,
        f"""
        MATCH (n)
        WHERE {_scope_clause('n')}
        UNWIND labels(n) AS label
        RETURN DISTINCT label
        ORDER BY label
        LIMIT 12
        """,
        _scope_params(),
    )
    predicates = _run_records(
        _driver,
        _rdf_database,
        """
        MATCH (n)-[r:RDF_PREDICATE]->()
        WHERE ($active_experiment = '' OR coalesce(r._experiment, '') = $active_experiment)
          AND ($active_example_id = '' OR coalesce(r._example_id, '') = $active_example_id)
        RETURN DISTINCT coalesce(r.predicate, type(r)) AS predicate
        ORDER BY predicate
        LIMIT 12
        """,
        _scope_params(),
    )
    return {
        "labels": [row["label"] for row in labels if row.get("label")],
        "predicates": [row["predicate"] for row in predicates if row.get("predicate")],
        "forbidden_assumptions": ["RELATED_TO", "HAS_FINANCIAL", "PAYS_DIVIDEND", "FinancialMetric", "value"],
        "scope": get_active_scope(),
    }


def get_lpg_schema_card() -> dict[str, Any]:
    """Summarize the observed LPG graph shape for the current scope."""
    if not _driver:
        return {"labels": [], "relationship_types": [], "scope": get_active_scope()}
    labels = _run_records(
        _driver,
        _lpg_database,
        f"""
        MATCH (n)
        WHERE {_scope_clause('n')}
        UNWIND labels(n) AS label
        RETURN DISTINCT label
        ORDER BY label
        LIMIT 12
        """,
        _scope_params(),
    )
    rel_types = _run_records(
        _driver,
        _lpg_database,
        """
        MATCH (a)-[r]->(b)
        WHERE ($active_experiment = '' OR coalesce(r._experiment, '') = $active_experiment)
          AND ($active_example_id = '' OR coalesce(r._example_id, '') = $active_example_id)
        RETURN DISTINCT type(r) AS rel_type
        ORDER BY rel_type
        LIMIT 20
        """,
        _scope_params(),
    )
    return {
        "labels": [row["label"] for row in labels if row.get("label")],
        "relationship_types": [row["rel_type"] for row in rel_types if row.get("rel_type")],
        "forbidden_assumptions": ["RELATED_TO", "HAS_FINANCIAL", "PAYS_DIVIDEND", "FinancialMetric", "value"],
        "scope": get_active_scope(),
    }
    if mentions_scope:
        return None
    return json.dumps(
        {
            "error": (
                "Scoped debate query rejected. The current debate run is bound to an active example/experiment, "
                "so free-form Cypher must include scope anchors such as _example_id, _experiment, uri, or _node_id."
            ),
            "active_scope": get_active_scope(),
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# RDF graph tools — query the RDF database
# ---------------------------------------------------------------------------

@function_tool
def query_rdf_cypher(cypher: str) -> str:
    """Execute a Cypher query against the RDF graph database.

    The RDF graph stores entities as :RDFNode nodes connected by :RDF_PREDICATE relationships.
    Each RDFNode has a 'uri' property. Each RDF_PREDICATE edge has a 'predicate' property.

    Example queries:
      MATCH (s:RDFNode)-[r:RDF_PREDICATE]->(o:RDFNode) RETURN s.uri, r.predicate, o.uri LIMIT 10
      MATCH (n:RDFNode) WHERE toLower(n.uri) CONTAINS 'apple' RETURN n.uri LIMIT 5
    """
    if not _driver:
        return json.dumps({"error": "RDF graph not connected. Call init_connections() first."})
    scope_error = _scoped_freeform_guard(cypher)
    if scope_error:
        return scope_error
    return _run_cypher(_driver, _rdf_database, cypher)


@function_tool
def search_rdf_entities(search_text: str) -> str:
    """Search for RDF entities whose URI contains the given text (case-insensitive).

    Returns matching entities and their outgoing triples (subject → predicate → object).
    Useful for finding entities related to a concept or company name.
    """
    if not _driver:
        return json.dumps({"error": "RDF graph not connected."})
    cypher = """
    MATCH (n:RDFNode)
    WHERE toLower(n.uri) CONTAINS toLower($text)
      AND ($active_experiment = '' OR coalesce(n._experiment, '') = $active_experiment)
      AND (
            $active_example_id = ''
            OR coalesce(n._example_id, '') = $active_example_id
            OR n.uri CONTAINS $active_example_id
      )
    OPTIONAL MATCH (n)-[r:RDF_PREDICATE]->(m:RDFNode)
    RETURN n.uri AS subject, r.predicate AS predicate, m.uri AS object
    LIMIT 25
    """
    return _run_cypher(_driver, _rdf_database, cypher, _scope_params({"text": search_text}))


@function_tool
def get_rdf_entity_context(entity_uri: str) -> str:
    """Get full context for an RDF entity: all triples where it appears as subject or object.

    Use this to understand an entity's relationships in the ontology-grounded graph.
    """
    if not _driver:
        return json.dumps({"error": "RDF graph not connected."})
    cypher = """
    MATCH (n:RDFNode {uri: $uri})-[r:RDF_PREDICATE]->(m:RDFNode)
    WHERE ($active_experiment = '' OR coalesce(n._experiment, '') = $active_experiment)
      AND (
            $active_example_id = ''
            OR coalesce(n._example_id, '') = $active_example_id
            OR n.uri CONTAINS $active_example_id
      )
    RETURN 'outgoing' AS direction, n.uri AS subject, r.predicate AS predicate, m.uri AS object
    UNION ALL
    MATCH (m:RDFNode)-[r:RDF_PREDICATE]->(n:RDFNode {uri: $uri})
    WHERE ($active_experiment = '' OR coalesce(n._experiment, '') = $active_experiment)
      AND (
            $active_example_id = ''
            OR coalesce(n._example_id, '') = $active_example_id
            OR n.uri CONTAINS $active_example_id
      )
    RETURN 'incoming' AS direction, m.uri AS subject, r.predicate AS predicate, n.uri AS object
    """
    return _run_cypher(_driver, _rdf_database, cypher, _scope_params({"uri": entity_uri}))


# ---------------------------------------------------------------------------
# LPG graph tools — query the LPG database
# ---------------------------------------------------------------------------

@function_tool
def query_lpg_cypher(cypher: str) -> str:
    """Execute a Cypher query against the LPG (Labeled Property Graph) database.

    The LPG stores entities as labeled nodes with properties, connected by typed edges.
    Node labels include entity types like LegalEntity, Revenue, Dividend, etc.
    Edge types include relationship types like HAS_REVENUE, RELATED_TO, etc.
    All nodes have a '_node_id' property for identification.

    Example queries:
      MATCH (n) RETURN labels(n), n._node_id, properties(n) LIMIT 10
      MATCH (a)-[r]->(b) RETURN a._node_id, type(r), b._node_id LIMIT 10
    """
    if not _driver:
        return json.dumps({"error": "LPG graph not connected. Call init_connections() first."})
    scope_error = _scoped_freeform_guard(cypher)
    if scope_error:
        return scope_error
    return _run_cypher(_driver, _lpg_database, cypher)


@function_tool
def search_lpg_nodes(search_text: str) -> str:
    """Search for LPG nodes whose properties contain the given text (case-insensitive).

    Searches across node IDs and all string property values.
    Returns node labels, IDs, and properties.
    """
    if not _driver:
        return json.dumps({"error": "LPG graph not connected."})
    cypher = """
    MATCH (n)
    WHERE toLower(n._node_id) CONTAINS toLower($text)
       OR any(key IN keys(n) WHERE toLower(toString(n[key])) CONTAINS toLower($text))
      AND ($active_experiment = '' OR coalesce(n._experiment, '') = $active_experiment)
      AND ($active_example_id = '' OR coalesce(n._example_id, '') = $active_example_id)
    RETURN labels(n) AS labels, n._node_id AS node_id, properties(n) AS props
    LIMIT 25
    """
    return _run_cypher(_driver, _lpg_database, cypher, _scope_params({"text": search_text}))


@function_tool
def get_lpg_node_neighbors(node_id: str) -> str:
    """Get all neighbors and relationships for an LPG node.

    Returns outgoing and incoming edges with their types and properties.
    Use this to explore the local neighborhood of an entity in the property graph.
    """
    if not _driver:
        return json.dumps({"error": "LPG graph not connected."})
    cypher = """
    MATCH (n {_node_id: $nid})-[r]->(m)
    WHERE ($active_experiment = '' OR coalesce(n._experiment, '') = $active_experiment)
      AND ($active_example_id = '' OR coalesce(n._example_id, '') = $active_example_id)
    RETURN 'outgoing' AS direction, type(r) AS rel_type, properties(r) AS rel_props,
           labels(m) AS target_labels, m._node_id AS target_id, properties(m) AS target_props
    UNION ALL
    MATCH (m)-[r]->(n {_node_id: $nid})
    WHERE ($active_experiment = '' OR coalesce(n._experiment, '') = $active_experiment)
      AND ($active_example_id = '' OR coalesce(n._example_id, '') = $active_example_id)
    RETURN 'incoming' AS direction, type(r) AS rel_type, properties(r) AS rel_props,
           labels(m) AS source_labels, m._node_id AS source_id, properties(m) AS source_props
    """
    return _run_cypher(_driver, _lpg_database, cypher, _scope_params({"nid": node_id}))
