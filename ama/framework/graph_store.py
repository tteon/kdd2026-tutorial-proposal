"""
Graph store: materialize experiment artifacts into DozerDB (Neo4j-compatible).

Runtime truth for this repository:

- one DozerDB instance
- one shared Neo4j-compatible Bolt endpoint
- two logical databases inside the instance
  - RDF database: finderrdf
  - LPG database: finderlpg

RDF is still exported canonically as local `.ttl` artifacts, but the queryable
graph-store truth for both representation paths is DozerDB.
"""

from __future__ import annotations

import os
from typing import Any

from neo4j import GraphDatabase

from framework.neo4j_safe import sanitize_label, sanitize_node_id, sanitize_relationship_type


class GraphStore:
    """Single-instance DozerDB store with per-database RDF/LPG routing."""

    def __init__(
        self,
        *,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        rdf_database: str | None = None,
        lpg_database: str | None = None,
    ) -> None:
        self._uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "ama-experiment-2026")
        self._rdf_database = rdf_database or os.getenv("NEO4J_DATABASE_RDF", "finderrdf")
        self._lpg_database = lpg_database or os.getenv("NEO4J_DATABASE_LPG", "finderlpg")
        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))

    def close(self) -> None:
        self._driver.close()

    # ------------------------------------------------------------------
    # Connection / setup
    # ------------------------------------------------------------------

    def verify_connection(self) -> bool:
        """Check that the server is reachable over Bolt."""
        try:
            with self._driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception:
            return False

    def ensure_databases(self) -> None:
        """Create logical RDF/LPG databases when supported by the server."""
        with self._driver.session(database="system") as session:
            session.run(f"CREATE DATABASE {self._rdf_database} IF NOT EXISTS")
            session.run(f"CREATE DATABASE {self._lpg_database} IF NOT EXISTS")

    def ensure_indexes(self) -> None:
        """Create indexes in both RDF and LPG databases for common lookups."""
        with self._driver.session(database=self._lpg_database) as session:
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n._node_id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Document) ON (n._node_id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:OntologyConcept) ON (n._node_id)")
        with self._driver.session(database=self._rdf_database) as session:
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:RDFNode) ON (n.uri)")

    # ------------------------------------------------------------------
    # LPG materialization
    # ------------------------------------------------------------------

    def materialize_lpg(self, artifact: dict[str, Any]) -> int:
        """Write LPG nodes and edges from artifact.graph_preview into the LPG database."""
        graph = artifact.get("graph_preview", {})
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        experiment = artifact.get("experiment_name", "unknown")
        example_id = artifact.get("example_id", "unknown")

        count = 0
        with self._driver.session(database=self._lpg_database) as session:
            for node in nodes:
                raw_labels = node.get("labels", ["Entity"])
                labels = [sanitize_label(str(label), default="Entity") for label in raw_labels]
                label_str = ":".join(labels) if labels else "Entity"
                props = dict(node.get("properties", {}))
                raw_node_id = str(node.get("id", ""))
                props["_node_id"] = sanitize_node_id(raw_node_id)
                props["_source_node_id"] = raw_node_id
                props["_experiment"] = experiment
                props["_example_id"] = example_id
                props["_representation"] = "lpg"

                session.run(
                    f"CREATE (n:{label_str} $props)",
                    props=props,
                )
                count += 1

            for edge in edges:
                rel_type = sanitize_relationship_type(str(edge.get("type", "RELATED_TO")))
                props = dict(edge.get("properties", {}))
                props["_experiment"] = experiment
                props["_example_id"] = example_id

                session.run(
                    f"""
                    MATCH (a {{_node_id: $source, _experiment: $exp, _example_id: $eid}})
                    MATCH (b {{_node_id: $target, _experiment: $exp, _example_id: $eid}})
                    CREATE (a)-[r:{rel_type} $props]->(b)
                    """,
                    source=sanitize_node_id(str(edge.get("source", ""))),
                    target=sanitize_node_id(str(edge.get("target", ""))),
                    exp=experiment,
                    eid=example_id,
                    props=props,
                )
                count += 1

        return count

    # ------------------------------------------------------------------
    # RDF materialization
    # ------------------------------------------------------------------

    def materialize_rdf_as_lpg(self, artifact: dict[str, Any]) -> int:
        """Write RDF triples into the RDF database as RDF-shaped nodes/edges."""
        graph = artifact.get("graph_preview", {})
        triples = graph.get("triples", [])
        experiment = artifact.get("experiment_name", "unknown")
        example_id = artifact.get("example_id", "unknown")

        count = 0
        with self._driver.session(database=self._rdf_database) as session:
            for triple in triples:
                subj = str(triple.get("subject", ""))
                pred = str(triple.get("predicate", ""))
                obj = str(triple.get("object", ""))

                session.run(
                    """
                    MERGE (s:RDFNode {uri: $subj, _experiment: $exp, _example_id: $eid})
                    MERGE (o:RDFNode {uri: $obj, _experiment: $exp, _example_id: $eid})
                    CREATE (s)-[:RDF_PREDICATE {
                        predicate: $pred,
                        _experiment: $exp,
                        _example_id: $eid,
                        _representation: 'rdf'
                    }]->(o)
                    """,
                    subj=subj,
                    pred=pred,
                    obj=obj,
                    exp=experiment,
                    eid=example_id,
                )
                count += 1

        return count

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear_experiment(self, experiment_name: str, *, database: str | None = None) -> int:
        """Delete all nodes/edges tagged with the given experiment in one database."""
        database_name = database or self._lpg_database
        with self._driver.session(database=database_name) as session:
            result = session.run(
                """
                MATCH (n {_experiment: $exp})
                DETACH DELETE n
                RETURN count(n) as deleted
                """,
                exp=experiment_name,
            )
            record = result.single()
            return record["deleted"] if record else 0
