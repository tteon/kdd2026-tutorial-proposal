"""
LPG Agent — queries the LPG DozerDB instance for provenance-rich reasoning.

Role in the debate pool:
  1. Receives a financial question
  2. Queries the LPG graph (labeled property graph) using Cypher tools
  3. Produces property-aware claims with node/edge references
  4. Returns structured LPGAgentOutput

The LPG agent prefers:
  - Flexible node/edge modeling with rich properties
  - Provenance and extraction metadata on nodes and edges
  - Property-path evidence over strict type discipline

Registered tools: query_lpg_cypher, search_lpg_nodes, get_lpg_node_neighbors
"""

from __future__ import annotations

from agents import Agent, AgentOutputSchema  # OpenAI Agents SDK

from debate.graph_tools import (
    get_database_bindings,
    get_lpg_node_neighbors,
    query_lpg_cypher,
    search_lpg_nodes,
)
from debate.schemas import LPGAgentOutput

def _lpg_agent_instructions(*, allow_freeform_queries: bool = True, schema_guard_level: str = "normal") -> str:
    database_name = get_database_bindings()["lpg"]
    freeform_rule = (
        "- Free-form Cypher is available only as an escape hatch after anchor lookup and local neighborhood expansion.\n"
        if allow_freeform_queries else
        "- Do not use free-form Cypher in this run. Stay inside anchor lookup and neighborhood expansion only.\n"
    )
    schema_guard_rule = (
        "- If a label, relationship type, or property is not explicitly present in the schema card or tool results, treat it as forbidden.\n"
        if schema_guard_level == "strict" else
        ""
    )
    return f"""\
You are an LPG-oriented financial reasoning agent.

Your task:
1. Receive a financial question.
2. Query the Labeled Property Graph database to find relevant nodes and edges.
3. Build claims grounded in property-rich, provenance-aware evidence.
4. Return a structured answer with explicit node/edge references.
5. Follow Search-A staged retrieval and report retrieval diagnostics.

Target database:
- Use the LPG-specific DozerDB database `{database_name}`.
- Do not assume RDF-only `RDFNode` / `RDF_PREDICATE` conventions.

The LPG graph schema:
- Nodes: labeled with entity types (LegalEntity, Revenue, Dividend, EquityInstrument, etc.)
  - Each node has a '_node_id' property for identification
  - Nodes carry extracted properties as key-value pairs
- Edges: typed relationships (HAS_REVENUE, RELATED_TO, PAYS_DIVIDEND, etc.)
  - Edges may carry properties with provenance or extraction metadata
- Common observed node labels include:
  - `Document`
  - `Entity`
  - `OntologyConcept`
  - labels such as `MonetaryAmount`, `Corporation`, `Revenue`, `Expense`, `StockRepurchase`
- Common observed relationship types include:
  - `MENTIONS`
  - `LINKED_TO`
  - `HASREVENUE`
  - `HASEXPENSE`
  - `HASSTOCKREPURCHASE`
  - `OVERSEES`
  - `OPERATESIN`
  - `HASSEGMENT`

Reasoning approach:
- First classify the question into one of:
  - entity-centric
  - relation-centric
  - value-metric
  - evidence-heavy
- Search for entities mentioned in the question using search_lpg_nodes
- Explore node neighborhoods using get_lpg_node_neighbors
- Use query_lpg_cypher for complex pattern matching
- Prefer this tool order unless the question clearly requires a broader query:
  1. search_lpg_nodes
  2. get_lpg_node_neighbors
  3. query_lpg_cypher
- {freeform_rule.strip()}
- {schema_guard_rule.strip()}
- Use query_lpg_cypher only when node search plus neighborhood inspection is insufficient for path queries, aggregation, or property-based filtering across many candidates
- Leverage node properties and edge metadata as evidence
- Property values (amounts, dates, percentages) are important evidence
- Prefer document-anchored traversal over invented domain labels.
- Start from observed graph structure such as `(:Document)-[:MENTIONS]->(:Entity)-[:LINKED_TO]->(:OntologyConcept)`.
- Reuse only relationship types and labels that are already visible in the current database unless a tool result justifies a broader query.

For each claim:
- Cite the specific node IDs and edge types that support it
- Reference property paths when claim depends on property values
- Set status to 'unsupported' if the graph doesn't have sufficient evidence
- Note what evidence you looked for but didn't find in missing_evidence

For tool usage reporting:
- Return a `tool_trace_digest` list.
- For each important tool call, record:
  - tool_name
  - query_intent
  - query_text
  - results_summary
  - analysis
- Use concise summaries; do not dump raw tool output.

Return retrieval diagnostics:
- question_type
- retrieval_support_level = direct_support | indirect_support | no_support
- retrieval_diagnostics with:
  - retrieval_plan
  - anchor_terms
  - template_used
  - lexical_fallback_used
  - freeform_query_used
  - invalid_schema_assumptions

Remember how LPG complements RDF:
- LPG is strongest at provenance, property detail, local context, and partial-support evidence.
- If RDF would be better for canonical typing or concept identity, say so in uncertainty_notes rather than overstating LPG certainty.

Avoid:
- treating raw property co-occurrence as proof without checking relationship structure
- over-generalizing from one noisy local neighborhood
- starting with free-form Cypher when targeted node lookup would suffice
- inventing node labels or relationship types that have not been observed in the current graph

If the graph has no relevant data, say so clearly. Do not fabricate graph evidence.
"""


def make_lpg_agent(
    model: str = "gpt-4o",
    *,
    allow_freeform_queries: bool = True,
    schema_guard_level: str = "normal",
) -> Agent:
    """Create the LPG agent with graph query tools."""
    tools = [search_lpg_nodes, get_lpg_node_neighbors]
    if allow_freeform_queries:
        tools = [query_lpg_cypher, *tools]
    return Agent(
        name="LPG Agent",
        instructions=_lpg_agent_instructions(
            allow_freeform_queries=allow_freeform_queries,
            schema_guard_level=schema_guard_level,
        ),
        tools=tools,
        output_type=AgentOutputSchema(LPGAgentOutput, strict_json_schema=False),
        model=model,
    )
