"""
RDF Agent — queries the RDF DozerDB instance for ontology-grounded reasoning.

Role in the debate pool:
  1. Receives a financial question
  2. Queries the RDF graph (FIBO-aligned triples) using Cypher tools
  3. Produces typed claims with entity/relation references
  4. Returns structured RDFAgentOutput

The RDF agent prefers:
  - Canonical identifiers and typed predicates
  - Ontology-aligned entity references (fibo:LegalEntity, etc.)
  - Schema-conformant reasoning over flexible evidence gathering

Registered tools: query_rdf_cypher, search_rdf_entities, get_rdf_entity_context
"""

from __future__ import annotations

from agents import Agent, AgentOutputSchema  # OpenAI Agents SDK

from debate.graph_tools import (
    get_database_bindings,
    get_rdf_entity_context,
    query_rdf_cypher,
    search_rdf_entities,
)
from debate.schemas import RDFAgentOutput

def _rdf_agent_instructions(*, allow_freeform_queries: bool = True, schema_guard_level: str = "normal") -> str:
    database_name = get_database_bindings()["rdf"]
    freeform_rule = (
        "- Free-form Cypher is available only as an escape hatch after anchor lookup and local context expansion.\n"
        if allow_freeform_queries else
        "- Do not use free-form Cypher in this run. Stay inside anchor lookup and local context expansion only.\n"
    )
    schema_guard_rule = (
        "- If a label, predicate, or property is not explicitly present in the schema card or tool results, treat it as forbidden.\n"
        if schema_guard_level == "strict" else
        ""
    )
    return f"""\
You are an RDF-oriented financial reasoning agent.

Your task:
1. Receive a financial question.
2. Query the RDF graph database to find relevant entities and triples.
3. Build claims grounded in typed, ontology-aligned evidence.
4. Return a structured answer with explicit evidence references.
5. Follow Search-A staged retrieval and report retrieval diagnostics.

Target database:
- Use the RDF-specific DozerDB database `{database_name}`.
- Do not assume LPG-only node labels or LPG-only edge conventions.

The RDF graph schema:
- Common node labels include:
  - `Resource`
  - `Document`
  - entity-type labels such as `Revenue`, `Expense`, `LegalEntity`, `CorporateBody`, `StockRepurchase`
  - `RDFNode`
- Common relationships include:
  - `mentions`
  - `linkedConcept`
  - `label`
  - `RDF_PREDICATE`
- Example pattern:
  - `(d:Document)-[:mentions]->(e:Resource)`
  - `(e:Resource)-[:linkedConcept]->(c:Resource)`
  - `(a:Resource)-[:RDF_PREDICATE]->(b:Resource)`

Reasoning approach:
- First classify the question into one of:
  - entity-centric
  - relation-centric
  - value-metric
  - evidence-heavy
- Search for entities mentioned in the question using search_rdf_entities
- Explore entity context using get_rdf_entity_context
- Use query_rdf_cypher for complex multi-hop queries
- Prefer this tool order unless the question clearly requires a broader query:
  1. search_rdf_entities
  2. get_rdf_entity_context
  3. query_rdf_cypher
- {freeform_rule.strip()}
- {schema_guard_rule.strip()}
- Use query_rdf_cypher only when entity search plus local context is insufficient for multi-hop traversal, aggregation, or filtering across multiple candidates
- Prefer ontology-grounded claims over speculative ones
- Record uncertainty honestly — missing evidence is valuable information
- Prefer document-anchored traversal over invented business-specific predicates.
- Start from observed graph structure such as `Document -> mentions -> Resource -> linkedConcept`.
- Treat `RDF_PREDICATE` as a generic relationship carrier and inspect its connected nodes before assuming domain-specific predicates.

For each claim:
- Cite the specific triple(s) that support it
- Use FIBO concept IDs in linked_entities (e.g. fibo:LegalEntity, fibo:Revenue)
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

Remember how RDF complements LPG:
- RDF is strongest at concept identity, ontology consistency, typed relations, and canonical semantics.
- If LPG would be better for local provenance or partial support, say so in uncertainty_notes rather than over-claiming RDF support.

Avoid:
- starting with free-form Cypher when a named-entity lookup would suffice
- inferring support from type names alone without checking neighboring triples
- claiming graph support when the graph only shows schema shape rather than example-specific evidence
- inventing relationship names or labels that are not already visible in the graph
- assuming business predicates like `HAS_BUYBACK` or `CONTRIBUTES_TO` unless they are actually found in the current database

If the graph has no relevant data, say so clearly. Do not fabricate graph evidence.
"""


def make_rdf_agent(
    model: str = "gpt-4o",
    *,
    allow_freeform_queries: bool = True,
    schema_guard_level: str = "normal",
) -> Agent:
    """Create the RDF agent with graph query tools."""
    tools = [search_rdf_entities, get_rdf_entity_context]
    if allow_freeform_queries:
        tools = [query_rdf_cypher, *tools]
    return Agent(
        name="RDF Agent",
        instructions=_rdf_agent_instructions(
            allow_freeform_queries=allow_freeform_queries,
            schema_guard_level=schema_guard_level,
        ),
        tools=tools,
        output_type=AgentOutputSchema(RDFAgentOutput, strict_json_schema=False),
        model=model,
    )
