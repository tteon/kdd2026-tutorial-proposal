"""
Agent debate pool for RDF vs LPG graph-based generation.

Uses OpenAI Agents SDK to create agents that query separate DozerDB instances
(one for RDF triples, one for LPG nodes/edges) and debate their answers.

Main entry point: DebatePool from debate.debate_pool
"""
