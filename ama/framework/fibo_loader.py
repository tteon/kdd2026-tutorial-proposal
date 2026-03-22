"""Dynamic FIBO OWL loader.

Fetches FIBO RDF files from GitHub, parses with rdflib, and builds
FIBOConceptSpec objects from real SKOS/RDFS annotations. Concepts not
found in FIBO are maintained as AMA-extension specs.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS


@dataclass
class FIBOConceptSpec:
    concept_id: str
    label: str
    definition: str
    synonyms: List[str] = field(default_factory=list)
    semantic_features: List[str] = field(default_factory=list)
    soft_signals: List[str] = field(default_factory=list)


logger = logging.getLogger(__name__)

# ── FIBO namespaces ────────────────────────────────────────────────────
FIBO_BASE_IRI = "https://spec.edmcouncil.org/fibo/ontology/"
FIBO_RAW_BASE = "https://raw.githubusercontent.com/edmcouncil/fibo/master/"

CMNS_AV = Namespace("https://www.omg.org/spec/Commons/AnnotationVocabulary/")

# Default cache location
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "fibo_cache"

# ── FIBO source registry ──────────────────────────────────────────────
# module_path → relative RDF file under FIBO_RAW_BASE
FIBO_SOURCES: dict[str, str] = {
    "BE/LegalEntities/LegalPersons": "BE/LegalEntities/LegalPersons.rdf",
    "BE/LegalEntities/CorporateBodies": "BE/LegalEntities/CorporateBodies.rdf",
    "BE/GovernmentEntities/GovernmentEntities": "BE/GovernmentEntities/GovernmentEntities.rdf",
    "FBC/FunctionalEntities/RegulatoryAgencies": "FBC/FunctionalEntities/RegulatoryAgencies.rdf",
    "FBC/FinancialInstruments/FinancialInstruments": "FBC/FinancialInstruments/FinancialInstruments.rdf",
    "FND/Accounting/CurrencyAmount": "FND/Accounting/CurrencyAmount.rdf",
    "FND/Agreements/Agreements": "FND/Agreements/Agreements.rdf",
    "FND/Law/LegalCore": "FND/Law/LegalCore.rdf",
    "SEC/Equities/EquityInstruments": "SEC/Equities/EquityInstruments.rdf",
}

# Which classes to extract from each module.
# Tuples: (spec_key, fibo_class_name) — spec_key is our project's concept name,
# fibo_class_name is the actual OWL class local name in the RDF file.
# When both are the same, a plain string is also accepted.
FIBO_TARGET_CLASSES: dict[str, list[str | tuple[str, str]]] = {
    "BE/LegalEntities/LegalPersons": ["LegalEntity"],
    "BE/LegalEntities/CorporateBodies": [("CorporateBody", "Corporation")],
    "BE/GovernmentEntities/GovernmentEntities": [("GovernmentEntity", "GovernmentBody")],
    "FBC/FunctionalEntities/RegulatoryAgencies": [
        "RegulatoryAgency",
        ("Legislation", "Regulation"),
    ],
    "FBC/FinancialInstruments/FinancialInstruments": [
        "FinancialInstrument",
        "EquityInstrument",
    ],
    "FND/Accounting/CurrencyAmount": ["MonetaryAmount"],
    "FND/Agreements/Agreements": [("ContractualObligation", "MutualCommitment")],
    "FND/Law/LegalCore": [("Court", "CourtOfLaw")],
    "SEC/Equities/EquityInstruments": ["Dividend"],
}

# Pre-computed prefix map (module_path → prefix string)
FIBO_MODULE_PREFIXES: dict[str, str] = {
    "BE/LegalEntities/LegalPersons": "fibo-be-le-lp",
    "BE/LegalEntities/CorporateBodies": "fibo-be-le-cb",
    "BE/GovernmentEntities/GovernmentEntities": "fibo-be-ge-ge",
    "FBC/FunctionalEntities/RegulatoryAgencies": "fibo-fbc-fct-rga",
    "FBC/FinancialInstruments/FinancialInstruments": "fibo-fbc-fi-fi",
    "FND/Accounting/CurrencyAmount": "fibo-fnd-acc-cur",
    "FND/Agreements/Agreements": "fibo-fnd-agr-agr",
    "FND/Law/LegalCore": "fibo-fnd-law-lcor",
    "SEC/Equities/EquityInstruments": "fibo-sec-eq-eq",
}

# Fallback annotations for OMG-Commons-migrated classes whose definitions
# are not present in the individual FIBO RDF files.
_DEFINITION_FALLBACKS: dict[str, str] = {
    "LegalEntity": "A legal person such as a corporation, partnership, or other organization with the capacity to enter contracts and assume obligations.",
    "RegulatoryAgency": "An authority responsible for supervision, regulation, or enforcement within a jurisdiction or sector.",
    "Legislation": "A law, statute, regulation, or rule enacted by a legislative or regulatory authority.",
    "ContractualObligation": "A legally binding commitment arising from a contract or agreement.",
}

_SYNONYM_FALLBACKS: dict[str, list[str]] = {
    "LegalEntity": ["legal entity", "registered entity", "corporation", "company"],
    "RegulatoryAgency": ["regulator", "regulatory agency", "supervisory authority"],
    "Legislation": ["legislation", "statute", "act", "regulation", "rule", "law"],
    "ContractualObligation": ["contractual obligation", "agreement", "contract", "commitment", "warranty"],
}


# ── AMA extension specs (not in FIBO) ─────────────────────────────────
AMA_EXTENSION_SPECS: dict[str, FIBOConceptSpec] = {
    "Revenue": FIBOConceptSpec(
        concept_id="ama:Revenue",
        label="Revenue",
        definition="An amount representing income earned from operations, contracts, or other business activity.",
        synonyms=["revenue", "sales", "top line"],
        semantic_features=["income", "sales", "earned", "period"],
        soft_signals=[
            "the text refers to inflows or earned business income",
            "the mention is about reported earnings before expense deduction",
        ],
    ),
    "Expense": FIBOConceptSpec(
        concept_id="ama:Expense",
        label="Expense",
        definition="An amount representing costs or outflows incurred in the course of operations or obligations.",
        synonyms=["expense", "cost", "operating expense"],
        semantic_features=["cost", "outflow", "incurred", "operating"],
        soft_signals=[
            "the text describes a cost or outflow",
            "the mention is about incurred business spending rather than income",
        ],
    ),
    "BoardOfDirectors": FIBOConceptSpec(
        concept_id="ama:BoardOfDirectors",
        label="BoardOfDirectors",
        definition="A governing body responsible for overseeing and directing a corporation or similar organization.",
        synonyms=["board of directors", "board", "directors"],
        semantic_features=["oversight", "governance", "board", "director"],
        soft_signals=[
            "the text describes oversight or governance responsibility",
            "the mention refers to the body of directors rather than a single person",
        ],
    ),
    "Shareholder": FIBOConceptSpec(
        concept_id="ama:Shareholder",
        label="Shareholder",
        definition="A person or organization that owns shares or equity interests in an entity.",
        synonyms=["shareholder", "stockholder", "investor"],
        semantic_features=["owns shares", "holder", "equity owner"],
        soft_signals=[
            "the text explicitly describes ownership of shares",
            "the mention is the owner, not the instrument itself",
        ],
    ),
    "StockRepurchase": FIBOConceptSpec(
        concept_id="ama:StockRepurchase",
        label="StockRepurchase",
        definition="A transaction in which an issuer buys back its own shares from the market or shareholders.",
        synonyms=["buyback", "share repurchase", "stock repurchase"],
        semantic_features=["issuer", "buys back", "shares", "repurchase"],
        soft_signals=[
            "the text describes an issuer purchasing its own shares",
            "the mention signals a shareholder return mechanism via buyback",
        ],
    ),
    "AuditCommittee": FIBOConceptSpec(
        concept_id="ama:AuditCommittee",
        label="AuditCommittee",
        definition="A committee of the board responsible for overseeing financial reporting, internal controls, and audit processes.",
        synonyms=["audit committee", "audit oversight", "financial oversight committee"],
        semantic_features=["audit", "oversight", "financial reporting", "internal controls"],
        soft_signals=[
            "the text describes the audit oversight function of the board",
            "the mention is about a governance committee rather than the full board",
        ],
    ),
    "ComplianceStandard": FIBOConceptSpec(
        concept_id="ama:ComplianceStandard",
        label="ComplianceStandard",
        definition="A standard, framework, or certification requirement for regulatory or industry compliance.",
        synonyms=["compliance standard", "regulatory standard", "certification", "security standard", "PCI DSS"],
        semantic_features=["compliance", "standard", "certification", "regulatory"],
        soft_signals=[
            "the text references a named compliance or security standard",
            "the mention is about a regulatory requirement rather than an organization",
        ],
    ),
    "CapitalExpenditure": FIBOConceptSpec(
        concept_id="ama:CapitalExpenditure",
        label="CapitalExpenditure",
        definition="An expenditure to acquire, upgrade, or maintain long-term physical or intangible assets.",
        synonyms=["capital expenditure", "capex", "capital spending", "investment in assets"],
        semantic_features=["capital", "expenditure", "asset", "investment"],
        soft_signals=[
            "the text describes spending on long-term assets or infrastructure",
            "the mention is about capital investment rather than operating expenses",
        ],
    ),
    "BusinessSegment": FIBOConceptSpec(
        concept_id="ama:BusinessSegment",
        label="BusinessSegment",
        definition="A distinguishable component of a business that engages in specific activities or serves specific markets.",
        synonyms=["business segment", "division", "vertical", "sector", "product line", "market segment"],
        semantic_features=["segment", "division", "market", "vertical"],
        soft_signals=[
            "the text describes a distinct business area, product line, or market vertical",
            "the mention is about a business unit rather than a single product",
        ],
    ),
    # --- Concepts verified NOT in FIBO (moved from hardcoded specs) ---
    "LegalProceeding": FIBOConceptSpec(
        concept_id="ama:LegalProceeding",
        label="LegalProceeding",
        definition="A formal proceeding before a court, tribunal, or regulatory body.",
        synonyms=["legal proceeding", "lawsuit", "litigation", "case", "legal action", "MDL"],
        semantic_features=["court", "proceeding", "lawsuit", "litigation"],
        soft_signals=[
            "the text describes a lawsuit, case number, or legal action",
            "the mention references a court proceeding or legal dispute",
        ],
    ),
    "CorporateOfficer": FIBOConceptSpec(
        concept_id="ama:CorporateOfficer",
        label="CorporateOfficer",
        definition="A senior executive or officer holding a named position within a corporation.",
        synonyms=["corporate officer", "CEO", "CFO", "CLO", "CISO", "president", "officer"],
        semantic_features=["executive", "officer", "position", "leadership"],
        soft_signals=[
            "the text names a specific corporate officer role or title",
            "the mention is about a person's executive role rather than the person themselves",
        ],
    ),
    "AccountingPolicy": FIBOConceptSpec(
        concept_id="ama:AccountingPolicy",
        label="AccountingPolicy",
        definition="A principle, method, or practice adopted by an entity for recognizing, measuring, and reporting financial items.",
        synonyms=["accounting policy", "accounting standard", "revenue recognition policy", "GAAP"],
        semantic_features=["recognition", "measurement", "reporting", "standard"],
        soft_signals=[
            "the text describes how a company recognizes or measures financial items",
            "the mention refers to an accounting rule or convention",
        ],
    ),
    "GeographicRegion": FIBOConceptSpec(
        concept_id="ama:GeographicRegion",
        label="GeographicRegion",
        definition="A named geographic area relevant to business operations, sales, or regulatory jurisdiction.",
        synonyms=["region", "geography", "country", "territory", "market region", "EMEA", "Americas"],
        semantic_features=["geographic", "region", "territory", "market"],
        soft_signals=[
            "the text references a named geographic area for business operations",
            "the mention is about a regional market or territory",
        ],
    ),
}


# ── Fetch / parse / cache ─────────────────────────────────────────────

def fetch_fibo_rdf(module_path: str, *, cache_dir: Path) -> Graph:
    """Fetch one FIBO RDF file from GitHub, cache locally as .rdf."""
    rdf_file = FIBO_SOURCES[module_path]
    cache_file = cache_dir / rdf_file.replace("/", "__")

    if cache_file.exists():
        logger.debug("Using cached %s", cache_file)
        g = Graph()
        g.parse(str(cache_file), format="xml")
        return g

    url = FIBO_RAW_BASE + rdf_file
    logger.info("Fetching %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "AMA-FIBO-Loader/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(data)

    g = Graph()
    g.parse(str(cache_file), format="xml")
    return g


def extract_class_metadata(
    graph: Graph,
    fibo_class_name: str,
    module_path: str,
    *,
    spec_key: str | None = None,
) -> FIBOConceptSpec | None:
    """Extract SKOS/RDFS annotations for a named class from a parsed FIBO graph.

    Args:
        fibo_class_name: The actual OWL class local name in the FIBO ontology.
        spec_key: Our project's concept name (used as dict key and label).
                  Defaults to fibo_class_name if not provided.
    """
    spec_key = spec_key or fibo_class_name

    # Find the class IRI ending with the local name
    target_cls: URIRef | None = None
    for s in graph.subjects(RDF.type, OWL.Class):
        if isinstance(s, URIRef) and str(s).endswith("/" + fibo_class_name):
            target_cls = s
            break

    if target_cls is None:
        logger.warning("Class %s not found in %s", fibo_class_name, module_path)
        return None

    # Extract annotations (with fallbacks for OMG-Commons-migrated classes)
    label = _first_literal(graph, target_cls, RDFS.label) or fibo_class_name
    definition = (
        _first_literal(graph, target_cls, SKOS.definition)
        or _first_literal(graph, target_cls, RDFS.comment)
        or _DEFINITION_FALLBACKS.get(spec_key, "")
    )

    synonyms: list[str] = []
    for syn in graph.objects(target_cls, CMNS_AV.synonym):
        synonyms.append(str(syn))
    for abbr in graph.objects(target_cls, CMNS_AV.abbreviation):
        synonyms.append(str(abbr))
    # Fallback synonyms for sparse annotations
    if not synonyms and spec_key in _SYNONYM_FALLBACKS:
        synonyms = list(_SYNONYM_FALLBACKS[spec_key])
    # Add both the spec_key and FIBO label as synonyms (space-separated form)
    for name in [spec_key, fibo_class_name]:
        spaced = _camel_to_spaced(name).lower()
        if spaced not in [s.lower() for s in synonyms]:
            synonyms.insert(0, spaced)

    soft_signals: list[str] = []
    for note in graph.objects(target_cls, CMNS_AV.explanatoryNote):
        soft_signals.append(str(note))

    semantic_features: list[str] = []
    for parent in graph.objects(target_cls, RDFS.subClassOf):
        if isinstance(parent, URIRef):
            parent_local = str(parent).rsplit("/", 1)[-1]
            if parent_local and parent_local != fibo_class_name:
                semantic_features.append(parent_local)

    # Build concept_id with module-specific prefix, using the spec_key
    prefix = FIBO_MODULE_PREFIXES.get(module_path, "fibo")
    concept_id = f"{prefix}:{spec_key}"

    return FIBOConceptSpec(
        concept_id=concept_id,
        label=spec_key,
        definition=definition,
        synonyms=synonyms,
        semantic_features=semantic_features,
        soft_signals=soft_signals,
    )


def _first_literal(graph: Graph, subject: URIRef, predicate: URIRef) -> str | None:
    """Return the first literal value for a subject+predicate, preferring English."""
    candidates: list[str] = []
    for obj in graph.objects(subject, predicate):
        val = str(obj)
        lang = getattr(obj, "language", None)
        if lang == "en":
            return val
        candidates.append(val)
    return candidates[0] if candidates else None


def _camel_to_spaced(name: str) -> str:
    """Convert CamelCase to spaced form: 'LegalEntity' → 'legal entity'."""
    result: list[str] = []
    current: list[str] = []
    for ch in name:
        if ch.isupper() and current:
            result.append("".join(current))
            current = [ch.lower()]
        else:
            current.append(ch.lower())
    if current:
        result.append("".join(current))
    return " ".join(result)


def _spec_to_dict(spec: FIBOConceptSpec) -> dict[str, Any]:
    return {
        "concept_id": spec.concept_id,
        "label": spec.label,
        "definition": spec.definition,
        "synonyms": spec.synonyms,
        "semantic_features": spec.semantic_features,
        "soft_signals": spec.soft_signals,
    }


def _spec_from_dict(d: dict[str, Any]) -> FIBOConceptSpec:
    return FIBOConceptSpec(
        concept_id=d["concept_id"],
        label=d["label"],
        definition=d["definition"],
        synonyms=d.get("synonyms", []),
        semantic_features=d.get("semantic_features", []),
        soft_signals=d.get("soft_signals", []),
    )


# ── Main entry point ──────────────────────────────────────────────────

def load_fibo_specs(
    *,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> dict[str, FIBOConceptSpec]:
    """Load all FIBO concept specs from OWL files + AMA extensions.

    Results are JSON-cached to avoid re-parsing on every import.
    """
    cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    specs_cache = cache_dir / "fibo_specs.json"

    # Return from JSON cache if available and not forcing refresh
    if not force_refresh and specs_cache.exists():
        logger.debug("Loading specs from cache: %s", specs_cache)
        with open(specs_cache) as f:
            cached = json.load(f)
        specs = {k: _spec_from_dict(v) for k, v in cached.get("fibo", {}).items()}
        specs.update(AMA_EXTENSION_SPECS)
        return specs

    # Fetch and parse each FIBO module
    fibo_specs: dict[str, FIBOConceptSpec] = {}
    for module_path, entries in FIBO_TARGET_CLASSES.items():
        try:
            graph = fetch_fibo_rdf(module_path, cache_dir=cache_dir)
        except Exception:
            logger.exception("Failed to fetch %s", module_path)
            continue

        for entry in entries:
            if isinstance(entry, tuple):
                spec_key, fibo_class = entry
            else:
                spec_key = fibo_class = entry

            spec = extract_class_metadata(
                graph, fibo_class, module_path, spec_key=spec_key,
            )
            if spec is not None:
                fibo_specs[spec_key] = spec
            else:
                logger.warning("Skipping %s/%s (not found in %s)", spec_key, fibo_class, module_path)

    # Persist parsed specs to JSON cache
    cache_data = {"fibo": {k: _spec_to_dict(v) for k, v in fibo_specs.items()}}
    with open(specs_cache, "w") as f:
        json.dump(cache_data, f, indent=2)
    logger.info("Cached %d FIBO specs to %s", len(fibo_specs), specs_cache)

    # Merge with AMA extensions
    fibo_specs.update(AMA_EXTENSION_SPECS)
    return fibo_specs


def get_fibo_prefixes() -> dict[str, str]:
    """Return namespace prefix map derived from FIBO module registry."""
    prefixes: dict[str, str] = {}
    for module_path, prefix in FIBO_MODULE_PREFIXES.items():
        prefixes[prefix] = FIBO_BASE_IRI + module_path + "/"
    return prefixes
