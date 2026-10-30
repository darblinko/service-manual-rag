"""Neo4J graph schema definitions."""

from datetime import datetime
from enum import Enum
from typing import Any

from service_manual_rag.utils.logging_config import get_logger

logger = get_logger(__name__)


class NodeType(str, Enum):
    """Node types in the knowledge graph."""

    EQUIPMENT = "Equipment"
    MANUAL = "Manual"
    SECTION = "Section"
    COMPONENT = "Component"
    PROCEDURE = "Procedure"
    SYMPTOM = "Symptom"
    SOLUTION = "Solution"
    SEARCH_QUERY = "SearchQuery"
    SEARCH_RESULT = "SearchResult"
    FAILURE_MODE = "FailureMode"
    SAFETY_HAZARD = "SafetyHazard"
    # Phase 2: Topology (position in the process and upstream or downstream effects)
    ZONE = "Zone"
    # Phase 3: Resources required for maintenance procedures
    PART = "Part"
    TOOL = "Tool"
    SKILL = "Skill"


class RelationshipType(str, Enum):
    """Relationship types in the knowledge graph."""

    HAS_MANUAL = "HAS_MANUAL"
    CONTAINS = "CONTAINS"
    MENTIONS = "MENTIONS"
    DESCRIBES = "DESCRIBES"
    SOLVED_BY = "SOLVED_BY"
    HAS_COMPONENT = "HAS_COMPONENT"
    REQUIRES_PROCEDURE = "REQUIRES_PROCEDURE"
    PRECEDES = "PRECEDES"
    REPLACED_BY = "REPLACED_BY"
    RETURNED = "RETURNED"
    DOWNLOADED_AS = "DOWNLOADED_AS"
    # Phase 1: Core Diagnostic Chain
    EXHIBITS = "EXHIBITS"  # Equipment to Symptom
    INDICATES = "INDICATES"  # Symptom to FailureMode
    RESOLVES = "RESOLVES"  # Procedure to FailureMode
    POSES = "POSES"  # FailureMode to SafetyHazard
    # Phase 2: Topology
    FEEDS_INTO = "FEEDS_INTO"  # Equipment to Equipment
    LOCATED_IN = "LOCATED_IN"  # Equipment to Zone
    # Phase 3: Resources
    REQUIRES = "REQUIRES"  # Procedure to Part
    REQUIRES_TOOL = "REQUIRES_TOOL"  # Procedure to Tool
    REQUIRES_SKILL = "REQUIRES_SKILL"  # Procedure to Skill


def create_constraints() -> list[str]:
    """Create Cypher queries for constraints.

    Returns:
        list: List of constraint creation queries
    """
    constraints = [
        # Equipment constraints
        f"CREATE CONSTRAINT equipment_id IF NOT EXISTS FOR (e:{NodeType.EQUIPMENT.value}) REQUIRE e.id IS UNIQUE",
        f"CREATE CONSTRAINT manual_doc_id IF NOT EXISTS FOR (m:{NodeType.MANUAL.value}) REQUIRE m.doc_id IS UNIQUE",
        f"CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:{NodeType.SECTION.value}) REQUIRE s.id IS UNIQUE",
        f"CREATE CONSTRAINT component_id IF NOT EXISTS FOR (c:{NodeType.COMPONENT.value}) REQUIRE c.id IS UNIQUE",
        f"CREATE CONSTRAINT procedure_id IF NOT EXISTS FOR (p:{NodeType.PROCEDURE.value}) REQUIRE p.id IS UNIQUE",
        f"CREATE CONSTRAINT symptom_id IF NOT EXISTS FOR (s:{NodeType.SYMPTOM.value}) REQUIRE s.id IS UNIQUE",
        f"CREATE CONSTRAINT solution_id IF NOT EXISTS FOR (s:{NodeType.SOLUTION.value}) REQUIRE s.id IS UNIQUE",
        f"CREATE CONSTRAINT search_query_id IF NOT EXISTS FOR (sq:{NodeType.SEARCH_QUERY.value}) REQUIRE sq.id IS UNIQUE",
        f"CREATE CONSTRAINT search_result_id IF NOT EXISTS FOR (sr:{NodeType.SEARCH_RESULT.value}) REQUIRE sr.id IS UNIQUE",
        # Phase 1: Core Diagnostic Chain constraints
        f"CREATE CONSTRAINT failure_mode_id IF NOT EXISTS FOR (fm:{NodeType.FAILURE_MODE.value}) REQUIRE fm.id IS UNIQUE",
        f"CREATE CONSTRAINT safety_hazard_id IF NOT EXISTS FOR (sh:{NodeType.SAFETY_HAZARD.value}) REQUIRE sh.id IS UNIQUE",
        # Phase 2: Topology constraints
        f"CREATE CONSTRAINT zone_id IF NOT EXISTS FOR (z:{NodeType.ZONE.value}) REQUIRE z.id IS UNIQUE",
        # Phase 3: Resource constraints
        f"CREATE CONSTRAINT part_id IF NOT EXISTS FOR (p:{NodeType.PART.value}) REQUIRE p.id IS UNIQUE",
        f"CREATE CONSTRAINT tool_id IF NOT EXISTS FOR (t:{NodeType.TOOL.value}) REQUIRE t.id IS UNIQUE",
        f"CREATE CONSTRAINT skill_id IF NOT EXISTS FOR (sk:{NodeType.SKILL.value}) REQUIRE sk.id IS UNIQUE",
    ]

    logger.debug(f"Created {len(constraints)} constraint queries")
    return constraints


FULLTEXT_INDEX_NAME = "fulltext_index"
FULLTEXT_LABELS = (
    "Equipment|Manual|Section|Component|Procedure|Symptom|Solution|FailureMode|SafetyHazard|"
    "Zone|Part|Tool|Skill"
)
FULLTEXT_PROPERTIES = (
    "name",
    "title",
    "description",
    "part_name",
    "fix_description",
    "content",
    "warning",
    "tool_name",
    "skill_name",
    "action_type",
    "steps",
    "difficulty",
)


def fulltext_index_statement() -> str:
    """CREATE statement for the full-text index that graph retrieval queries."""
    properties = ", ".join(f"n.{name}" for name in FULLTEXT_PROPERTIES)
    return (
        f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS "
        f"FOR (n:{FULLTEXT_LABELS}) ON EACH [{properties}]"
    )


def create_indexes() -> list[str]:
    """Create Cypher queries for indexes.

    Returns:
        list: List of index creation queries
    """
    indexes = [
        # Equipment indexes
        f"CREATE INDEX equipment_oem IF NOT EXISTS FOR (e:{NodeType.EQUIPMENT.value}) ON (e.oem)",
        (
            f"CREATE INDEX equipment_model IF NOT EXISTS "
            f"FOR (e:{NodeType.EQUIPMENT.value}) ON (e.model)"
        ),
        # Manual indexes
        f"CREATE INDEX manual_title IF NOT EXISTS FOR (m:{NodeType.MANUAL.value}) ON (m.title)",
        f"CREATE INDEX manual_version IF NOT EXISTS FOR (m:{NodeType.MANUAL.value}) ON (m.version)",
        # Section indexes
        f"CREATE INDEX section_name IF NOT EXISTS FOR (s:{NodeType.SECTION.value}) ON (s.name)",
        f"CREATE INDEX section_type IF NOT EXISTS FOR (s:{NodeType.SECTION.value}) ON (s.type)",
        # Component indexes
        (
            f"CREATE INDEX component_part_number IF NOT EXISTS "
            f"FOR (c:{NodeType.COMPONENT.value}) ON (c.part_number)"
        ),
        (
            f"CREATE INDEX component_name IF NOT EXISTS "
            f"FOR (c:{NodeType.COMPONENT.value}) ON (c.part_name)"
        ),
        # Procedure indexes
        (
            f"CREATE INDEX procedure_action_type IF NOT EXISTS "
            f"FOR (p:{NodeType.PROCEDURE.value}) ON (p.action_type)"
        ),
        # SearchQuery indexes
        (
            f"CREATE INDEX search_query_timestamp IF NOT EXISTS "
            f"FOR (sq:{NodeType.SEARCH_QUERY.value}) ON (sq.timestamp)"
        ),
        # Fulltext index for text search across all node types
        fulltext_index_statement(),
        # Phase 1: Core Diagnostic Chain indexes
        (
            f"CREATE INDEX failure_mode_category IF NOT EXISTS "
            f"FOR (fm:{NodeType.FAILURE_MODE.value}) ON (fm.category)"
        ),
        (
            f"CREATE INDEX failure_mode_severity IF NOT EXISTS "
            f"FOR (fm:{NodeType.FAILURE_MODE.value}) ON (fm.severity)"
        ),
        (
            f"CREATE INDEX safety_hazard_type IF NOT EXISTS "
            f"FOR (sh:{NodeType.SAFETY_HAZARD.value}) ON (sh.hazard_type)"
        ),
        # Phase 2: Topology indexes
        f"CREATE INDEX zone_type IF NOT EXISTS FOR (z:{NodeType.ZONE.value}) ON (z.zone_type)",
        # Phase 3: Resource indexes
        f"CREATE INDEX part_number IF NOT EXISTS FOR (p:{NodeType.PART.value}) ON (p.part_number)",
        f"CREATE INDEX tool_type IF NOT EXISTS FOR (t:{NodeType.TOOL.value}) ON (t.tool_type)",
        (
            f"CREATE INDEX skill_level IF NOT EXISTS "
            f"FOR (sk:{NodeType.SKILL.value}) ON (sk.certification_level)"
        ),
    ]

    logger.debug(f"Created {len(indexes)} index queries")
    return indexes


# Node property schemas
EQUIPMENT_SCHEMA = {
    "id": "string",  # Unique identifier (e.g., "EXAMPLECO_CP-200")
    "oem": "string",  # Manufacturer
    "model": "string",  # Model number
    "category": "string",  # Equipment category (optional)
}

MANUAL_SCHEMA = {
    "doc_id": "string",  # Unique document identifier
    "title": "string",  # Manual title
    "version": "string",  # Version/revision
    "date": "string",  # Publication date
    "file_path": "string",  # Local file path
    "manual_type": "string",  # service, installation, parts, etc.
}

SECTION_SCHEMA = {
    "id": "string",  # Unique identifier
    "name": "string",  # Section name
    "type": "string",  # Section type (installation, maintenance, etc.)
    "page_range": "string",  # Page range (e.g., "10-25")
    "level": "integer",  # Hierarchy level (2 or 3)
}

COMPONENT_SCHEMA = {
    "id": "string",  # Unique identifier
    "part_name": "string",  # Component name
    "part_number": "string",  # Part number
    "description": "string",  # Description
}

PROCEDURE_SCHEMA = {
    "id": "string",  # Unique identifier
    "action_type": "string",  # replace, install, adjust, etc.
    "steps": "string",  # Procedure steps (JSON or text)
    "difficulty": "string",  # easy, medium, hard
    "estimated_time": "string",  # Estimated time
}

SYMPTOM_SCHEMA = {
    "id": "string",  # Unique identifier
    "description": "string",  # Symptom description
    "severity": "string",  # low, medium, high, critical
}

SOLUTION_SCHEMA = {
    "id": "string",  # Unique identifier
    "fix_description": "string",  # Solution description
    "parts_needed": "list",  # List of part numbers
}

SEARCH_QUERY_SCHEMA = {
    "id": "string",  # Unique identifier
    "oem": "string",  # OEM searched
    "model": "string",  # Model searched
    "timestamp": "datetime",  # Search timestamp
}

SEARCH_RESULT_SCHEMA = {
    "id": "string",  # Unique identifier
    "url": "string",  # Result URL
    "title": "string",  # Result title
    "rank": "integer",  # Search rank
    "source": "string",  # Search engine source
}

# Phase 1: Core Diagnostic Chain Node Schemas
FAILURE_MODE_SCHEMA = {
    "id": "string",  # Unique identifier
    "name": "string",  # e.g., "Bearing Wear", "Misalignment"
    "description": "string",  # Detailed description
    "category": "string",  # mechanical, electrical, hydraulic, pneumatic, etc.
    "severity": "string",  # low, medium, high, critical
}

SAFETY_HAZARD_SCHEMA = {
    "id": "string",  # Unique identifier
    "hazard_type": "string",  # electrical, pressure, chemical, thermal, etc.
    "warning": "string",  # Warning message
    "severity": "string",  # warning, danger, critical
    "required_ppe": "string",  # Required personal protective equipment (optional)
}

# Phase 2: Topology Node Schemas
ZONE_SCHEMA = {
    "id": "string",  # Unique identifier
    "name": "string",  # "Building A, Floor 2", "Zone 3"
    "zone_type": "string",  # production, storage, maintenance, outdoor
    "environment": "string",  # controlled, outdoor, harsh, clean_room, etc.
    "description": "string",  # Additional details (optional)
}

# Phase 3: Resource Node Schemas
PART_SCHEMA = {
    "id": "string",  # Unique identifier
    "part_number": "string",  # Part number
    "description": "string",  # Part description
    "stock_level": "integer",  # Current stock level (optional)
    "reorder_point": "integer",  # Reorder threshold (optional)
    "lead_time_days": "integer",  # Lead time in days (optional)
    "unit_cost": "float",  # Cost per unit (optional)
}

TOOL_SCHEMA = {
    "id": "string",  # Unique identifier
    "tool_name": "string",  # Tool name
    "tool_type": "string",  # specialized, common, calibrated, diagnostic
    "requires_certification": "boolean",  # Whether certification is needed
    "description": "string",  # Tool description (optional)
}

SKILL_SCHEMA = {
    "id": "string",  # Unique identifier
    "skill_name": "string",  # Skill/certification name
    "certification_level": "string",  # basic, intermediate, expert, master
    "certification_body": "string",  # Certifying organization (optional)
    "description": "string",  # Skill description (optional)
}

# Relationship property schemas for continuous learning
RELATIONSHIP_PROPERTY_SCHEMA = {
    "weight": "float",  # Confidence score (0.0 to 1.0)
    "evidence_count": "integer",  # Total number of confirmations
    "success_count": "integer",  # Number of successful outcomes
    "failure_count": "integer",  # Number of failed attempts
    "last_confirmed": "datetime",  # Timestamp of last confirmation
    "created_at": "datetime",  # Timestamp of creation
}

# Initial weights by relationship type
INITIAL_WEIGHTS = {
    RelationshipType.SOLVED_BY: 0.5,  # Critical for diagnostics, start neutral
    RelationshipType.REQUIRES_PROCEDURE: 0.7,  # Manual-defined, relatively reliable
    RelationshipType.HAS_COMPONENT: 0.9,  # Structural, highly reliable
    RelationshipType.PRECEDES: 0.8,  # Sequence relationships, reliable
    RelationshipType.HAS_MANUAL: 0.95,  # Direct hierarchy, very reliable
    RelationshipType.CONTAINS: 0.95,  # Document structure, very reliable
    RelationshipType.MENTIONS: 0.6,  # Text-based inference, moderate confidence
    RelationshipType.DESCRIBES: 0.7,  # Descriptive relationships
    RelationshipType.REPLACED_BY: 0.8,  # Part replacement info
    # Phase 1: Core Diagnostic Chain weights
    RelationshipType.EXHIBITS: 0.6,  # Equipment to Symptom (learns from observations)
    RelationshipType.INDICATES: 0.5,  # Symptom to FailureMode (critical, learns from diagnostics)
    RelationshipType.RESOLVES: 0.7,  # Procedure to FailureMode (manual-based, strengthens with success)
    RelationshipType.POSES: 0.95,  # FailureMode to SafetyHazard (safety-critical, rarely wrong)
    # Phase 2: Topology weights
    RelationshipType.FEEDS_INTO: 0.9,  # Equipment to Equipment (structural, reliable)
    RelationshipType.LOCATED_IN: 0.95,  # Equipment to Zone (rarely changes, reliable)
    # Phase 3: Resource weights
    RelationshipType.REQUIRES: 0.85,  # Procedure to Part (manual-defined, reliable)
    RelationshipType.REQUIRES_TOOL: 0.85,  # Procedure to Tool (manual-defined, reliable)
    RelationshipType.REQUIRES_SKILL: 0.9,  # Procedure to Skill (compliance-critical, very reliable)
}


def get_initial_relationship_properties(relationship_type: RelationshipType) -> dict[str, Any]:
    """Get initial properties for a relationship.

    Args:
        relationship_type: Type of relationship

    Returns:
        dict: Initial properties with weight, evidence counts, and timestamps
    """
    initial_weight = INITIAL_WEIGHTS.get(relationship_type, 0.5)

    return {
        "weight": initial_weight,
        "evidence_count": 1,  # Start with 1 (from manual extraction)
        "success_count": 0,  # No confirmations yet
        "failure_count": 0,  # No failures yet
        "last_confirmed": datetime.now().isoformat(),
        "created_at": datetime.now().isoformat(),
    }


def get_node_creation_query(node_type: NodeType, properties: dict[str, Any]) -> str:
    """Generate Cypher query for creating a node.

    Args:
        node_type: Type of node to create
        properties: Node properties

    Returns:
        str: Cypher CREATE query
    """
    # Escape and format properties
    props_str = ", ".join([f"{k}: ${k}" for k in properties])
    query = f"CREATE (n:{node_type.value} {{{props_str}}}) RETURN n"
    return query


def get_node_merge_query(
    node_type: NodeType, match_properties: dict[str, Any], set_properties: dict[str, Any]
) -> str:
    """Generate Cypher query for merging a node (create or update).

    Args:
        node_type: Type of node to merge
        match_properties: Properties to match on
        set_properties: Properties to set

    Returns:
        str: Cypher MERGE query
    """
    match_str = ", ".join([f"{k}: ${k}" for k in match_properties])
    set_str = ", ".join([f"n.{k} = ${k}" for k in set_properties])
    query = f"MERGE (n:{node_type.value} {{{match_str}}}) SET {set_str} RETURN n"
    return query


def get_relationship_creation_query(
    from_node_type: NodeType,
    from_node_id: str,
    relationship_type: RelationshipType,
    to_node_type: NodeType,
    to_node_id: str,
    properties: dict[str, Any] | None = None,
) -> str:
    """Generate Cypher query for creating a relationship.

    Args:
        from_node_type: Source node type
        from_node_id: Source node ID property name
        relationship_type: Relationship type
        to_node_type: Target node type
        to_node_id: Target node ID property name
        properties: Relationship properties (optional)

    Returns:
        str: Cypher CREATE relationship query
    """
    props_str = ""
    if properties:
        props_str = " {" + ", ".join([f"{k}: ${k}" for k in properties]) + "}"

    query = f"""
    MATCH (a:{from_node_type.value} {{{from_node_id}: $from_id}})
    MATCH (b:{to_node_type.value} {{{to_node_id}: $to_id}})
    CREATE (a)-[r:{relationship_type.value}{props_str}]->(b)
    RETURN r
    """
    return query.strip()


# Common query templates
CYPHER_QUERIES = {
    "find_equipment_manuals": """
        MATCH (e:Equipment {id: $equipment_id})-[:HAS_MANUAL]->(m:Manual)
        RETURN m
    """,
    "find_manual_sections": """
        MATCH (m:Manual {doc_id: $doc_id})-[:CONTAINS]->(s:Section)
        RETURN s ORDER BY s.page_range
    """,
    "find_component_procedures": """
        MATCH (c:Component {id: $component_id})-[:REQUIRES_PROCEDURE]->(p:Procedure)
        RETURN p
    """,
    "find_symptom_solutions": """
        MATCH (s:Symptom {id: $symptom_id})-[:SOLVED_BY]->(sol:Solution)
        RETURN sol
    """,
    "find_equipment_components": """
        MATCH (e:Equipment {id: $equipment_id})-[:HAS_COMPONENT]->(c:Component)
        RETURN c
    """,
    "find_related_procedures": """
        MATCH (p1:Procedure {id: $procedure_id})-[:PRECEDES*1..3]->(p2:Procedure)
        RETURN p2
    """,
    # Phase 1: Core Diagnostic Chain queries
    "find_failure_modes_from_symptoms": """
        MATCH (s:Symptom)-[ind:INDICATES]->(fm:FailureMode)
        WHERE s.description IN $symptoms
        OPTIONAL MATCH (fm)-[:POSES]->(sh:SafetyHazard)
        RETURN fm, ind.weight as confidence, collect(sh) as safety_hazards
        ORDER BY ind.weight DESC
    """,
    "find_resolution_procedures": """
        MATCH (fm:FailureMode {id: $failure_mode_id})
        MATCH (p:Procedure)-[res:RESOLVES]->(fm)
        OPTIONAL MATCH (p)-[req:REQUIRES]->(part:Part)
        OPTIONAL MATCH (p)-[:REQUIRES_TOOL]->(tool:Tool)
        OPTIONAL MATCH (p)-[:REQUIRES_SKILL]->(skill:Skill)
        RETURN p, res.weight as confidence,
               collect(DISTINCT {part: part.part_number, quantity: req.quantity}) as parts,
               collect(DISTINCT tool.tool_name) as tools,
               collect(DISTINCT skill.skill_name) as skills
        ORDER BY res.weight DESC
    """,
    "diagnostic_chain_full": """
        MATCH (eq:Equipment {id: $equipment_id})-[:EXHIBITS]->(s:Symptom)
        WHERE s.description IN $symptoms
        MATCH (s)-[ind:INDICATES]->(fm:FailureMode)
        MATCH (p:Procedure)-[res:RESOLVES]->(fm)
        OPTIONAL MATCH (fm)-[:POSES]->(sh:SafetyHazard)
        OPTIONAL MATCH (p)-[req:REQUIRES]->(part:Part)
        RETURN fm, p, sh,
               (ind.weight * res.weight) as combined_confidence,
               collect(DISTINCT {part: part.part_number, qty: req.quantity}) as required_parts
        ORDER BY combined_confidence DESC
        LIMIT 5
    """,
    "find_equipment_typical_symptoms": """
        MATCH (eq:Equipment {id: $equipment_id})-[ex:EXHIBITS]->(s:Symptom)
        RETURN s, ex.weight as frequency
        ORDER BY ex.weight DESC
    """,
    "find_safety_hazards_for_failure": """
        MATCH (fm:FailureMode {id: $failure_mode_id})-[p:POSES]->(sh:SafetyHazard)
        RETURN sh, p.weight as severity
        ORDER BY sh.severity DESC, p.weight DESC
    """,
    # Phase 2: Topology queries
    "find_cascading_failures": """
        MATCH path = (failed:Equipment {id: $failed_equipment_id})
              -[:FEEDS_INTO*1..3]->(downstream:Equipment)
        RETURN downstream.id, downstream.name,
               length(path) as hops
        ORDER BY hops
    """,
    "find_environmental_correlation": """
        MATCH (eq1:Equipment {id: $equipment_id})-[:LOCATED_IN]->(z:Zone)
              <-[:LOCATED_IN]-(eq2:Equipment)
        WHERE eq2.id <> $equipment_id
        MATCH (eq2)-[:EXHIBITS]->(s:Symptom)
        WHERE s.timestamp > datetime() - duration('P7D')
        RETURN eq2, s, z, count(s) as symptom_count
        ORDER BY symptom_count DESC
    """,
    # Phase 3: Resource constraint queries
    "check_procedure_resources": """
        MATCH (p:Procedure {id: $procedure_id})
        OPTIONAL MATCH (p)-[rp:REQUIRES]->(part:Part)
        OPTIONAL MATCH (p)-[:REQUIRES_TOOL]->(tool:Tool)
        OPTIONAL MATCH (p)-[:REQUIRES_SKILL]->(skill:Skill)
        RETURN
          collect(DISTINCT {
            part: part.part_number,
            description: part.description,
            qty_required: rp.quantity,
            in_stock: part.stock_level,
            available: CASE WHEN part.stock_level >= rp.quantity THEN true ELSE false END
          }) as parts,
          collect(DISTINCT {
            tool: tool.tool_name,
            requires_cert: tool.requires_certification
          }) as tools,
          collect(DISTINCT {
            skill: skill.skill_name,
            level: skill.certification_level
          }) as skills
    """,
}
