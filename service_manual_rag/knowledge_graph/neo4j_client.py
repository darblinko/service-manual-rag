"""Neo4J client for knowledge graph operations."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from neo4j import Driver, GraphDatabase, Session, Transaction
from neo4j.exceptions import ClientError, DriverError, Neo4jError, ServiceUnavailable

from service_manual_rag.config import get_settings
from service_manual_rag.utils.db_utils import retry_with_backoff
from service_manual_rag.utils.logging_config import get_logger

from .graph_schema import (
    CYPHER_QUERIES,
    NodeType,
    RelationshipType,
    create_constraints,
    create_indexes,
    get_node_creation_query,
    get_node_merge_query,
    get_relationship_creation_query,
)

__all__ = ["Neo4JManager", "NodeType", "RelationshipType"]

logger = get_logger(__name__)

# SET clauses shared by the continuous learning queries; both expect `r` and $timestamp.
STRENGTHEN_SET_CLAUSE = """
SET r.success_count = COALESCE(r.success_count, 0) + 1,
    r.evidence_count = COALESCE(r.evidence_count, 0) + 1,
    r.weight = CASE WHEN r.weight >= 0.95 THEN 0.95 ELSE r.weight + 0.05 END,
    r.last_confirmed = $timestamp
"""

WEAKEN_SET_CLAUSE = """
SET r.failure_count = COALESCE(r.failure_count, 0) + 1,
    r.evidence_count = COALESCE(r.evidence_count, 0) + 1,
    r.weight = CASE WHEN r.weight <= 0.1 THEN 0.1 ELSE r.weight - 0.1 END,
    r.last_confirmed = $timestamp
"""

# Adds a document and an equipment ID to a node's provenance lists without duplicates.
PROVENANCE_SET_CLAUSE = """
SET n.doc_ids = CASE
        WHEN $doc_id IS NULL OR $doc_id IN coalesce(n.doc_ids, []) THEN coalesce(n.doc_ids, [])
        ELSE coalesce(n.doc_ids, []) + $doc_id
    END,
    n.equipment_ids = CASE
        WHEN $equipment_id IN coalesce(n.equipment_ids, []) THEN coalesce(n.equipment_ids, [])
        ELSE coalesce(n.equipment_ids, []) + $equipment_id
    END
"""


class Neo4JManager:
    """Manager for Neo4J knowledge graph operations."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        """Initialize Neo4J manager.

        Args:
            uri: Neo4J URI (defaults to settings)
            user: Neo4J username (defaults to settings)
            password: Neo4J password (defaults to NEO4J_PASSWORD; raises
                MissingCredentialError when neither is set)
            database: Neo4J database name (defaults to settings)
        """
        self.settings = get_settings()
        self.uri = uri or self.settings.neo4j_uri
        self.user = user or self.settings.neo4j_user
        resolved_password = password or self.settings.require_neo4j_password()
        self.database = database or self.settings.neo4j_database

        logger.info(f"Initializing Neo4J client: {self.uri}")
        self.driver: Driver | None = None
        self._connect(resolved_password)

    def _connect(self, password: str) -> None:
        """Create the Neo4J driver."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, password))
            logger.info("Connected to Neo4J")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4J: {e}")
            raise

    @contextmanager
    def get_session(self) -> Iterator[Session]:
        """Get a Neo4J session.

        Yields:
            Session: Neo4J session

        Example:
            >>> with manager.get_session() as session:
            >>>     session.run("MATCH (n) RETURN n LIMIT 1")
        """
        if not self.driver:
            raise RuntimeError("Neo4J driver not initialized")

        session = self.driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()

    @retry_with_backoff(max_attempts=3, exceptions=(ServiceUnavailable, ConnectionError))
    def health_check(self) -> bool:
        """Check if Neo4J is healthy.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            with self.get_session() as session:
                result = session.run("RETURN 1 AS num")
                value = result.single()["num"]
                if value == 1:
                    logger.info("Neo4J is healthy")
                    return True
                return False
        except (Neo4jError, DriverError) as e:
            logger.error(f"Neo4J health check failed: {e}")
            return False

    def initialize_schema(self) -> bool:
        """Create the graph constraints and indexes.

        Rules that already exist are skipped; any other Neo4J error is raised.

        Returns:
            bool: True once every constraint and index exists
        """
        with self.get_session() as session:
            logger.info("Creating constraints")
            for constraint in create_constraints():
                self._run_schema_statement(session, constraint)

            logger.info("Creating indexes")
            for index in create_indexes():
                self._run_schema_statement(session, index)

        logger.info("Schema initialized")
        return True

    @staticmethod
    def _run_schema_statement(session: Session, statement: str) -> None:
        """Run one CREATE CONSTRAINT or CREATE INDEX statement, skipping existing rules."""
        try:
            session.run(statement).consume()
            logger.debug(f"Created schema rule: {statement[:50]}")
        except ClientError as error:
            if "AlreadyExists" not in (error.code or ""):
                raise
            logger.debug(f"Schema rule already exists: {statement[:50]}")

    def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        write: bool = False,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query.

        Args:
            query: Cypher query string
            parameters: Query parameters
            write: If True, use write transaction

        Returns:
            list: Query results as list of dictionaries

        Raises:
            Neo4jError, DriverError: The query failed; it is logged first.

        Example:
            >>> results = manager.execute_query(
            >>>     "MATCH (n:Equipment {oem: $oem}) RETURN n",
            >>>     parameters={"oem": "ExampleCo"}
            >>> )
        """
        parameters = parameters or {}

        try:
            with self.get_session() as session:
                if write:
                    result = session.execute_write(
                        lambda tx: self._run_query(tx, query, parameters)
                    )
                else:
                    result = session.execute_read(lambda tx: self._run_query(tx, query, parameters))

                logger.debug(f"Query returned {len(result)} results")
                return result

        except Exception as error:
            logger.error(f"Error executing query: {error}. Query: {query.strip()}")
            raise

    @staticmethod
    def _run_query(tx: Transaction, query: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        """Run query within a transaction.

        Args:
            tx: Neo4J transaction
            query: Cypher query
            parameters: Query parameters

        Returns:
            list: Query results
        """
        result = tx.run(query, parameters)
        return [record.data() for record in result]

    def create_node(self, node_type: NodeType, properties: dict[str, Any]) -> dict[str, Any] | None:
        """Create a new node.

        Args:
            node_type: Type of node to create
            properties: Node properties

        Returns:
            Optional[dict]: Created node or None

        Example:
            >>> node = manager.create_node(
            >>>     NodeType.EQUIPMENT,
            >>>     {"id": "EXAMPLECO_CP-200", "oem": "ExampleCo", "model": "CP-200"}
            >>> )
        """
        query = get_node_creation_query(node_type, properties)
        results = self.execute_query(query, properties, write=True)

        if results:
            logger.debug(f"Created {node_type} node: {properties.get('id')}")
            return results[0].get("n")
        return None

    def merge_node(
        self,
        node_type: NodeType,
        match_properties: dict[str, Any],
        set_properties: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Merge a node (create or update).

        Args:
            node_type: Type of node to merge
            match_properties: Properties to match on
            set_properties: Properties to set (defaults to match_properties)

        Returns:
            Optional[dict]: Merged node or None

        Example:
            >>> node = manager.merge_node(
            >>>     NodeType.EQUIPMENT,
            >>>     {"id": "EXAMPLECO_CP-200"},
            >>>     {"oem": "ExampleCo", "model": "CP-200", "category": "Centrifugal Pump"}
            >>> )
        """
        set_props = set_properties or match_properties
        all_props = {**match_properties, **set_props}

        query = get_node_merge_query(node_type, match_properties, set_props)
        results = self.execute_query(query, all_props, write=True)

        if results:
            logger.debug(f"Merged {node_type} node: {match_properties}")
            return results[0].get("n")
        return None

    def get_node(self, node_type: NodeType, node_id: str) -> dict[str, Any] | None:
        """Get a node by ID.

        Args:
            node_type: Type of node to retrieve
            node_id: Node ID

        Returns:
            Optional[dict]: Node data or None

        Example:
            >>> node = manager.get_node(NodeType.EQUIPMENT, "EXAMPLECO_CP-200")
        """
        query = f"MATCH (n:{node_type.value} {{id: $id}}) RETURN n"
        results = self.execute_query(query, {"id": node_id})

        if results:
            return results[0].get("n")
        return None

    def delete_node(self, node_type: NodeType, node_id: str) -> bool:
        """Delete a node and all of its relationships by ID.

        Args:
            node_type: Type of node to delete
            node_id: Node ID

        Returns:
            bool: True if successful

        Example:
            >>> manager.delete_node(NodeType.EQUIPMENT, "EXAMPLECO_CP-200")
        """
        query = f"""
        MATCH (n:{node_type.value} {{id: $id}})
        DETACH DELETE n
        """
        self.execute_query(query, {"id": node_id}, write=True)
        logger.info(f"Deleted {node_type} node: {node_id}")
        return True

    def create_relationship(
        self,
        from_node_type: NodeType,
        from_node_id: str,
        relationship_type: RelationshipType,
        to_node_type: NodeType,
        to_node_id: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """Create a relationship between two nodes.

        Args:
            from_node_type: Source node type
            from_node_id: Source node ID
            relationship_type: Relationship type
            to_node_type: Target node type
            to_node_id: Target node ID
            properties: Relationship properties (optional)

        Returns:
            bool: True if successful

        Example:
            >>> manager.create_relationship(
            >>>     NodeType.EQUIPMENT, "EXAMPLECO_CP-200",
            >>>     RelationshipType.HAS_MANUAL,
            >>>     NodeType.MANUAL, "manual_001"
            >>> )
        """
        query = get_relationship_creation_query(
            from_node_type,
            "id",
            relationship_type,
            to_node_type,
            "id",
            properties,
        )

        params = {
            "from_id": from_node_id,
            "to_id": to_node_id,
            **(properties or {}),
        }

        results = self.execute_query(query, params, write=True)

        if results:
            logger.debug(
                f"Created relationship: {from_node_id} -[{relationship_type.value}]-> {to_node_id}"
            )
            return True
        return False

    def add_provenance(
        self, node_type: NodeType, node_id: str, doc_id: str | None, equipment_id: str
    ) -> None:
        """Record that a node was extracted from a document about the given equipment."""
        query = f"MATCH (n:{node_type.value} {{id: $id}})" + PROVENANCE_SET_CLAUSE
        self.execute_query(
            query, {"id": node_id, "doc_id": doc_id, "equipment_id": equipment_id}, write=True
        )

    def find_nodes(
        self,
        node_type: NodeType,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Find nodes with optional filters.

        Args:
            node_type: Type of nodes to find
            filters: Property filters
            limit: Maximum number of results

        Returns:
            list: List of nodes

        Example:
            >>> nodes = manager.find_nodes(
            >>>     NodeType.COMPONENT,
            >>>     filters={"part_number": "12345"}
            >>> )
        """
        if filters:
            where_clauses = [f"n.{k} = ${k}" for k in filters]
            where_str = " AND ".join(where_clauses)
            query = f"MATCH (n:{node_type.value}) WHERE {where_str} RETURN n LIMIT {limit}"
        else:
            query = f"MATCH (n:{node_type.value}) RETURN n LIMIT {limit}"

        results = self.execute_query(query, filters or {})
        return [r.get("n") for r in results if r.get("n")]

    def traverse_relationships(
        self,
        from_node_id: str,
        from_node_type: NodeType,
        relationship_type: RelationshipType,
        max_depth: int = 1,
    ) -> list[dict[str, Any]]:
        """Traverse relationships from a node.

        Args:
            from_node_id: Starting node ID
            from_node_type: Starting node type
            relationship_type: Relationship type to traverse
            max_depth: Maximum traversal depth

        Returns:
            list: List of related nodes

        Example:
            >>> related = manager.traverse_relationships(
            >>>     "EXAMPLECO_CP-200",
            >>>     NodeType.EQUIPMENT,
            >>>     RelationshipType.HAS_COMPONENT,
            >>>     max_depth=2
            >>> )
        """
        depth_str = f"*1..{max_depth}" if max_depth > 1 else ""
        query = f"""
        MATCH (a:{from_node_type.value} {{id: $from_id}})-[:{relationship_type.value}{depth_str}]->(b)
        RETURN DISTINCT b
        """

        results = self.execute_query(query, {"from_id": from_node_id})
        return [r.get("b") for r in results if r.get("b")]

    def execute_template_query(
        self, template_name: str, parameters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Execute a pre-defined query template.

        Args:
            template_name: Name of the query template
            parameters: Query parameters

        Returns:
            list: Query results

        Example:
            >>> results = manager.execute_template_query(
            >>>     "find_equipment_manuals",
            >>>     {"equipment_id": "EXAMPLECO_CP-200"}
            >>> )
        """
        if template_name not in CYPHER_QUERIES:
            raise ValueError(f"Unknown query template: {template_name}")

        query = CYPHER_QUERIES[template_name]
        return self.execute_query(query, parameters)

    def count_nodes(self, node_type: NodeType) -> int:
        """Count nodes of a specific type.

        Args:
            node_type: Type of nodes to count

        Returns:
            int: Number of nodes
        """
        query = f"MATCH (n:{node_type.value}) RETURN count(n) as count"
        results = self.execute_query(query)

        if results:
            return results[0].get("count", 0)
        return 0

    def clear_database(self, confirm: bool = False) -> bool:
        """Delete all nodes and relationships from the database.

        Args:
            confirm: Must be True to execute

        Returns:
            bool: True once cleared, False when confirm is not True
        """
        if not confirm:
            logger.warning("Database clear requires confirmation")
            return False

        with self.get_session() as session:
            session.run("MATCH (n) DETACH DELETE n").consume()
        logger.warning("Neo4J database cleared")
        return True

    def get_statistics(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            dict: Node counts per type plus total node and relationship counts
        """
        stats = {f"{node_type.value}_count": self.count_nodes(node_type) for node_type in NodeType}

        results = self.execute_query("MATCH (n) RETURN count(n) as total_nodes")
        stats["total_nodes"] = results[0].get("total_nodes", 0) if results else 0

        results = self.execute_query("MATCH ()-[r]->() RETURN count(r) as total_relationships")
        stats["total_relationships"] = results[0].get("total_relationships", 0) if results else 0

        return stats

    def update_relationship_weight(
        self,
        from_node_type: NodeType,
        from_node_id: str,
        relationship_type: RelationshipType,
        to_node_type: NodeType,
        to_node_id: str,
        success: bool,
    ) -> bool:
        """Update relationship weight based on outcome (continuous learning).

        Success adds 0.05 to the weight (capped at 0.95). Failure subtracts 0.1
        (floored at 0.1).

        Args:
            from_node_type: Source node type
            from_node_id: Source node ID
            relationship_type: Relationship type
            to_node_type: Target node type
            to_node_id: Target node ID
            success: True if successful outcome, False if failed

        Returns:
            bool: True if successful

        Example:
            >>> manager.update_relationship_weight(
            >>>     NodeType.SYMPTOM, "symp_abc123",
            >>>     RelationshipType.SOLVED_BY,
            >>>     NodeType.SOLUTION, "sol_xyz789",
            >>>     success=True
            >>> )
        """
        set_clause = STRENGTHEN_SET_CLAUSE if success else WEAKEN_SET_CLAUSE
        query = (
            f"MATCH (a:{from_node_type.value} {{id: $from_id}})"
            f"-[r:{relationship_type.value}]->(b:{to_node_type.value} {{id: $to_id}})"
            f"{set_clause}RETURN r"
        )
        params = {
            "from_id": from_node_id,
            "to_id": to_node_id,
            "timestamp": datetime.now().isoformat(),
        }

        results = self.execute_query(query, params, write=True)
        description = f"{from_node_id} -[{relationship_type.value}]-> {to_node_id}"

        if results:
            action = "Strengthened" if success else "Weakened"
            logger.info(f"{action} relationship: {description}")
            return True
        logger.warning(f"Could not update relationship weight: {description}")
        return False

    def strengthen_diagnostic_path(
        self,
        symptom_description: str,
        solution_description: str,
        equipment_id: str | None = None,
    ) -> bool:
        """Strengthen a diagnostic path when a solution successfully resolves a symptom.

        Args:
            symptom_description: Description of the symptom
            solution_description: Description of the solution
            equipment_id: Optional equipment ID to narrow search

        Returns:
            bool: True if path was strengthened

        Example:
            >>> manager.strengthen_diagnostic_path(
            >>>     "pump loses discharge pressure",
            >>>     "replace mechanical seal",
            >>>     equipment_id="EXAMPLECO_CP-200"
            >>> )
        """
        equipment_match = (
            "MATCH (e:Equipment {id: $equipment_id})-[:HAS_COMPONENT]->(c:Component)\n"
            if equipment_id
            else ""
        )
        query = (
            f"{equipment_match}"
            "MATCH (s:Symptom)-[r:SOLVED_BY]->(sol:Solution)\n"
            "WHERE s.description CONTAINS $symptom_desc\n"
            "  AND sol.fix_description CONTAINS $solution_desc"
            f"{STRENGTHEN_SET_CLAUSE}"
            "RETURN count(r) as updated_count"
        )
        params = {
            "symptom_desc": symptom_description.lower(),
            "solution_desc": solution_description.lower(),
            "timestamp": datetime.now().isoformat(),
        }
        if equipment_id:
            params["equipment_id"] = equipment_id

        results = self.execute_query(query, params, write=True)
        path = f"'{symptom_description}' to '{solution_description}'"

        if results and results[0].get("updated_count", 0) > 0:
            count = results[0]["updated_count"]
            logger.info(f"Strengthened {count} diagnostic path(s): {path}")
            return True
        logger.warning(f"No diagnostic paths found to strengthen for: {path}")
        return False

    def get_weighted_solutions(
        self, symptom_description: str, equipment_id: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get solutions for a symptom, ordered by relationship weight (continuous learning).

        Args:
            symptom_description: Description of the symptom
            equipment_id: Optional equipment ID to narrow search
            limit: Maximum number of solutions to return

        Returns:
            list: Solutions with weights and metadata

        Example:
            >>> solutions = manager.get_weighted_solutions(
            >>>     "pump loses discharge pressure",
            >>>     equipment_id="EXAMPLECO_CP-200"
            >>> )
        """
        equipment_match = "MATCH (e:Equipment {id: $equipment_id})\n" if equipment_id else ""
        query = (
            f"{equipment_match}"
            "MATCH (s:Symptom)-[r:SOLVED_BY]->(sol:Solution)\n"
            "WHERE s.description CONTAINS $symptom_desc\n"
            "RETURN sol, r.weight as weight, r.success_count as success_count,\n"
            "       r.failure_count as failure_count, r.evidence_count as evidence_count\n"
            "ORDER BY r.weight DESC, r.success_count DESC\n"
            "LIMIT $limit"
        )
        params = {
            "symptom_desc": symptom_description.lower(),
            "limit": limit,
        }
        if equipment_id:
            params["equipment_id"] = equipment_id

        return [
            {
                "solution": result.get("sol"),
                "weight": result.get("weight", 0.5),
                "success_count": result.get("success_count", 0),
                "failure_count": result.get("failure_count", 0),
                "evidence_count": result.get("evidence_count", 0),
            }
            for result in self.execute_query(query, params)
        ]

    def get_onboarded_equipment(self) -> list[dict[str, str]]:
        """Get list of all onboarded equipment (unique OEM/Model combinations).

        Returns:
            list: List of equipment dictionaries with 'id', 'oem', 'model', 'category'

        Example:
            >>> manager.get_onboarded_equipment()
            [{'id': 'EXAMPLECO_CP-200', 'oem': 'ExampleCo', 'model': 'CP-200',
              'category': 'Centrifugal Pump'}]
        """
        query = """
        MATCH (e:Equipment)
        RETURN e.id as id, e.oem as oem, e.model as model, e.category as category
        ORDER BY e.oem, e.model
        """

        equipment_list = [
            {
                "id": result.get("id", ""),
                "oem": result.get("oem", ""),
                "model": result.get("model", ""),
                "category": result.get("category", "Unknown"),
            }
            for result in self.execute_query(query)
            if result.get("oem") and result.get("model")
        ]

        logger.info(f"Retrieved {len(equipment_list)} onboarded equipment items")
        return equipment_list

    def close(self) -> None:
        """Close the Neo4J driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4J driver")
