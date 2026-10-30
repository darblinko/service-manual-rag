"""Populate Neo4J knowledge graph from extracted entities."""

import asyncio
from collections import Counter
from typing import Any
from uuid import uuid4

from service_manual_rag.knowledge_graph import Neo4JManager, NodeType, RelationshipType
from service_manual_rag.knowledge_graph.graph_schema import get_initial_relationship_properties
from service_manual_rag.processing.hierarchical_chunker import Chunk
from service_manual_rag.utils.logging_config import get_logger

from .entity_extractor import EntityExtractor

logger = get_logger(__name__)

STAT_KEYS = (
    "equipment_created",
    "manuals_created",
    "sections_created",
    "components_created",
    "procedures_created",
    "symptoms_created",
    "solutions_created",
    "failure_modes_created",
    "safety_hazards_created",
    "parts_created",
    "tools_created",
    "skills_created",
    "relationships_created",
)


def equipment_id_for(chunk: Chunk) -> str:
    """Equipment node ID built from OEM and model, for example "EXAMPLECO_CP-200"."""
    return f"{chunk.oem.upper()}_{chunk.model.upper()}"


def new_node_id(prefix: str) -> str:
    """Random node ID with a type prefix, for example "comp_1a2b3c4d5e6f"."""
    return f"{prefix}_{uuid4().hex[:12]}"


def solution_key(fix_description: str) -> str:
    """Case and whitespace insensitive key that matches a symptom's solution to a Solution."""
    return " ".join(fix_description.lower().split())


class GraphPopulator:
    """Populate Neo4J knowledge graph with parallel processing."""

    def __init__(
        self,
        neo4j_manager: Neo4JManager | None = None,
        entity_extractor: EntityExtractor | None = None,
    ):
        """Initialize graph populator.

        Args:
            neo4j_manager: Neo4J manager instance
            entity_extractor: Entity extractor instance
        """
        self.neo4j = neo4j_manager or Neo4JManager()
        self.extractor = entity_extractor or EntityExtractor()
        self._runner: asyncio.Runner | None = None

        logger.info("GraphPopulator initialized")

    def populate(self, chunks: list[Chunk]) -> dict:
        """Run populate_from_chunks to completion from synchronous code.

        Every call runs on the same event loop, because the extractor's async OpenAI
        client keeps connections bound to the loop that opened them.
        """
        if self._runner is None:
            self._runner = asyncio.Runner()
        return self._runner.run(self.populate_from_chunks(chunks))

    def close(self) -> None:
        """Close the event loop used by populate and the Neo4j driver."""
        if self._runner is not None:
            self._runner.close()
            self._runner = None
        self.neo4j.close()

    async def populate_from_chunks(self, chunks: list[Chunk]) -> dict:
        """Populate graph from chunks with parallel entity extraction.

        Args:
            chunks: List of chunks to process

        Returns:
            dict: Population statistics, one count per key in STAT_KEYS
        """
        logger.info(f"Populating graph from {len(chunks)} chunks")

        doc_chunks = [c for c in chunks if c.level == 1]
        section_chunks = [c for c in chunks if c.level in [2, 3]]
        semantic_chunks = [c for c in chunks if c.level == 4]
        logger.info(
            f"Processing {len(doc_chunks)} document chunks, {len(section_chunks)} section "
            f"chunks and {len(semantic_chunks)} semantic chunks"
        )

        stats = Counter(dict.fromkeys(STAT_KEYS, 0))

        logger.info("Step 0/5: Creating Equipment and Manual nodes")
        stats.update(self._create_document_nodes(doc_chunks))

        logger.info("Step 1/5: Creating section nodes")
        stats.update(self._create_section_nodes(section_chunks))

        logger.info(
            f"Step 2/5: Extracting high-level entities from {len(section_chunks)} section chunks"
        )
        section_entities = await self.extractor.extract_entities_batch(section_chunks)
        logger.info("Step 2a/5: Creating high-level entity nodes from sections")
        stats.update(self._create_section_entities(section_chunks, section_entities))

        logger.info(
            f"Step 3/5: Extracting detailed entities from {len(semantic_chunks)} semantic chunks"
        )
        semantic_entities = await self.extractor.extract_entities_batch(semantic_chunks)
        logger.info("Step 4/5: Creating detailed entity nodes and relationships")
        stats.update(self._create_semantic_entities(semantic_chunks, semantic_entities))

        result = {key: stats[key] for key in STAT_KEYS}
        logger.info(f"Graph population complete: {result}")
        return result

    def _create_document_nodes(self, doc_chunks: list[Chunk]) -> Counter:
        """Create Equipment and Manual nodes for each level 1 chunk."""
        counts = Counter()
        if not doc_chunks:
            logger.warning(
                "No level 1 document chunks found. Equipment and Manual nodes will not be "
                "created, so the Documents page and the Equipment dropdown will be empty."
            )

        for doc_chunk in doc_chunks:
            logger.info(
                f"Creating Equipment and Manual for: OEM={doc_chunk.oem}, "
                f"Model={doc_chunk.model}, DocID={doc_chunk.doc_id}"
            )
            self._create_equipment_and_manual_nodes(doc_chunk)
            counts.update(equipment_created=1, manuals_created=1, relationships_created=1)
        return counts

    def _create_section_nodes(self, section_chunks: list[Chunk]) -> Counter:
        """Create a Section node per level 2 or 3 chunk and link it to its manual."""
        counts = Counter()
        for chunk in section_chunks:
            _, relationship_created = self._create_section_node(chunk)
            counts["sections_created"] += 1
            counts["relationships_created"] += int(relationship_created)
        return counts

    def _create_section_entities(
        self, chunks: list[Chunk], entities_list: list[dict[str, Any]]
    ) -> Counter:
        """Create components and safety hazards extracted from section chunks."""
        counts = Counter()
        total = len(chunks)
        for index, (chunk, entities) in enumerate(zip(chunks, entities_list, strict=True), start=1):
            counts.update(self._create_components(entities, chunk))
            counts.update(self._create_standalone_hazards(entities, chunk))

            if index % 10 == 0 or index == total:
                logger.info(
                    f"Section entity creation progress: {index}/{total} ({index * 100 // total}%)"
                )
        return counts

    def _create_semantic_entities(
        self, chunks: list[Chunk], entities_list: list[dict[str, Any]]
    ) -> Counter:
        """Create all detailed entities and relationships extracted from semantic chunks."""
        counts = Counter()
        total = len(chunks)
        for index, (chunk, entities) in enumerate(zip(chunks, entities_list, strict=True), start=1):
            counts.update(self._create_components(entities, chunk))
            counts.update(self._create_procedures(entities, chunk))
            failure_mode_ids, failure_mode_counts = self._create_failure_modes(entities, chunk)
            counts.update(failure_mode_counts)
            counts.update(self._create_standalone_hazards(entities, chunk))
            solution_ids = self._create_solutions(entities, chunk)
            counts["solutions_created"] += len(solution_ids)
            counts.update(self._create_symptoms(entities, chunk, failure_mode_ids, solution_ids))
            counts.update(self._create_standalone_resources(entities, chunk))

            if index % 100 == 0 or index == total:
                logger.info(
                    f"Detailed node creation progress: {index}/{total} ({index * 100 // total}%)"
                )
        return counts

    def _create_components(self, entities: dict[str, Any], chunk: Chunk) -> Counter:
        """Create Component nodes linked to the chunk's equipment."""
        counts = Counter()
        for component in entities.get("components", []):
            _, relationship_created = self._create_component_node(component, chunk)
            counts["components_created"] += 1
            counts["relationships_created"] += int(relationship_created)
        return counts

    def _create_standalone_hazards(self, entities: dict[str, Any], chunk: Chunk) -> Counter:
        """Create SafetyHazard nodes that are not tied to a failure mode."""
        counts = Counter()
        for hazard in entities.get("safety_hazards", []):
            self._create_safety_hazard_node(hazard, chunk)
            counts["safety_hazards_created"] += 1
        return counts

    def _create_procedures(self, entities: dict[str, Any], chunk: Chunk) -> Counter:
        """Create Procedure nodes and link each to its required parts, tools and skills."""
        counts = Counter()
        for procedure in entities.get("procedures", []):
            procedure_id = self._create_procedure_node(procedure, chunk)
            counts["procedures_created"] += 1
            counts.update(self._link_procedure_resources(procedure, procedure_id, chunk))
        return counts

    def _link_procedure_resources(
        self, procedure: dict[str, Any], procedure_id: str, chunk: Chunk
    ) -> Counter:
        """Create Part, Tool and Skill nodes named by a procedure and link them to it."""
        counts = Counter()
        for part_number in procedure.get("required_parts", []):
            part_id = self._create_part_node({"part_number": part_number, "description": ""}, chunk)
            self._link(
                NodeType.PROCEDURE,
                procedure_id,
                RelationshipType.REQUIRES,
                NodeType.PART,
                part_id,
                extra_properties={"quantity": 1},
            )
            counts.update(parts_created=1, relationships_created=1)

        for tool_name in procedure.get("required_tools", []):
            tool_id = self._create_tool_node({"tool_name": tool_name, "tool_type": "common"}, chunk)
            self._link(
                NodeType.PROCEDURE,
                procedure_id,
                RelationshipType.REQUIRES_TOOL,
                NodeType.TOOL,
                tool_id,
            )
            counts.update(tools_created=1, relationships_created=1)

        for skill_name in procedure.get("required_skills", []):
            skill_id = self._create_skill_node(
                {"skill_name": skill_name, "certification_level": "basic"}, chunk
            )
            self._link(
                NodeType.PROCEDURE,
                procedure_id,
                RelationshipType.REQUIRES_SKILL,
                NodeType.SKILL,
                skill_id,
            )
            counts.update(skills_created=1, relationships_created=1)
        return counts

    def _create_failure_modes(
        self, entities: dict[str, Any], chunk: Chunk
    ) -> tuple[dict[str, str], Counter]:
        """Create FailureMode nodes and the safety hazards each one poses.

        Returns:
            tuple: (failure mode IDs keyed by name, counts)
        """
        failure_mode_ids: dict[str, str] = {}
        counts = Counter()
        for failure_mode in entities.get("failure_modes", []):
            failure_mode_id = self._create_failure_mode_node(failure_mode, chunk)
            failure_mode_ids[failure_mode.get("name", "")] = failure_mode_id
            counts["failure_modes_created"] += 1

            for hazard_description in failure_mode.get("safety_hazards", []):
                hazard = {
                    "hazard_type": "mechanical",
                    "warning": hazard_description,
                    "severity": "warning",
                }
                hazard_id = self._create_safety_hazard_node(hazard, chunk)
                self._link(
                    NodeType.FAILURE_MODE,
                    failure_mode_id,
                    RelationshipType.POSES,
                    NodeType.SAFETY_HAZARD,
                    hazard_id,
                )
                counts.update(safety_hazards_created=1, relationships_created=1)
        return failure_mode_ids, counts

    def _create_solutions(self, entities: dict[str, Any], chunk: Chunk) -> dict[str, str]:
        """Create one Solution node per extracted solution; return their IDs by solution_key."""
        solution_ids: dict[str, str] = {}
        for solution in entities.get("solutions", []):
            key = solution_key(solution.get("fix_description", ""))
            if key and key not in solution_ids:
                solution_ids[key] = self._create_solution_node(solution, chunk)
        return solution_ids

    def _create_symptoms(
        self,
        entities: dict[str, Any],
        chunk: Chunk,
        failure_mode_ids: dict[str, str],
        solution_ids: dict[str, str],
    ) -> Counter:
        """Create Symptom nodes linked to equipment, failure modes and their own solutions.

        A symptom is linked only to the solutions it names in its "solutions" list.
        """
        counts = Counter()
        equipment_id = equipment_id_for(chunk)
        for symptom in entities.get("symptoms", []):
            symptom_id = self._create_symptom_node(symptom, chunk)
            counts["symptoms_created"] += 1

            self._link(
                NodeType.EQUIPMENT,
                equipment_id,
                RelationshipType.EXHIBITS,
                NodeType.SYMPTOM,
                symptom_id,
            )
            counts["relationships_created"] += 1

            for failure_mode_name in symptom.get("failure_modes", []):
                if failure_mode_name in failure_mode_ids:
                    self._link(
                        NodeType.SYMPTOM,
                        symptom_id,
                        RelationshipType.INDICATES,
                        NodeType.FAILURE_MODE,
                        failure_mode_ids[failure_mode_name],
                    )
                    counts["relationships_created"] += 1

            for fix_description in symptom.get("solutions", []):
                solution_id = solution_ids.get(solution_key(fix_description))
                if solution_id is None:
                    continue
                self._link(
                    NodeType.SYMPTOM,
                    symptom_id,
                    RelationshipType.SOLVED_BY,
                    NodeType.SOLUTION,
                    solution_id,
                )
                counts["relationships_created"] += 1
        return counts

    def _create_standalone_resources(self, entities: dict[str, Any], chunk: Chunk) -> Counter:
        """Create Part, Tool and Skill nodes not tied to a procedure."""
        counts = Counter()
        for part in entities.get("parts", []):
            self._create_part_node(part, chunk)
            counts["parts_created"] += 1
        for tool in entities.get("tools", []):
            self._create_tool_node(tool, chunk)
            counts["tools_created"] += 1
        for skill in entities.get("skills", []):
            self._create_skill_node(skill, chunk)
            counts["skills_created"] += 1
        return counts

    def _link(
        self,
        from_type: NodeType,
        from_id: str,
        relationship_type: RelationshipType,
        to_type: NodeType,
        to_id: str,
        extra_properties: dict[str, Any] | None = None,
    ) -> bool:
        """Create a relationship carrying the initial learning properties for its type."""
        properties = {
            **get_initial_relationship_properties(relationship_type),
            **(extra_properties or {}),
        }
        return self.neo4j.create_relationship(
            from_type, from_id, relationship_type, to_type, to_id, properties=properties
        )

    def _record_source(self, node_type: NodeType, node_id: str, chunk: Chunk) -> None:
        """Add the chunk's document and equipment to the node's doc_ids and equipment_ids.

        Graph search filters on these lists and document deletion removes them.
        """
        self.neo4j.add_provenance(node_type, node_id, chunk.doc_id, equipment_id_for(chunk))

    def _create_equipment_and_manual_nodes(self, doc_chunk: Chunk) -> tuple[str, str]:
        """Create Equipment and Manual nodes from document chunk.

        Args:
            doc_chunk: Document-level chunk (Level 1)

        Returns:
            tuple: (equipment_id, manual_id)
        """
        equipment_id = equipment_id_for(doc_chunk)
        self.neo4j.merge_node(
            NodeType.EQUIPMENT,
            match_properties={"id": equipment_id},
            set_properties={
                "id": equipment_id,
                "oem": doc_chunk.oem,
                "model": doc_chunk.model,
                "category": "equipment",
            },
        )

        manual_id = doc_chunk.doc_id or doc_chunk.chunk_id
        self.neo4j.merge_node(
            NodeType.MANUAL,
            match_properties={"id": manual_id},
            set_properties={
                "id": manual_id,
                "doc_id": manual_id,
                "title": f"{doc_chunk.oem} {doc_chunk.model} Service Manual",
                "file_path": doc_chunk.file_path,
                "manual_type": "service",
            },
        )

        self._record_source(NodeType.EQUIPMENT, equipment_id, doc_chunk)
        self._record_source(NodeType.MANUAL, manual_id, doc_chunk)
        self._link(
            NodeType.EQUIPMENT,
            equipment_id,
            RelationshipType.HAS_MANUAL,
            NodeType.MANUAL,
            manual_id,
        )

        logger.debug(f"Created Equipment ({equipment_id}) and Manual ({manual_id}) nodes")
        return equipment_id, manual_id

    def _create_section_node(self, chunk: Chunk) -> tuple[str, bool]:
        """Create section node and link to manual.

        Returns:
            tuple: (section_id, relationship_created)
        """
        section_id = f"section_{chunk.chunk_id}"
        page_range = (
            f"{min(chunk.page_numbers)}-{max(chunk.page_numbers)}" if chunk.page_numbers else "N/A"
        )

        self.neo4j.merge_node(
            NodeType.SECTION,
            match_properties={"id": section_id},
            set_properties={
                "id": section_id,
                "name": chunk.section_name or "Unknown",
                "type": chunk.element_type,
                "page_range": page_range,
                "level": int(chunk.level),
            },
        )
        self._record_source(NodeType.SECTION, section_id, chunk)

        if not chunk.doc_id:
            logger.warning(f"Section {section_id} has no doc_id; CONTAINS relationship not created")
            return section_id, False

        logger.debug(
            f"Creating CONTAINS relationship: Manual[{chunk.doc_id}] -> Section[{section_id}]"
        )
        # chunk.doc_id is the Manual node's "id" property.
        success = self._link(
            NodeType.MANUAL,
            chunk.doc_id,
            RelationshipType.CONTAINS,
            NodeType.SECTION,
            section_id,
        )
        if not success:
            logger.error(
                f"Failed to create CONTAINS relationship: "
                f"Manual[{chunk.doc_id}] -> Section[{section_id}]"
            )
        return section_id, success

    def _create_component_node(self, component: dict, chunk: Chunk) -> tuple[str, bool]:
        """Create component node and link to equipment.

        Returns:
            tuple: (component_id, relationship_created)
        """
        comp_id = new_node_id("comp")

        self.neo4j.create_node(
            NodeType.COMPONENT,
            {
                "id": comp_id,
                "part_name": component.get("part_name", ""),
                "part_number": component.get("part_number", ""),
                "description": component.get("description", ""),
            },
        )
        self._record_source(NodeType.COMPONENT, comp_id, chunk)

        success = self._link(
            NodeType.EQUIPMENT,
            equipment_id_for(chunk),
            RelationshipType.HAS_COMPONENT,
            NodeType.COMPONENT,
            comp_id,
        )
        return comp_id, success

    def _create_procedure_node(self, procedure: dict, chunk: Chunk) -> str:
        """Create procedure node."""
        proc_id = new_node_id("proc")
        self.neo4j.create_node(
            NodeType.PROCEDURE,
            {
                "id": proc_id,
                "action_type": procedure.get("action_type", ""),
                "steps": procedure.get("steps", ""),
                "difficulty": procedure.get("difficulty", "medium"),
            },
        )
        self._record_source(NodeType.PROCEDURE, proc_id, chunk)
        return proc_id

    def _create_symptom_node(self, symptom: dict, chunk: Chunk) -> str:
        """Create symptom node."""
        symp_id = new_node_id("symp")
        self.neo4j.create_node(
            NodeType.SYMPTOM,
            {
                "id": symp_id,
                "description": symptom.get("description", ""),
                "severity": symptom.get("severity", "medium"),
            },
        )
        self._record_source(NodeType.SYMPTOM, symp_id, chunk)
        return symp_id

    def _create_solution_node(self, solution: dict, chunk: Chunk) -> str:
        """Create solution node."""
        sol_id = new_node_id("sol")
        self.neo4j.create_node(
            NodeType.SOLUTION,
            {
                "id": sol_id,
                "fix_description": solution.get("fix_description", ""),
                "parts_needed": solution.get("parts_needed", []),
            },
        )
        self._record_source(NodeType.SOLUTION, sol_id, chunk)
        return sol_id

    def _create_failure_mode_node(self, failure_mode: dict, chunk: Chunk) -> str:
        """Create failure mode node."""
        fm_id = new_node_id("fm")
        self.neo4j.create_node(
            NodeType.FAILURE_MODE,
            {
                "id": fm_id,
                "name": failure_mode.get("name", ""),
                "description": failure_mode.get("description", ""),
                "category": failure_mode.get("category", "mechanical"),
                "severity": failure_mode.get("severity", "medium"),
            },
        )
        self._record_source(NodeType.FAILURE_MODE, fm_id, chunk)
        return fm_id

    def _create_safety_hazard_node(self, hazard: dict, chunk: Chunk) -> str:
        """Create safety hazard node."""
        hazard_id = new_node_id("hazard")
        self.neo4j.create_node(
            NodeType.SAFETY_HAZARD,
            {
                "id": hazard_id,
                "hazard_type": hazard.get("hazard_type", "mechanical"),
                "warning": hazard.get("warning", ""),
                "severity": hazard.get("severity", "warning"),
                "required_ppe": hazard.get("required_ppe", ""),
            },
        )
        self._record_source(NodeType.SAFETY_HAZARD, hazard_id, chunk)
        return hazard_id

    def _create_part_node(self, part: dict, chunk: Chunk) -> str:
        """Create part node, merging on part number when one is given."""
        part_id = new_node_id("part")
        part_number = part.get("part_number", "")
        properties = {
            "id": part_id,
            "part_number": part_number,
            "description": part.get("description", ""),
        }
        if part_number:
            self.neo4j.merge_node(
                NodeType.PART,
                match_properties={"part_number": part_number},
                set_properties=properties,
            )
        else:
            self.neo4j.create_node(NodeType.PART, properties)
        self._record_source(NodeType.PART, part_id, chunk)
        return part_id

    def _create_tool_node(self, tool: dict, chunk: Chunk) -> str:
        """Create tool node, merging on tool name when one is given."""
        tool_id = new_node_id("tool")
        tool_name = tool.get("tool_name", "")
        properties = {
            "id": tool_id,
            "tool_name": tool_name,
            "tool_type": tool.get("tool_type", "common"),
            "requires_certification": tool.get("requires_certification", False),
        }
        if tool_name:
            self.neo4j.merge_node(
                NodeType.TOOL,
                match_properties={"tool_name": tool_name},
                set_properties=properties,
            )
        else:
            self.neo4j.create_node(NodeType.TOOL, properties)
        self._record_source(NodeType.TOOL, tool_id, chunk)
        return tool_id

    def _create_skill_node(self, skill: dict, chunk: Chunk) -> str:
        """Create skill node, merging on skill name when one is given."""
        skill_id = new_node_id("skill")
        skill_name = skill.get("skill_name", "")
        properties = {
            "id": skill_id,
            "skill_name": skill_name,
            "certification_level": skill.get("certification_level", "basic"),
            "certification_body": skill.get("certification_body", ""),
        }
        if skill_name:
            self.neo4j.merge_node(
                NodeType.SKILL,
                match_properties={"skill_name": skill_name},
                set_properties=properties,
            )
        else:
            self.neo4j.create_node(NodeType.SKILL, properties)
        self._record_source(NodeType.SKILL, skill_id, chunk)
        return skill_id


def open_graph_populator(reset: bool = False) -> GraphPopulator:
    """Return a GraphPopulator whose database has the schema; reset deletes every node first."""
    populator = GraphPopulator()
    if reset:
        populator.neo4j.clear_database(confirm=True)
    populator.neo4j.initialize_schema()
    return populator
