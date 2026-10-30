"""Entity extraction from service manual chunks using GPT."""

import asyncio
import json
import logging

from openai import AsyncOpenAI, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from service_manual_rag.config import get_settings
from service_manual_rag.processing.hierarchical_chunker import Chunk
from service_manual_rag.utils.logging_config import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are an expert at extracting structured information from industrial equipment "
    "service manuals."
)

SECTION_PROMPT_TEMPLATE = """Extract HIGH-LEVEL entities from this service manual section overview:

Text:
{content}

Section: {section}
OEM: {oem}
Model: {model}
Chunk Level: {level} (Section/Subsection)

This is a section overview. Focus on extracting:
1. Major components/systems mentioned
2. General safety warnings for the section
3. Overall section topics

Return JSON with these entities:
{{
  "components": [
    {{"part_name": "hydraulic system", "part_number": "", "description": "main hydraulic system"}}
  ],
  "safety_hazards": [
    {{"hazard_type": "pressure", "warning": "high pressure hydraulic lines", "severity": "danger", "required_ppe": "safety glasses, gloves"}}
  ],
  "parts": [],
  "tools": [],
  "skills": [],
  "procedures": [],
  "symptoms": [],
  "failure_modes": [],
  "solutions": []
}}

EXAMPLES:

Example 1 (Hydraulic System Section):
Input: "Hydraulic System Maintenance. This section covers the hydraulic pump, control valves, and hydraulic cylinders. WARNING: System operates at 3000 PSI."
Output:
{{
  "components": [
    {{"part_name": "hydraulic pump", "part_number": "", "description": "main hydraulic pump"}},
    {{"part_name": "control valves", "part_number": "", "description": "hydraulic control valves"}},
    {{"part_name": "hydraulic cylinders", "part_number": "", "description": "hydraulic cylinders"}}
  ],
  "safety_hazards": [
    {{"hazard_type": "pressure", "warning": "System operates at 3000 PSI", "severity": "danger", "required_ppe": ""}}
  ],
  "parts": [], "tools": [], "skills": [], "procedures": [], "symptoms": [], "failure_modes": [], "solutions": []
}}

Example 2 (Motor Section):
Input: "Motor Specifications. Drive motor model M-450. Rated power: 75 kW. Supply: 480 V, 3-phase."
Output:
{{
  "components": [
    {{"part_name": "drive motor", "part_number": "M-450", "description": "75 kW, 480 V 3-phase drive motor"}}
  ],
  "safety_hazards": [],
  "parts": [], "tools": [], "skills": [], "procedures": [], "symptoms": [], "failure_modes": [], "solutions": []
}}

Guidelines:
- Extract ONLY major components/systems (not individual parts)
- Extract general section-level safety warnings
- Keep descriptions high-level
- Leave detailed arrays empty (procedures, symptoms, failure_modes, solutions)
- If nothing relevant, return empty arrays

Now extract from the provided text above."""

SEMANTIC_PROMPT_TEMPLATE = """Extract DETAILED entities from this service manual text:

Text:
{content}

Section: {section}
OEM: {oem}
Model: {model}
Chunk Level: {level} (Semantic/Detailed)

Extract and return as JSON with ALL applicable entities:
{{
  "components": [
    {{"part_name": "...", "part_number": "...", "description": "..."}}
  ],
  "procedures": [
    {{
      "action_type": "install|replace|adjust|remove|inspect|test|calibrate",
      "steps": "...",
      "difficulty": "easy|medium|hard",
      "estimated_time": "...",
      "required_parts": ["part_number1"],
      "required_tools": ["tool_name1"],
      "required_skills": ["skill_name1"]
    }}
  ],
  "symptoms": [
    {{
      "description": "...",
      "severity": "low|medium|high|critical",
      "failure_modes": ["failure_mode_name"]
    }}
  ],
  "failure_modes": [
    {{
      "name": "...",
      "description": "...",
      "category": "mechanical|electrical|hydraulic|pneumatic|software|structural",
      "severity": "low|medium|high|critical",
      "safety_hazards": ["hazard_description"]
    }}
  ],
  "safety_hazards": [
    {{
      "hazard_type": "electrical|pressure|chemical|thermal|mechanical|fall|crush",
      "warning": "...",
      "severity": "warning|danger|critical",
      "required_ppe": "..."
    }}
  ],
  "parts": [
    {{
      "part_number": "123-4567",
      "description": "...",
      "specifications": "..."
    }}
  ],
  "tools": [
    {{
      "tool_name": "torque wrench",
      "tool_type": "specialized|common|calibrated|diagnostic",
      "requires_certification": true|false
    }}
  ],
  "skills": [
    {{
      "skill_name": "...",
      "certification_level": "basic|intermediate|expert|master",
      "certification_body": "..."
    }}
  ],
  "solutions": [
    {{"fix_description": "...", "parts_needed": ["part1"]}}
  ]
}}

EXAMPLES:

Example 1 (Filter Replacement Procedure):
Input: "To replace the hydraulic filter (Part# 123-4567): 1) Stop the unit and lock out power. 2) Relieve system pressure. 3) Use 1-1/4 inch wrench to remove filter. 4) Install new filter. Torque to 25 ft-lbs. Time: 30 minutes. WARNING: Hot oil under pressure."
Output:
{{
  "components": [{{"part_name": "hydraulic filter", "part_number": "123-4567", "description": "hydraulic oil filter"}}],
  "procedures": [{{
    "action_type": "replace",
    "steps": "1) Stop the unit and lock out power 2) Relieve system pressure 3) Remove old filter with wrench 4) Install new filter, torque to 25 ft-lbs",
    "difficulty": "easy",
    "estimated_time": "30 minutes",
    "required_parts": ["123-4567"],
    "required_tools": ["1-1/4 inch wrench", "torque wrench"],
    "required_skills": []
  }}],
  "parts": [{{"part_number": "123-4567", "description": "hydraulic oil filter", "specifications": "torque to 25 ft-lbs"}}],
  "tools": [
    {{"tool_name": "1-1/4 inch wrench", "tool_type": "common", "requires_certification": false}},
    {{"tool_name": "torque wrench", "tool_type": "calibrated", "requires_certification": false}}
  ],
  "safety_hazards": [{{"hazard_type": "thermal", "warning": "Hot oil under pressure", "severity": "danger", "required_ppe": "gloves, safety glasses"}}],
  "symptoms": [], "failure_modes": [], "skills": [], "solutions": []
}}

Example 2 (Troubleshooting):
Input: "Symptom: Pump loses discharge pressure under load. Possible causes: 1) Clogged suction strainer 2) Worn impeller 3) Air in suction line. Solution: Check suction strainer first (Part# 456-7890), replace if blocked."
Output:
{{
  "symptoms": [{{"description": "Pump loses discharge pressure under load", "severity": "medium", "failure_modes": ["clogged suction strainer", "worn impeller", "air in suction line"]}}],
  "failure_modes": [
    {{"name": "clogged suction strainer", "description": "suction strainer blocked", "category": "mechanical", "severity": "medium", "safety_hazards": []}},
    {{"name": "worn impeller", "description": "impeller vanes eroded or damaged", "category": "mechanical", "severity": "high", "safety_hazards": []}},
    {{"name": "air in suction line", "description": "air infiltration in suction piping", "category": "mechanical", "severity": "medium", "safety_hazards": []}}
  ],
  "solutions": [{{"fix_description": "Check and replace suction strainer if blocked", "parts_needed": ["456-7890"]}}],
  "parts": [{{"part_number": "456-7890", "description": "suction strainer", "specifications": ""}}],
  "components": [], "procedures": [], "safety_hazards": [], "tools": [], "skills": []
}}

Example 3 (Electrical Test):
Input: "To test motor winding insulation: Requires qualified electrician certification. Use an insulation resistance tester. Disconnect the motor leads and apply 500 V DC between each winding and ground. Reading should exceed 100 megohms. DANGER: Lock out and tag out the supply before testing."
Output:
{{
  "procedures": [{{
    "action_type": "test",
    "steps": "Disconnect the motor leads. Apply 500 V DC between each winding and ground. Reading should exceed 100 megohms",
    "difficulty": "medium",
    "estimated_time": "",
    "required_parts": [],
    "required_tools": ["insulation resistance tester"],
    "required_skills": ["qualified electrician certification"]
  }}],
  "tools": [{{"tool_name": "insulation resistance tester", "tool_type": "diagnostic", "requires_certification": false}}],
  "skills": [{{"skill_name": "qualified electrician certification", "certification_level": "intermediate", "certification_body": ""}}],
  "safety_hazards": [{{"hazard_type": "electrical", "warning": "Lock out and tag out the supply before testing", "severity": "danger", "required_ppe": ""}}],
  "components": [], "parts": [], "symptoms": [], "failure_modes": [], "solutions": []
}}

Guidelines:
- BE THOROUGH: Extract ALL mentioned entities
- FAILURE MODES: Root causes (e.g., "bearing wear", "sensor malfunction", "clogged filter")
- SYMPTOMS: Observable issues (e.g., "motor will not start", "excessive vibration")
- LINK ENTITIES: Connect symptoms to failure_modes, failure_modes to safety_hazards, and procedures to parts, tools and skills
- PART NUMBERS: Always extract when mentioned
- TOOLS: Be specific (e.g., "1-1/4 inch wrench", "insulation resistance tester")
- SPECIFICATIONS: Include torque values, voltages, pressures in part specifications
- SAFETY: Extract ALL warnings, cautions, dangers
- EMPTY ARRAYS: Return [] for categories with no entities

Now extract from the provided text above."""


class EntityExtractor:
    """Extract entities from chunks using GPT with async batch processing."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int = 10,
        max_concurrent: int = 2,
    ):
        """Initialize entity extractor.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY; raises
                MissingCredentialError when neither is set)
            model: Model to use for extraction
            batch_size: Number of chunks to process per batch
            max_concurrent: Maximum concurrent API calls (default 2 to stay under rate limits)
        """
        settings = get_settings()
        resolved_api_key = settings.require_openai_key(api_key)
        self.model = model or settings.openai_model
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent

        self.client = AsyncOpenAI(api_key=resolved_api_key)
        self.semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(f"EntityExtractor initialized with model: {self.model}")
        logger.info(f"Batch processing: batch_size={batch_size}, max_concurrent={max_concurrent}")

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _call_gpt_with_retry(self, prompt: str, chunk_id: str) -> dict:
        """Call GPT with retry logic for rate limits.

        Args:
            prompt: Extraction prompt
            chunk_id: Chunk ID for logging

        Returns:
            dict: Extracted entities
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        entities = json.loads(response.choices[0].message.content)
        logger.debug(f"Extracted entities from chunk {chunk_id}")
        return entities

    async def extract_entities(self, chunk: Chunk) -> dict:
        """Extract entities from a chunk (async).

        Args:
            chunk: Chunk to extract from

        Returns:
            dict: Extracted entities

        Example:
            >>> extractor = EntityExtractor()
            >>> entities = await extractor.extract_entities(chunk)
            >>> print(entities['components'])
        """
        prompt = self._build_extraction_prompt(chunk)
        async with self.semaphore:
            return await self._call_gpt_with_retry(prompt, chunk.chunk_id)

    async def extract_entities_batch(self, chunks: list[Chunk]) -> list[dict]:
        """Extract entities from multiple chunks in parallel.

        Args:
            chunks: List of chunks to process

        Returns:
            list[dict]: Extracted entities for each chunk, in the same order as ``chunks``
        """
        total = len(chunks)
        logger.info(f"Processing {total} chunks in parallel (max_concurrent={self.max_concurrent})")

        tasks = [asyncio.ensure_future(self.extract_entities(chunk)) for chunk in chunks]
        try:
            for completed, finished in enumerate(asyncio.as_completed(tasks), start=1):
                await finished
                if completed % 50 == 0 or completed == total:
                    logger.info(
                        f"Entity extraction progress: {completed}/{total} "
                        f"({completed * 100 // total}%)"
                    )
        except BaseException:
            for task in tasks:
                task.cancel()
            raise

        return [task.result() for task in tasks]

    def _build_extraction_prompt(self, chunk: Chunk) -> str:
        """Build extraction prompt based on chunk level.

        Args:
            chunk: Chunk to extract from

        Returns:
            str: Extraction prompt
        """
        if chunk.level in [2, 3]:
            return self._build_section_prompt(chunk)
        return self._build_semantic_prompt(chunk)

    def _build_section_prompt(self, chunk: Chunk) -> str:
        """Build prompt for section-level chunks (L2/L3).

        Focus on high-level entities: major components, section overview.

        Args:
            chunk: Section/subsection chunk

        Returns:
            str: Extraction prompt
        """
        return SECTION_PROMPT_TEMPLATE.format(**self._prompt_fields(chunk))

    def _build_semantic_prompt(self, chunk: Chunk) -> str:
        """Build prompt for semantic chunks (L4).

        Focus on detailed entities: procedures, parts, tools, symptoms, etc.

        Args:
            chunk: Semantic chunk

        Returns:
            str: Extraction prompt
        """
        return SEMANTIC_PROMPT_TEMPLATE.format(**self._prompt_fields(chunk))

    @staticmethod
    def _prompt_fields(chunk: Chunk) -> dict[str, object]:
        """Values substituted into the prompt templates."""
        return {
            "content": chunk.content,
            "section": chunk.section_name or "Unknown",
            "oem": chunk.oem,
            "model": chunk.model,
            "level": chunk.level,
        }
