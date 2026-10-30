"""Image and video analysis for equipment diagnostics through Google Gemini.

The Gemini client is created on the first analysis call, so the service can be
constructed and can store uploaded media without GOOGLE_API_KEY. ``google.generativeai``
and ``PIL`` are imported only at that point; both come with the ``agents`` extra.
Uploaded media is stored on the local disk under ``data/agent_media``.
"""

import base64
import io
import json
import re
from pathlib import Path
from typing import Any

from service_manual_rag.config import MissingCredentialError, Settings, get_settings
from service_manual_rag.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MEDIA_DIR = Path("data/agent_media")


class VLMService:
    """Gemini vision service for defect detection, OCR of labels and error codes,
    component identification and general visual inspection."""

    def __init__(
        self,
        api_key: str | None = None,
        media_dir: Path | None = None,
        settings: Settings | None = None,
    ):
        """
        Args:
            api_key: Google AI API key; defaults to ``settings.google_api_key``.
            media_dir: Directory for uploaded media; defaults to ``data/agent_media``.
            settings: Application settings; defaults to ``get_settings()``.
        """
        settings = settings or get_settings()
        self.api_key = api_key or settings.google_api_key
        self.model_name = settings.gemini_model
        self.media_dir = Path(media_dir) if media_dir else DEFAULT_MEDIA_DIR
        self._genai: Any = None
        self._model: Any = None

    def _get_model(self) -> Any:
        """Configure Gemini on first use and return the generative model."""
        if self._model is None:
            if not self.api_key:
                raise MissingCredentialError(
                    "GOOGLE_API_KEY is not set. Image and video analysis needs a Google AI "
                    "API key; set it in the environment or in .env."
                )
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._genai = genai
            self._model = genai.GenerativeModel(self.model_name)
            logger.info("VLM service connected to %s", self.model_name)
        return self._model

    def analyze_image(
        self,
        image_path: str,
        analysis_type: str = "general",
        equipment_context: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Analyze an image with Gemini.

        Args:
            image_path: Path to an image file, or a base64 encoded image
            analysis_type: 'defect', 'ocr', 'component' or 'general'
            equipment_context: Optional context with 'oem' and 'model'

        Returns:
            Dictionary with findings, labels, extracted_text and confidence

        Raises:
            MissingCredentialError: GOOGLE_API_KEY is not set.
        """
        model = self._get_model()
        from PIL import Image

        logger.info("Analyzing image: %s (type: %s)", image_path, analysis_type)

        if Path(image_path).exists():
            image = Image.open(image_path)
        else:
            image = Image.open(io.BytesIO(base64.b64decode(image_path)))

        prompt = self._build_analysis_prompt(analysis_type, equipment_context)
        response = model.generate_content([prompt, image])
        result = self._parse_analysis_response(response.text)
        logger.info("Image analysis complete: %d findings", len(result["findings"]))
        return result

    def analyze_video(
        self,
        video_path: str,
        analysis_type: str = "general",
        equipment_context: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Upload a video to Gemini and analyze it.

        Args:
            video_path: Path to the video file
            analysis_type: Type of analysis; videos use one general prompt
            equipment_context: Optional context with 'oem' and 'model'

        Returns:
            Dictionary with findings, labels, extracted_text and confidence

        Raises:
            MissingCredentialError: GOOGLE_API_KEY is not set.
        """
        model = self._get_model()
        logger.info("Analyzing video: %s (type: %s)", video_path, analysis_type)

        video_file = self._genai.upload_file(path=video_path)
        try:
            prompt = self._build_video_analysis_prompt(equipment_context)
            response = model.generate_content([prompt, video_file])
        finally:
            self._genai.delete_file(video_file.name)
        result = self._parse_analysis_response(response.text)
        logger.info("Video analysis complete: %d findings", len(result["findings"]))
        return result

    def save_media(self, file_data: bytes, filename: str) -> str:
        """Save uploaded media to the local media directory and return its path."""
        self.media_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.media_dir / Path(filename).name
        file_path.write_bytes(file_data)
        logger.info("Media saved to: %s", file_path)
        return str(file_path)

    @staticmethod
    def _equipment_phrase(equipment_context: dict[str, str] | None) -> str:
        if not equipment_context:
            return ""
        equipment = f"{equipment_context.get('oem') or ''} {equipment_context.get('model') or ''}"
        return equipment.strip()

    def _build_analysis_prompt(
        self, analysis_type: str, equipment_context: dict[str, str] | None
    ) -> str:
        """Build the image analysis prompt for the requested analysis type."""
        equipment = self._equipment_phrase(equipment_context)
        base_context = f"This image shows {equipment} equipment. " if equipment else ""

        prompts = {
            "defect": (
                f"{base_context}Analyze this image for equipment defects. "
                "Look for: cracks, leaks, corrosion, wear, damage, smoke, unusual discoloration. "
                "For each finding, describe: type, location, severity (low/medium/high). "
                "Format as JSON with 'findings' array containing "
                "{type, location, severity, description}."
            ),
            "ocr": (
                f"{base_context}Extract all visible text from this image. "
                "Focus on: error codes, part numbers, labels, warnings, gauges and readings. "
                "Format as JSON with 'extracted_text' array and 'error_codes' array."
            ),
            "component": (
                f"{base_context}Identify equipment components visible in this image. "
                "List each component with: name, condition, visible part numbers. "
                "Format as JSON with 'components' array containing {name, condition, part_numbers}."
            ),
            "general": (
                f"{base_context}Describe what you see in this equipment image. "
                "Note: overall condition, any issues, relevant details for troubleshooting. "
                "Format as JSON with 'description', 'condition', 'issues' array."
            ),
        }
        return prompts.get(analysis_type, prompts["general"])

    def _build_video_analysis_prompt(self, equipment_context: dict[str, str] | None) -> str:
        """Build the video analysis prompt."""
        equipment = self._equipment_phrase(equipment_context)
        base_context = f"This video shows {equipment} equipment. " if equipment else ""
        return (
            f"{base_context}Analyze this equipment video. "
            "Look for: unusual movements, vibrations, leaks, smoke, abnormal operation. "
            "Describe: what is happening, timeline of events, severity of any issues. "
            "Format as JSON with 'observations' array containing "
            "{timestamp, description, severity}."
        )

    @staticmethod
    def _parse_analysis_response(response_text: str) -> dict[str, Any]:
        """Parse a Gemini reply that is either a fenced JSON block or plain text."""
        result: dict[str, Any] = {
            "findings": [],
            "labels": [],
            "extracted_text": "",
            "confidence": 0.8,
        }

        json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
            except json.JSONDecodeError as error:
                logger.warning("Gemini returned invalid JSON (%s); treating it as text", error)
            else:
                for key in ("findings", "components", "observations"):
                    if key in parsed:
                        result["findings"] = parsed[key]
                if "extracted_text" in parsed:
                    result["extracted_text"] = parsed["extracted_text"]
                if "error_codes" in parsed:
                    result["labels"] = parsed["error_codes"]
                return result

        result["findings"] = [
            {"type": "general_observation", "description": response_text, "confidence": 0.7}
        ]
        return result


_vlm_service: VLMService | None = None


def get_vlm_service() -> VLMService:
    """Return the process-wide VLM service, creating it on first use."""
    global _vlm_service
    if _vlm_service is None:
        _vlm_service = VLMService()
    return _vlm_service
