from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel


class MarkerDrivenBaseModel(BaseModel):

    related_svg_unique_top_4: ClassVar[Path]

    def get_marker_mapping(self) -> dict:
        # Find all class vars that start with 'marker_' in self and parent classes
        mapping = {}
        processed_markers = set()
        for cls in type(self).__mro__:
            for name in vars(cls):
                if name.startswith("marker_") and name not in processed_markers:
                    processed_markers.add(name)
                    non_marker_name = name.replace("marker_", "")
                    marker_value = getattr(cls, name)
                    # Get the non-marker attribute value from the instance (self)
                    non_marker_value = getattr(self, non_marker_name)
                    mapping[marker_value] = non_marker_value
        return mapping

    def apply_self_to_text(self, text: str, is_ranked: bool = True) -> str:

        for _k, v in self.get_marker_mapping().items():
            k = self.key_processor(_k, is_ranked=is_ranked)

            text = text.replace(k, str(v))

        return text

    @staticmethod
    def replace_lq_gt(t: str):
        return f"&lt;{t}&gt;"

    def key_processor(self, k: str, is_ranked: bool):
        """call to replace_lq_gt() by default"""
        return self.replace_lq_gt(k)
