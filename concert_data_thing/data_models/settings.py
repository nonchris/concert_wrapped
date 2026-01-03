from typing import ClassVar

from concert_data_thing.data_models.marker_driven_base_model import (
    MarkerDrivenBaseModel,
)


class SVGStyleGuide(MarkerDrivenBaseModel):
    """Colors to apply to the SVGs"""

    gradient_high: str = "#000000"
    gradient_low: str = "#0000ff"
    text_color: str = "#00ff00"

    # these are the default colors in the SVG templates
    marker_gradient_high: ClassVar[str] = "#000000"
    marker_gradient_low: ClassVar[str] = "#0000ff"
    marker_text_color: ClassVar[str] = "#00ff00"

    def key_processor(self, k: str, is_ranked: bool):
        return k
