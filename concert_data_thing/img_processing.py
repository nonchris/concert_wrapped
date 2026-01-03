import datetime as dt
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import numpy as np
from pydantic import AliasPath
from pydantic import BaseModel
from pydantic.fields import FieldInfo

# get folder of this file
images_path = Path(__file__).resolve().parent / "images"


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


class PriceAble(MarkerDrivenBaseModel):
    prices: list[float]

    marker_total_cost: ClassVar[str] = "Tcx"

    @property
    def total_cost(self) -> float:
        # TODO this fails for multi day festivals
        return self._total_cost(self.prices)

    @staticmethod
    def _total_cost(prices: list[float]):
        return round(sum(prices), 2)

    marker_mean_ticket_cost: ClassVar[str] = "Mcx"

    @property
    def mean_ticket_cost(self):
        return self._mean_ticket_cost(self.prices)

    @staticmethod
    def _mean_ticket_cost(prices: list[float]):
        return round(np.mean(prices), 2) if prices else float("nan")

    marker_most_expensive_ticket: ClassVar[str] = "Ecx"

    @property
    def most_expensive_ticket(self) -> float:
        return self._most_expensive_ticket(self.prices)

    @staticmethod
    def _most_expensive_ticket(prices: list[float]) -> float:
        return round(max(prices), 2) if prices else float("nan")

    marker_most_cheap_ticket: ClassVar[str] = "Ccx"

    @property
    def most_cheap_ticket(self) -> float:
        return self._most_cheap_ticket(self.prices)

    @staticmethod
    def _most_cheap_ticket(prices: list[float]) -> float:
        return round(min(prices), 2) if prices else float("nan")


class TopBandContext(PriceAble):
    TYPE_HEADLINE: ClassVar[int] = 1
    TYPE_SUPPORT: ClassVar[int] = 2
    TYPE_FESTIVAL: ClassVar[int] = 3

    related_svg_unique_top_4: ClassVar[Path] = images_path / "top4bands.svg"

    related_svg_solo_export: ClassVar[Path] = images_path / "one-artist.svg"

    def key(self):
        return self.headline_shows_count, self.total_cost

    def key_processor(self, k: str, is_ranked: bool):
        if is_ranked:
            return self.replace_lq_gt(k.replace("x", str(self.position)))
        return self.replace_lq_gt(k.replace("x", "1"))

    position: int | None
    """ranking of the band"""

    marker_name: ClassVar[str] = "Bx"
    name: str

    marker_seen_sets: ClassVar[str] = "Sx"

    marker_headline_shows_count: ClassVar[str] = "Hsx"
    marker_support_shows_count: ClassVar[str] = "Ssx"
    marker_festival_shows_count: ClassVar[str] = "Fsx"

    classified_sets: list[int]
    """0: support, 1: headline, 2: festival"""

    @property
    def seen_sets(self) -> int:
        return len(self.classified_sets)

    def _count_shows(self, _type: int):
        return len(list(filter(lambda x: x == _type, self.classified_sets)))

    @property
    def support_shows_count(self) -> int:
        return self._count_shows(self.TYPE_SUPPORT)

    @property
    def headline_shows_count(self) -> int:
        return self._count_shows(self.TYPE_HEADLINE)

    @property
    def festival_shows_count(self) -> int:
        return self._count_shows(self.TYPE_FESTIVAL)

    marker_visit_dates_formatted: ClassVar[str] = "Dtx"

    @property
    def visit_dates_formatted(self):
        return ", ".join(dt.strftime("%d.%m.%Y") for dt in self.visit_dates)

    marker_venues_at_dates_formatted: ClassVar[str] = "Vdx"

    @property
    def venues_at_dates_formatted(self):
        return ", ".join(f"{dt.strftime('%d.%m.%Y')}@{venue}" for dt, venue in zip(self.visit_dates, self.venues))

    visit_dates: list[dt.datetime]

    headline_per_night: list[str]

    marker_venues_count: ClassVar[str] = "Vx"

    venues: list[str]

    @property
    def venues_count(self) -> int:
        return len(set(self.venues))

    # marker_cities: ClassVar[str] = "<TBD>"
    marker_cities_count: ClassVar[str] = "Cix"
    cities: list[str]

    @property
    def cities_count(self) -> int:
        return len(set(self.cities))

    marker_countries_count: ClassVar[str] = "Cox"

    countries: list[str]

    @property
    def countries_count(self) -> int:
        return len(set(self.countries))

    marker_total_headline_cost: ClassVar[str] = "Thx"
    marker_total_support_cost: ClassVar[str] = "Tsx"
    marker_total_festival_cost: ClassVar[str] = "Tfx"

    @property
    def total_headline_cost(self) -> float:
        return self._total_cost(self._get_prices_for_type(self.TYPE_HEADLINE))

    @property
    def total_support_cost(self) -> float:
        return self._total_cost(self._get_prices_for_type(self.TYPE_SUPPORT))

    @property
    def total_festival_cost(self) -> float:
        return self._total_cost(self._get_prices_for_type(self.TYPE_FESTIVAL))

    def _get_prices_for_type(self, _type: int):
        price_array = np.array(self.prices)
        type_array = np.array(self.classified_sets)
        return price_array[type_array == _type].tolist()

    def _get_headline_band_for_price_extreme(self, prices: list[float], use_max: bool) -> str:
        """Get headline band name for the most/least expensive ticket."""
        if not prices:
            return "[unknown]"
        prices_np = np.array(prices)
        idx = np.argmax(prices_np) if use_max else np.argmin(prices_np)
        headline_name = self.headline_per_night[idx]
        # If the artist name matches the headline, return "headliner"
        if self.name == headline_name:
            return "Headliner"
        return headline_name

    marker_most_expensive_ticket_headline_band: ClassVar[str] = "Ebx"

    @property
    def most_expensive_ticket_headline_band(self) -> str:
        return self._get_headline_band_for_price_extreme(self.prices, use_max=True)

    marker_cheapest_ticket_headline_band: ClassVar[str] = "Cbx"

    @property
    def cheapest_ticket_headline_band(self) -> str:
        return self._get_headline_band_for_price_extreme(self.prices, use_max=False)


class VenueContext(PriceAble):
    related_svg_unique_top_4 = images_path / "top4venues.svg"
    related_svg_solo_export: ClassVar[Path] = images_path / "one-venue.svg"

    def key(self):
        return self.total_visits, self.total_sets, self.total_cost

    def key_processor(self, k: str, is_ranked: bool):
        if is_ranked:
            return self.replace_lq_gt(k.replace("x", str(self.position)))
        return self.replace_lq_gt(k.replace("x", str(1)))

    marker_name: ClassVar[str] = "Vx"

    position: int | None

    name: str

    marker_total_cost: ClassVar[str] = "Tcx"

    visit_dates: list[dt.datetime]

    marker_total_visits: ClassVar[str] = "Tvx"

    @property
    def total_visits(self):
        return len(self.visit_dates)

    num_bands_per_night: list[int]
    headline_per_night: list[str]

    marker_mean_bands_per_night: ClassVar[str] = "Bnx"

    @property
    def mean_bands_per_night(self):
        return round(np.mean(self.num_bands_per_night), 2)

    marker_total_sets: ClassVar[str] = "Tsx"

    @property
    def total_sets(self):
        return sum(self.num_bands_per_night)

    marker_most_expensive_artist: ClassVar[str] = "Ebx"

    @property
    def most_expensive_artist(self):
        if not self.prices:
            return "[unknown]"

        return self.headline_per_night[np.argmax(self.prices)]

    marker_cheapest_artist: ClassVar[str] = "Cbx"

    @property
    def cheapest_artist(self):
        if not self.prices:
            return "[unknown]"
        return self.headline_per_night[np.argmin(self.prices)]

    marker_artist_at_dates_formatted: ClassVar[str] = "Adx"

    @property
    def artist_at_dates_formatted(self):
        return " - ".join(
            f"{dt.strftime('%d.%m.%Y')}, {artist}" for dt, artist in zip(self.visit_dates, self.headline_per_night)
        )


class VenueSummary(MarkerDrivenBaseModel):
    related_svg_summary: ClassVar[Path] = images_path / "venue-multi-slide.svg"

    marker_times_visited: ClassVar[str] = "N"

    times_visited: int = FieldInfo(validation_alias=AliasPath("times"))

    marker_total_sets: ClassVar[str] = "S"

    venues: list[VenueContext] = FieldInfo(validation_alias=AliasPath("elements"))

    @property
    def total_sets(self):
        return sum(venue.total_bands_seen for venue in self.venues)

    marker_venues_summary: ClassVar[str] = "VENUES"

    @property
    def venue_summary(self):
        return ", ".join(venue.name for venue in self.venues)


class BandSeenSetSummary(MarkerDrivenBaseModel):

    related_svg_summary: ClassVar[Path] = images_path / "many-artists-overview.svg"

    marker_times_seen: ClassVar[str] = "N"

    times_seen: int = FieldInfo(validation_alias=AliasPath("times"))

    bands: list[TopBandContext] = FieldInfo(validation_alias=AliasPath("elements"))

    marker_total_sets: ClassVar[str] = "S"

    @property
    def total_sets(self):
        return self.times_seen * len(self.bands)

    marker_band_summary: ClassVar[str] = "BANDS"

    @property
    def band_summary(self):
        return ", ".join(band.name for band in self.bands)


class MetaInfo(MarkerDrivenBaseModel):
    marker_user_name: ClassVar[str] = "USER"

    user_name: str

    marker_year: ClassVar[str] = "YEAR"
    year: int


class UserAnalysis(MarkerDrivenBaseModel):
    related_svg_solo_export: ClassVar[Path] = images_path / "user-high-level.svg"

    marker_days_with_show: ClassVar[str] = "Ds"
    days_with_show: int

    marker_days_with_show_wo_festival: ClassVar[str] = "Dso"
    days_with_show_wo_festival: int

    marker_days_with_show_festival: ClassVar[str] = "Dsf"

    @property
    def days_with_show_festival(self):
        return self.days_with_show - self.days_with_show_wo_festival

    marker_sets_seen: ClassVar[str] = "St"
    sets_seen: int

    marker_sets_seen_wo_festival: ClassVar[str] = "Sto"
    sets_seen_wo_festival: int

    marker_sets_seen_festival: ClassVar[str] = "Sf"

    @property
    def sets_seen_festival(self):
        return self.sets_seen - self.sets_seen_wo_festival

    marker_total_ticket_cost: ClassVar[str] = "Tc"
    total_ticket_cost: float

    marker_total_ticket_cost_wo_festival: ClassVar[str] = "Toc"
    total_ticket_cost_wo_festival: float

    marker_mean_ticket_cost: ClassVar[str] = "Mc"
    mean_ticket_cost: float

    marker_mean_ticket_cost_wo_festival: ClassVar[str] = "Moc"
    mean_ticket_cost_wo_festival: float

    marker_most_expensive: ClassVar[str] = "E"
    most_expensive: float

    marker_most_expensive_show: ClassVar[str] = "Eb"
    most_expensive_show: str

    marker_most_expensive_wo_festival: ClassVar[str] = "Eo"
    most_expensive_wo_festival: float

    marker_most_expensive_show_wo_festival: ClassVar[str] = "Eob"
    most_expensive_show_wo_festival: str

    marker_price_per_set: ClassVar[str] = "Ps"
    price_per_set: float

    marker_price_per_set_wo_festival: ClassVar[str] = "Pso"
    price_per_set_wo_festival: float

    marker_total_festival_cost: ClassVar[str] = "Tfc"
    total_festival_cost: float

    marker_total_countries: ClassVar[str] = "Cn"
    total_countries: int

    marker_total_cities: ClassVar[str] = "Ci"
    total_cities: int

    marker_total_venues: ClassVar[str] = "Vn"
    total_venues: int

    marker_total_artists: ClassVar[str] = "Ar"
    total_artists: int


if __name__ == "__main__":
    meta_data = MetaInfo(user_name="cyber_chris", year=2025)

    top_band = TopBandContext(
        position=1,
        name="BMTH",
        classified_sets=[True, False, True],
        venues=["Palladium", "RIP"],
        cities=["Köln", "ADW"],
        countries=["Germany", "Germany"],
        prices=[98, 121],
    )

    svg = images_path / "drawing.svg"

    svg_content = svg.read_text()

    svg_content = top_band.apply_self_to_text(svg_content)

    svg_content = meta_data.apply_self_to_text(svg_content)

    svgg = images_path / "auto.svg"

    svgg.write_text(svg_content)
