import datetime
import io
import os
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Callable
from typing import Dict

import dayplot as dp
import numpy as np
import pandas as pd
from fastapi import HTTPException
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pandas import DataFrame

from concert_data_thing.constants import ARTIST
from concert_data_thing.constants import CITY
from concert_data_thing.constants import COUNTRY
from concert_data_thing.constants import DATE
from concert_data_thing.constants import EVENT_NAME
from concert_data_thing.constants import MERCH_COST
from concert_data_thing.constants import ORIGINAL_PRICE
from concert_data_thing.constants import PAID_PRICE
from concert_data_thing.constants import QUALIFIED_NAME
from concert_data_thing.constants import TYPE
from concert_data_thing.constants import VENUE
from concert_data_thing.data_models.settings import SVGStyleGuide
from concert_data_thing.helpers.deduplication import split_day_into_batches
from concert_data_thing.img_processing import BandSeenSetSummary
from concert_data_thing.img_processing import MarkerDrivenBaseModel
from concert_data_thing.img_processing import MetaInfo
from concert_data_thing.img_processing import TopBandContext
from concert_data_thing.img_processing import UserAnalysis
from concert_data_thing.img_processing import VenueContext
from concert_data_thing.img_processing import VenueSummary
from concert_data_thing.logger import LOGGING_PROVIDER

logger = LOGGING_PROVIDER.new_logger("concert_data_thing.main")

# Global constants for DataFrame column names


def parse_concert_csv(
    csv_content: str,
    date: str = "Date",
    date_format: str = "%d.%m.%y",
    sep: str = ",",
    artist: str = "Artist",
    venue: str = "Venue",
    city: str = "City",
    country: str = "Country",
    paid_price: str = "Bezahlt Preis",
    original_price: str = "Preis",
    merch_cost: str = "Merch Ausgaben",
    type: str = "Typ",
    event_name: str = "Event Name",
) -> pd.DataFrame:
    """
    Parse the concert CSV file and extract relevant columns with proper data types.

    Args:
        csv_content: Content of the CSV file as a string
        date: Column name for date (default: "Date")
        date_format: Date format string for parsing (default: "%d.%m.%y")
        sep: CSV separator character (default: ",")
        artist: Column name for artist (default: "Artist")
        venue: Column name for venue (default: "Venue")
        city: Column name for city (default: "City")
        country: Column name for country (default: "Country")
        paid_price: Column name for paid price (default: "Bezahlt Preis")
        original_price: Column name for original price (default: "Preis")
        merch_cost: Column name for merch cost (default: "Merch Ausgaben")
        type: Column name for type (default: "Typ")

    Returns:
        DataFrame with parsed dates and selected columns
    """
    logger.debug(
        f"Parsing CSV with column names: date={date}, date_format={date_format}, sep={sep}, artist={artist}, venue={venue}"
    )
    df = pd.read_csv(io.StringIO(csv_content), sep=sep)

    # Select only the columns we care about
    columns_to_select = [date, artist, venue, city, country, paid_price, original_price, merch_cost, type, event_name]
    try:
        df = df[columns_to_select].copy()
    except KeyError as e:
        raise HTTPException(
            400,
            f"Seems like some of your provided column names are wrong. Are you sure you included the header line in the text? - Specific error: {e}",
        )
    logger.debug(f"Selected {len(columns_to_select)} columns, initial row count: {len(df)}")

    # Convert date column to datetime with specified format
    df[date] = pd.to_datetime(df[date], format=date_format, errors="coerce")
    invalid_dates = df[date].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"Found {invalid_dates} rows with invalid or unparseable dates")

    # Convert numeric columns to float (handles empty strings as NaN)
    df[paid_price] = pd.to_numeric(df[paid_price], errors="coerce").fillna(0.0)
    df[original_price] = pd.to_numeric(df[original_price], errors="coerce").fillna(0.0)
    df[merch_cost] = pd.to_numeric(df[merch_cost], errors="coerce").fillna(0.0)

    # Rename columns to internal names for consistency
    df = df.rename(
        columns={
            date: DATE,
            artist: ARTIST,
            venue: VENUE,
            city: CITY,
            paid_price: PAID_PRICE,
            original_price: ORIGINAL_PRICE,
            merch_cost: MERCH_COST,
            type: TYPE,
            country: COUNTRY,
            event_name: EVENT_NAME,
        }
    )

    logger.info(f"Successfully parsed CSV: {len(df)} rows, date range: {df[DATE].min()} to {df[DATE].max()}")
    return df


def group_by_day(
    df: pd.DataFrame,
    running_order_headline_last: bool = True,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Group the DataFrame by day, preserving the order within each day.

    For concert data, bands are listed in running order. If headline_last is True,
    the order in the DataFrame is correct (headline act is last). If False, the
    order is reversed (headline act is first, so reverse to get running order).

    Args:
        df: DataFrame with a date column (from parse_concert_csv)
        running_order_headline_last: If True, order is correct (headline last).
                                     If False, reverse the order (headline first).

    Returns:
        Dictionary mapping dates to DataFrames for each day
    """
    # Normalize dates to just the date part (remove time component)
    df = df.copy()
    df["date_only"] = pd.to_datetime(df[DATE]).dt.date

    # Add original index to preserve order
    df["_original_index"] = range(len(df))

    grouped = {}

    for date, group_df in df.groupby("date_only", sort=False):
        # Sort by original index to preserve order within the group
        group_df = group_df.sort_values("_original_index").copy()

        # If headline is first (not last), reverse the order to get running order
        if not running_order_headline_last:
            group_df = group_df.iloc[::-1].reset_index(drop=True)
        else:
            group_df = group_df.reset_index(drop=True)

        # Convert date back to Timestamp for consistency
        date_timestamp = pd.Timestamp(date)
        grouped[date_timestamp] = group_df.drop(columns=["date_only", "_original_index"])

    return grouped


def count_column_occurrences(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Count how often each entry in a column occurs.

    Args:
        df: DataFrame to query
        column: Name of the column to count occurrences for

    Returns:
        Series with value counts, sorted in descending order
    """
    if column not in df.columns:
        logger.error(f"Column '{column}' not found in DataFrame. Available columns: {list(df.columns)}")
        raise ValueError(f"Column '{column}' not found in DataFrame. Available columns: {list(df.columns)}")

    counts = df[column].value_counts()
    logger.debug(f"Counted occurrences for column '{column}': {len(counts)} unique values")
    return counts


def select_rows_by_year(df: pd.DataFrame, year: int, date_column: str = DATE) -> pd.DataFrame:
    """
    Select only rows of the given year based on the date column.

    Args:
        df: DataFrame with a date column.
        year: The year to filter by (e.g., 2023).
        date_column: Name of the date column (default DATE).

    Returns:
        DataFrame with only rows where the date_column is in the given year.
    """
    initial_count = len(df)
    date_series = pd.to_datetime(df[DATE], errors="coerce")
    mask = date_series.dt.year == year
    filtered_df = df[mask].reset_index(drop=True)
    logger.info(f"Filtered by year {year}: {initial_count} rows -> {len(filtered_df)} rows")
    return filtered_df


def analyze_concert_csv_file(csv_path: Path, *args, **kwargs):
    """
    See analyze_concert_csv for more details.
    """
    return analyze_concert_csv(csv_path.read_text(), *args, **kwargs)


def analyze_concert_csv(
    csv_str: str,
    filter_year: int,
    user_name: str,
    city: str,
    date: str = "Date",
    date_format: str = "%d.%m.%y",
    sep: str = ",",
    artist: str = "Artist",
    venue: str = "Venue",
    city_column: str = "City",
    country: str = "Country",
    paid_price: str = "Bezahlt Preis",
    original_price: str = "Preis",
    merch_cost: str = "Merch Ausgaben",
    type: str = "Typ",
    headline_label: str = "auto",
    support_label: str = "auto",
    festival_label: str = "F",
    event_name: str = "Event Name",
    running_order_headline_last: bool = True,
    color_scheme: SVGStyleGuide = SVGStyleGuide(),
    request_id: uuid.UUID = "00000000-0000-0000-0000-000000000000",
):
    """
    Perform concert CSV analysis, printing artist value counts, unique artists, and grouped day stats.

    Args:
        csv_str (str): CSV content as a string.
        filter_year (int): Year to filter by.
        user_name (str): User name for the analysis.
        city (str): City name for the analysis.
        date (str): Column name for date (default: "Date").
        date_format (str): Date format string for parsing (default: "%d.%m.%y").
        sep (str): CSV separator character (default: ",").
        artist (str): Column name for artist (default: "Artist").
        venue (str): Column name for venue (default: "Venue").
        city_column (str): Column name for city (default: "City").
        country (str): Column name for country (default: "Country").
        paid_price (str): Column name for paid price (default: "Bezahlt Preis").
        original_price (str): Column name for original price (default: "Preis").
        merch_cost (str): Column name for merch cost (default: "Merch Ausgaben").
        type (str): Column name for type (default: "Typ").
        headline_label (str): Label for headline acts in Type column, or "auto" to detect by running order (default: "auto").
        support_label (str): Label for support acts in Type column, or "auto" to detect by running order (default: "auto").
        festival_label (str): Label for festival acts in Type column (default: "F").
        event_name (str): Column name for event name (default: "Event Name").
        running_order_headline_last (bool): If True, order is correct (headline last).
                                           If False, reverse the order (headline first).

        request_id (uuid.UUID): Request ID for the analysis.
    """

    logger.info(
        f"Starting analysis for user={user_name}, year={filter_year}, city={city}, running_order_headline_last={running_order_headline_last}, request_id={request_id}"
    )
    df = parse_concert_csv(
        csv_str,
        date=date,
        date_format=date_format,
        sep=sep,
        artist=artist,
        venue=venue,
        city=city_column,
        country=country,
        paid_price=paid_price,
        original_price=original_price,
        merch_cost=merch_cost,
        type=type,
        event_name=event_name,
    )

    # Add QUALIFIED_NAME: EVENT_NAME if present (takes priority), otherwise headliner artist (fallback)
    # First, determine headliner for each batch and initialize QUALIFIED_NAME with headliner
    logger.debug("Determining headliners for each batch")
    headliner_by_id = {}
    batch_id_list = []
    i = 0

    for date_val, day_df in df.groupby(DATE):
        # Split day into batches if ARTIST values are not unique
        batches = split_day_into_batches(day_df, headline_label)

        for batch_df in batches:
            target_index = -1 if running_order_headline_last else 0
            headliner = determine_headliner_for_day(batch_df, festival_label, headline_label, target_index)
            headliner_by_id[i] = headliner

            # Assign batch_id to each row in this batch
            batch_df["batch_id"] = i
            batch_id_list.append(batch_df)
            i += 1

    if len(headliner_by_id) == 0:
        raise HTTPException(
            400,
            f"Parsing of data failed (no valid row found). "
            f"This is likely due an invalid date format string. "
            f"Other issues could be: Invalid data in the Artist, Type or Venue columns.",
        )

    # Reconstruct DataFrame with batch_id assigned
    df = pd.concat(batch_id_list, ignore_index=True)

    # Initialize QUALIFIED_NAME with headliner for all rows
    df[QUALIFIED_NAME] = df["batch_id"].map(headliner_by_id)

    # Overwrite with EVENT_NAME where EVENT_NAME is not empty
    event_name_mask = df[EVENT_NAME].notna()
    event_name_count = event_name_mask.sum()
    if event_name_count > 0:
        logger.debug(f"Found {event_name_count} rows with event names, using them for QUALIFIED_NAME")
    df.loc[event_name_mask, QUALIFIED_NAME] = df.loc[event_name_mask, EVENT_NAME]

    df = select_rows_by_year(df, filter_year)

    # Create a MultiIndex DataFrame with batch_id as first index and a second index (0..N) per batch
    df_indexed = df.copy()

    # Group by batch_id and assign a second index from 0..N for each batch
    df_indexed["per_day_idx"] = df_indexed.groupby("batch_id").cumcount()
    df_indexed = df_indexed.set_index(["batch_id", "per_day_idx"])
    unique_batches = len(df_indexed.index.get_level_values(0).unique())
    logger.info(f"Created indexed DataFrame with {unique_batches} unique batches, {len(df_indexed)} total entries")

    meta_info = MetaInfo(user_name=user_name, year=filter_year)

    # TODO: later use coockies to store the request_id (basically reclide uuids)
    ARTIFACTS_PATH = os.environ.get("ARTIFACTS_PATH", "out")
    user_data_folder = Path(f"{ARTIFACTS_PATH}/user_data_{request_id}")
    user_data_folder.mkdir(parents=True, exist_ok=True)

    logger.info("Generating artist SVGs")
    artist_svgs = create_svgs_for(
        df_indexed,
        meta_info,
        ARTIST,
        running_order_headline_last,
        user_data_folder,
        headline_label=headline_label,
        support_label=support_label,
        festival_label=festival_label,
        color_scheme=color_scheme,
    )
    logger.info(f"Generated {len(artist_svgs)} artist SVG files")

    logger.info("Generating venue SVGs")
    venue_svgs = create_svgs_for(
        df_indexed, meta_info, VENUE, running_order_headline_last, user_data_folder, color_scheme=color_scheme
    )
    logger.info(f"Generated {len(venue_svgs)} venue SVG files")

    logger.info("Generating city SVGs")
    city_svgs = create_svgs_for(
        df_indexed, meta_info, CITY, running_order_headline_last, user_data_folder, color_scheme=color_scheme
    )
    logger.info(f"Generated {len(city_svgs)} city SVG files")

    logger.info("Performing high-level user analysis")
    user_svg_paths = high_level_user_analysis(
        df_indexed,
        meta_info,
        running_order_headline_last,
        user_data_folder,
        festival_label=festival_label,
        color_scheme=color_scheme,
    )
    logger.info("Analysis complete")

    return {
        "request_id": str(request_id),
        "user_svgs": user_svg_paths,
        "artist_svgs": artist_svgs,
        "venue_svgs": venue_svgs,
        "city_svgs": city_svgs,
    }


def find_most_expensive_ticket(df: DataFrame) -> tuple[float, str]:
    """
    Find the most expensive ticket from a daily summary DataFrame.

    Returns:
        Tuple of (price, show_description)
    """
    if len(df) > 0 and df["price"].max() > 0:
        most_expensive_idx = df["price"].idxmax()
        most_expensive_row = df.loc[most_expensive_idx]
        most_expensive = most_expensive_row["price"]
        most_expensive_show = f"{most_expensive_row[QUALIFIED_NAME]}"
        return most_expensive, most_expensive_show
    else:
        return 0.0, "[unknown]"


def collect_data_for_user_analysis(
    df_indexed: DataFrame, running_order_headline_last: bool, festival_label: str = "F"
) -> UserAnalysis:
    """
    Collect user-level analysis data from the DataFrame.

    Handles multi-day festivals by only counting ticket cost once for sequential
    days where TYPE equals festival_label and VENUE is the same.
    """
    logger.debug(f"Collecting user-level analysis data with festival_label={festival_label}")
    # Reset index to work with dates as a column
    df_reset = df_indexed.reset_index()

    # Count unique days with shows
    unique_dates = df_reset[DATE].unique()
    days_with_show = len(unique_dates)

    # Count total sets seen (total number of entries)
    sets_seen = len(df_indexed)

    # Aggregate by date: one row per day (one venue per day)
    df_daily = (
        df_reset.groupby(DATE)
        .agg(
            {
                # max bc some ppl only enter the price on one artist per night (the headline one)
                PAID_PRICE: "max",
                TYPE: lambda x: (x == festival_label).any(),
                VENUE: "first",
                ARTIST: "last" if running_order_headline_last else "first",
                QUALIFIED_NAME: "first",
            }
        )
        .reset_index()
    )
    df_daily.columns = [DATE, "price", "is_festival", VENUE, ARTIST, QUALIFIED_NAME]
    df_daily = df_daily.sort_values(DATE)

    # Count days without festivals
    days_with_show_wo_festival = len(df_daily[~df_daily["is_festival"]])

    # Determine include_in_sum: False for consecutive festival days (same venue)
    last_festival_date_by_venue = {}
    df_daily["include_in_sum"] = True

    for idx, row in df_daily.iterrows():
        date = row[DATE]
        qualified_name = row[QUALIFIED_NAME]
        is_festival = row["is_festival"]

        if is_festival and qualified_name in last_festival_date_by_venue:
            last_festival_date = last_festival_date_by_venue[qualified_name]
            if (date - last_festival_date).days >= 1:
                df_daily.at[idx, "include_in_sum"] = False

        if is_festival:
            last_festival_date_by_venue[qualified_name] = date
        elif qualified_name in last_festival_date_by_venue:
            del last_festival_date_by_venue[qualified_name]

    total_ticket_cost = df_daily[df_daily["include_in_sum"]]["price"].sum()

    # Calculate total ticket cost without festivals
    df_no_festival = df_daily[~df_daily["is_festival"]]
    total_ticket_cost_wo_festival = df_no_festival["price"].sum()

    # Find most expensive ticket
    most_expensive, most_expensive_show = find_most_expensive_ticket(df_daily)

    # Find most expensive ticket without festivals
    most_expensive_wo_festival, most_expensive_show_wo_festival = find_most_expensive_ticket(df_no_festival)

    # Calculate mean ticket costs
    # Mean with festivals: total cost (festivals counted once) / total days (all festival days included)
    mean_ticket_cost = round(df_daily[df_daily["include_in_sum"]]["price"].mean(), 2) if len(df_daily) > 0 else 0.0
    mean_ticket_cost_wo_festival = round(df_no_festival["price"].mean(), 2) if len(df_no_festival) > 0 else 0.0

    # Calculate price per set
    sets_wo_festival = len(df_reset[df_reset[TYPE] != festival_label])
    price_per_set = round(total_ticket_cost / sets_seen, 2) if sets_seen > 0 else 0.0
    price_per_set_wo_festival = (
        round(total_ticket_cost_wo_festival / sets_wo_festival, 2) if sets_wo_festival > 0 else 0.0
    )

    # Calculate total festival cost (festivals counted once for multi-day festivals)
    df_festivals = df_daily[(df_daily["is_festival"]) & (df_daily["include_in_sum"])]
    total_festival_cost = df_festivals["price"].sum()

    # total countries
    total_countries = len(df_reset[COUNTRY].dropna().unique())
    total_cities = len(df_reset[CITY].dropna().unique())
    total_venues = len(df_reset[VENUE].dropna().unique())
    total_artists = len(df_reset[ARTIST].dropna().unique())

    logger.info(
        f"User analysis summary: {days_with_show} days, {sets_seen} sets, "
        f"€{total_ticket_cost:.2f} total cost, {total_artists} artists, {total_venues} venues, "
        f"{total_cities} cities, {total_countries} countries"
    )

    return UserAnalysis(
        days_with_show=days_with_show,
        days_with_show_wo_festival=days_with_show_wo_festival,
        sets_seen=sets_seen,
        sets_seen_wo_festival=sets_wo_festival,
        total_ticket_cost=round(total_ticket_cost, 2),
        total_ticket_cost_wo_festival=round(total_ticket_cost_wo_festival, 2),
        most_expensive=round(most_expensive, 2),
        most_expensive_show=most_expensive_show,
        most_expensive_wo_festival=round(most_expensive_wo_festival, 2),
        most_expensive_show_wo_festival=most_expensive_show_wo_festival,
        mean_ticket_cost=mean_ticket_cost,
        mean_ticket_cost_wo_festival=mean_ticket_cost_wo_festival,
        price_per_set=price_per_set,
        price_per_set_wo_festival=price_per_set_wo_festival,
        total_festival_cost=round(total_festival_cost, 2),
        total_countries=total_countries,
        total_cities=total_cities,
        total_venues=total_venues,
        total_artists=total_artists,
    )


def high_level_user_analysis(
    df_indexed: DataFrame,
    meta_info: MetaInfo,
    running_order_headline_last: bool,
    user_data_folder: Path,
    festival_label: str = "F",
    color_scheme: SVGStyleGuide = SVGStyleGuide(),
) -> list[Path]:
    """Perform high-level user analysis and return SVG paths."""
    logger.debug(f"Starting high-level user analysis with festival_label={festival_label}")
    visited_venues = count_column_occurrences(df_indexed, VENUE)
    venues_unique = visited_venues.unique()
    unique_artists = df_indexed[ARTIST].dropna().unique()

    user_analysis = collect_data_for_user_analysis(
        df_indexed, running_order_headline_last, festival_label=festival_label
    )

    # Count how many entries there are per day
    # Reset index to access DATE column, group by DATE to get entries per day
    df_reset_calendar = df_indexed.reset_index()
    entries_per_day = df_reset_calendar.groupby(DATE).size()
    entries_per_day = np.log(entries_per_day)

    logger.debug(f"Creating calendar visualization for {len(entries_per_day)} days")
    fig, ax = plt.subplots(figsize=(15, 6))

    dp.calendar(
        dates=entries_per_day.index.tolist(),
        values=entries_per_day.values.tolist(),
        start_date=f"{meta_info.year}-01-01",
        end_date=f"{meta_info.year}-12-31",
        ax=ax,
        cmap=LinearSegmentedColormap.from_list("my_cmap", [color_scheme.gradient_high, color_scheme.text_color]),
        day_kws={"color": color_scheme.text_color},
        month_kws={"color": color_scheme.text_color},
    )

    # Set calendar background to transparent
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    fig.tight_layout()

    map_svg = user_data_folder / "map.svg"
    plt.savefig(map_svg)
    logger.debug(f"Saved calendar map to {map_svg}")

    svg_text = UserAnalysis.related_svg_solo_export.read_text()

    with open(map_svg) as f:
        lines = f.readlines()[4:]
        lines.insert(
            0,
            '<svg xmlns:xlink="http://www.w3.org/1999/xlink" x="40" y="550" width="760pt" height="600pt" viewBox="0 0 1080 432" xmlns="http://www.w3.org/2000/svg" version="1.1">',
        )
        map_svg_text = "\n".join(lines)

    svg_text = svg_text.replace("<!--MAP-->", map_svg_text)

    # Apply user analysis and meta info to SVG
    svg_text = user_analysis.apply_self_to_text(svg_text)
    svg_text = meta_info.apply_self_to_text(svg_text)
    svg_text = color_scheme.apply_self_to_text(svg_text)

    user_svg_path = user_data_folder / "user-high-level.svg"
    user_svg_path.write_text(svg_text)
    logger.info(f"Saved high-level user analysis SVG to {user_svg_path}")

    return [user_svg_path]


def create_svgs_for(
    df_indexed: DataFrame,
    meta_info: MetaInfo,
    column: str,
    running_order_headline_last: bool,
    user_data_folder: Path,
    headline_label: str = "auto",
    support_label: str = "auto",
    festival_label: str = "F",
    color_scheme: SVGStyleGuide = SVGStyleGuide(),
) -> list[Path]:

    logger.debug(f"Creating SVGs for column: {column}")
    # TODO: smh reliably dedupe concert costs if artists from same night appear on different occasions
    attendance_counts = count_column_occurrences(df_indexed, column)
    _attendance_unique = attendance_counts.unique()
    _attendance_unique.sort()
    top_idxes = list(reversed(_attendance_unique.tolist()))
    logger.debug(f"Top {len(top_idxes)} attendance counts for {column}: {top_idxes}")

    svgs = []

    context_collector, collection_class = context_collectors[column]

    data_contexts_dict: defaultdict[int, list[MarkerDrivenBaseModel]] = defaultdict(list)
    for i, seen_nr in enumerate(top_idxes, start=1):
        elements = attendance_counts[attendance_counts == seen_nr].index.values
        logger.debug(f"Processing {len(elements)} elements with {seen_nr} occurrences for {column}")
        for a in elements:
            if column == ARTIST:
                ctx = context_collector(
                    df_indexed,
                    a,
                    position_in_ranking=i,
                    running_order_headline_last=running_order_headline_last,
                    column=column,
                    headline_label=headline_label,
                    support_label=support_label,
                    festival_label=festival_label,
                )
            else:
                ctx = context_collector(
                    df_indexed,
                    a,
                    position_in_ranking=i,
                    running_order_headline_last=running_order_headline_last,
                    column=column,
                )
            data_contexts_dict[seen_nr].append(ctx)

        key = lambda x: x.key()
        data_contexts_dict[seen_nr] = sorted(data_contexts_dict[seen_nr], key=key, reverse=False)

    # if this is the case we can do the cool single slide overview
    # bc all top numbers only occur once
    if all(map(lambda x: len(x) == 1, data_contexts_dict.values())):
        logger.debug(f"All top entries are unique for {column}, creating single overview SVG")
        svg_text = data_contexts_dict[next(iter(data_contexts_dict.keys()))][0].related_svg_unique_top_4.read_text()

        svg_text = meta_info.apply_self_to_text(svg_text)
        svg_text = color_scheme.apply_self_to_text(svg_text)

        for a in data_contexts_dict.values():
            element = a[0]
            svg_text = element.apply_self_to_text(svg_text)

            # TODO unique name & return
            output_path = user_data_folder / f"top-{column}.svg"
            output_path.write_text(svg_text)
            logger.debug(f"Saved overview SVG to {output_path}")

    # here we do one aggregate slide and then unique slides for each artist
    # else:
    #     # TODO: if one number is unique we can pull out the larger statistics
    #     for seen_nr, bands in data_contexts_dict.items():
    #         band_summary_svg_maker = collection_class(times=seen_nr, elements=bands)
    #
    #         svg_text = band_summary_svg_maker.related_svg_summary.read_text()
    #         svg_text = band_summary_svg_maker.apply_self_to_text(svg_text)
    #
    #         svg_text = meta_info.apply_self_to_text(svg_text)
    #
    #         # TODO unique name & return
    #         Path(f"out/summary-{colum}-{seen_nr}-times.svg").write_text(svg_text)

    # Flatten the data_contexts_dict to a 1D list of all MarkerDrivenBaseModel values using extend
    all_contexts_flat = []
    for values in data_contexts_dict.values():
        all_contexts_flat.extend(values)

    logger.debug(f"Generating {len(all_contexts_flat)} solo SVG files for {column}")
    svg_text_template = all_contexts_flat[0].related_svg_solo_export.read_text()
    svg_text_template = meta_info.apply_self_to_text(svg_text_template)
    svg_text_template = color_scheme.apply_self_to_text(svg_text_template)
    context: TopBandContext | VenueContext

    for context in all_contexts_flat:
        svg_text = context.apply_self_to_text(svg_text_template, is_ranked=False)

        output_path = user_data_folder / f"solo-{column}-{context.position}-{sanitize_filename(context.name)}.svg"
        output_path.write_text(svg_text)
        svgs.append(output_path)
        logger.debug(f"Saved solo SVG: {output_path}")

    logger.info(f"Generated {len(svgs)} SVG files for {column}")
    return svgs


def sanitize_filename(name):
    return re.sub(r"[^a-zA-Z0-9_-]", "", name)


def collect_data_for_venue_like(
    df: pd.DataFrame,
    venue: str,
    *,
    running_order_headline_last: bool,
    position_in_ranking: int,
    column: str = VENUE,
) -> VenueContext:
    """Collect venue-related data from the DataFrame to create a VenueContext."""

    logger.debug(f"Collecting data for venue: {venue} (position {position_in_ranking})")

    venue_df = df[df[column] == venue]

    # Collect all ticket prices for this venue
    prices = venue_df[PAID_PRICE].dropna().tolist()

    # Group by batch_id to get unique visit dates and count bands per night
    visit_dates = []
    num_bands_per_night = []
    headline_per_night = []
    prices = []

    for batch_id, group in venue_df.groupby(level=0):  # level=0 is batch_id
        if venue not in group[column].values:
            continue

        # Get the date from the DATE column (should be same for all rows in a batch)
        day = group[DATE].iloc[0]
        visit_dates.append(day)

        headline_per_night.append(group[QUALIFIED_NAME].iloc[0])
        # Count number of unique artists/bands for this night at this venue
        # TODO can dropna be evil here?!
        num_bands = len(group[ARTIST].dropna().unique())
        num_bands_per_night.append(num_bands)
        # Some ppl store the price only for the headline instead of for all bands, so wetake max instead of unique
        prices.append(group[PAID_PRICE].max().tolist())

    assert all(map(lambda x: isinstance(x, float), prices))

    logger.debug(f"Venue {venue}: {len(visit_dates)} visits, {len(prices)} price entries")

    context = VenueContext(
        position=position_in_ranking,
        name=venue,
        prices=prices,
        visit_dates=visit_dates,
        num_bands_per_night=num_bands_per_night,
        headline_per_night=headline_per_night,
    )

    return context


def collect_data_for_artist(
    df: pd.DataFrame,
    artist: str,
    *,
    running_order_headline_last: bool,
    position_in_ranking: int,
    column: str = ARTIST,
    headline_label: str = "auto",
    support_label: str = "auto",
    festival_label: str = "F",
) -> TopBandContext:

    logger.debug(
        f"Collecting data for artist: {artist} (position {position_in_ranking}), "
        f"headline_label={headline_label}, support_label={support_label}, festival_label={festival_label}"
    )
    target_index = -1 if running_order_headline_last else 0

    # Check for each day if the artist was at the target_index (headliner position) for that day
    show_classification_per_date = get_concert_types_for_artist(
        artist, df, festival_label, headline_label, target_index
    )

    show_classification_per_date = pd.DataFrame(show_classification_per_date, columns=[DATE, "is_headliner"])

    headliner_bool_array = show_classification_per_date["is_headliner"].to_numpy(dtype=int).tolist()

    artist_df = df[df[ARTIST] == artist]

    # Collect all venues this artist was seen in and save in a list called venues
    venues = artist_df[VENUE].dropna().tolist()

    # Collect all countries this artist was seen in and save in a list called countries
    countries = artist_df[COUNTRY].dropna().tolist()

    # Collect all ticket prices this artist was seen in and save in a list called prices
    prices = artist_df[PAID_PRICE].dropna().tolist()

    cities = artist_df[CITY].dropna().tolist()

    # Collect unique visit dates for this artist and headline_per_night
    visit_dates = []
    headline_per_night = []
    for batch_id, group in artist_df.groupby(level=0):  # level=0 is batch_id
        # Get the date from the DATE column (should be same for all rows in a batch)
        day = group[DATE].iloc[0]
        visit_dates.append(day)
        headline_per_night.append(group[QUALIFIED_NAME].iloc[0])

    headline_count = sum(1 for t in headliner_bool_array if t == TopBandContext.TYPE_HEADLINE)
    festival_count = sum(1 for t in headliner_bool_array if t == TopBandContext.TYPE_FESTIVAL)
    support_count = sum(1 for t in headliner_bool_array if t == TopBandContext.TYPE_SUPPORT)
    logger.debug(
        f"Artist {artist}: {len(visit_dates)} visits ({headline_count} headline, "
        f"{festival_count} festival, {support_count} support), {len(venues)} venues, "
        f"{len(countries)} countries, {len(cities)} cities"
    )

    context = TopBandContext(
        position=position_in_ranking,
        name=artist,
        classified_sets=headliner_bool_array,
        venues=venues,
        cities=cities,
        countries=countries,
        prices=prices,
        visit_dates=visit_dates,
        headline_per_night=headline_per_night,
    )

    return context


def get_concert_types_for_artist(
    artist: str, df: DataFrame, festival_label: str, headline_label: str, target_index: int
) -> list[tuple[datetime.datetime, int]]:

    classification_for_day = []

    for batch_id, group in df.groupby(level=0):  # level=0 is batch_id

        if artist not in group[ARTIST].values:
            continue

        # Get the date from the DATE column (should be same for all rows in a batch)
        day = group[DATE].iloc[0]

        # Check festival first (always uses TYPE column)
        type_val = group.loc[group[ARTIST] == artist, TYPE].iloc[0]

        headliner = determine_headliner_for_day(group, festival_label, headline_label, target_index)

        # Festival stays festival
        if festival_label in type_val:
            classification_for_day.append((day, TopBandContext.TYPE_FESTIVAL))
        # Headliner is the artist
        elif headliner == artist:
            classification_for_day.append((day, TopBandContext.TYPE_HEADLINE))
        # Default to support if neither headline nor support label matched
        else:
            classification_for_day.append((day, TopBandContext.TYPE_SUPPORT))

    return classification_for_day


def query_for_label(group: DataFrame, label: str, target_col: str = TYPE) -> DataFrame:
    return group.loc[group[target_col].str.contains(label, na=False)]


def determine_headliner_for_day(group: DataFrame, festival_label: str, headline_label: str, target_index: int) -> str:

    # detect by running order
    if headline_label == "auto":
        return group.iloc[target_index][ARTIST]

    # detect by festival label (choose venue as headliner, since we can't determine the headliner from the festival label alone)
    # we use contains since Franka has multiple labels for headline slots in festivals (e.g. "Festival, Main Act")
    elif len(query_for_label(group, festival_label)) > 0:
        return group.iloc[target_index][VENUE]

    # detect by headline label (choose the headliner from the headline label)
    else:
        # We need those rows of 'TYPE' where 'headline_label' is in the type string
        # reason for conatins at elif above
        return query_for_label(group, headline_label)[ARTIST].iloc[0]


context_collectors: dict[str, tuple[Callable, Callable]] = {
    ARTIST: (collect_data_for_artist, BandSeenSetSummary),
    VENUE: (collect_data_for_venue_like, VenueSummary),
    CITY: (collect_data_for_venue_like, VenueSummary),
}


if __name__ == "__main__":

    analyze_concert_csv_file(Path(__file__).parent / "Konzerte - Shows.csv", 2025, "cyber_chris", "Berlin")
