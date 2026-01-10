import datetime as dt
import io
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
from concert_data_thing.constants import INCLUDE_IN_PRICE
from concert_data_thing.constants import MERCH_COST
from concert_data_thing.constants import ORIGINAL_PRICE
from concert_data_thing.constants import PAID_PRICE
from concert_data_thing.constants import QUALIFIED_NAME
from concert_data_thing.constants import TYPE
from concert_data_thing.constants import TYPE_CLASSIFICATION
from concert_data_thing.constants import VENUE
from concert_data_thing.data_models.settings import SVGStyleGuide
from concert_data_thing.evnironment import ARTIFACTS_PATH
from concert_data_thing.helpers.color import moderate_color
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


def select_rows_by_date_range(
    df: pd.DataFrame,
    start_date: dt.datetime,
    end_date: dt.datetime,
    date_column: str = DATE,
) -> pd.DataFrame:
    """
    Select only rows within the given date range based on the date column.

    Args:
        df: DataFrame with a date column.
        start_date: The start date (inclusive).
        end_date: The end date (inclusive).
        date_column: Name of the date column (default DATE).

    Returns:
        DataFrame with only rows where the date_column is between start_date and end_date (inclusive).
    """
    initial_count = len(df)
    date_series = pd.to_datetime(df[DATE], errors="coerce")
    mask = (date_series >= start_date) & (date_series <= end_date)
    filtered_df = df[mask].reset_index(drop=True)
    logger.info(
        f"Filtered by date range {start_date.date()} to {end_date.date()}: "
        f"{initial_count} rows -> {len(filtered_df)} rows"
    )
    return filtered_df


def analyze_concert_csv_file(csv_path: Path, *args, **kwargs):
    """
    See analyze_concert_csv for more details.
    """
    return analyze_concert_csv(csv_path.read_text(), *args, **kwargs)


def analyze_concert_csv(
    csv_str: str,
    start_date: dt.datetime,
    end_date: dt.datetime,
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
        start_date (dt.datetime): Start date for filtering (inclusive).
        end_date (dt.datetime): End date for filtering (inclusive).
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
        f"Starting analysis for user={user_name}, date_range={start_date.date()} to {end_date.date()}, "
        f"city={city}, running_order_headline_last={running_order_headline_last}, request_id={request_id}"
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

    df = select_rows_by_date_range(df, start_date, end_date)

    # Create a MultiIndex DataFrame with batch_id as first index and a second index (0..N) per batch
    df_indexed = df.copy()

    # Group by batch_id and assign a second index from 0..N for each batch
    df_indexed["per_day_idx"] = df_indexed.groupby("batch_id").cumcount()
    df_indexed = df_indexed.set_index(["batch_id", "per_day_idx"])
    unique_batches = len(df_indexed.index.get_level_values(0).unique())
    logger.info(f"Created indexed DataFrame with {unique_batches} unique batches, {len(df_indexed)} total entries")

    # Classify each row with type_classification
    logger.debug("Classifying rows with type_classification")
    target_index = -1 if running_order_headline_last else 0
    classify_shows(df_indexed, headline_label, festival_label, target_index)

    # find headliners and attribute them
    # TODO: this does not work when there is only a festival+headliner atm
    #  in these cases we should go by the max of the batch id
    df_indexed[INCLUDE_IN_PRICE] = False
    df_indexed.loc[get_headline_rows_mask(df_indexed), INCLUDE_IN_PRICE] = True

    logger.info(f"Classified {len(df_indexed)} rows with type_classification")

    # Extract actual date range from filtered data
    date_series = pd.to_datetime(df_indexed[DATE], errors="coerce")
    # Extract year from the date range (use start_date year, or end_date year if they differ we could log a warning)
    year = start_date.year

    meta_info = MetaInfo(
        user_name=user_name,
        year=year,
        start_date_dt=start_date,
        end_date_dt=end_date,
    )

    # TODO: later use coockies to store the request_id (basically reclide uuids)
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


def classify_shows(df_indexed, headline_label: str, festival_label: str, target_index: int):
    df_indexed[TYPE_CLASSIFICATION] = None

    for batch_id, group in df_indexed.groupby(level=0):  # level=0 is batch_id
        # Determine headliner for this batch
        headliner = determine_headliner_for_day(group, festival_label, headline_label, target_index)

        # Classify each row in the batch
        for idx in group.index:
            artist = group.loc[idx, ARTIST]
            type_val = group.loc[idx, TYPE]

            # Check if it's a festival (always uses TYPE column)
            is_festival = pd.notna(type_val) and festival_label in str(type_val)
            is_headliner = headliner == artist

            # Festival headline: festival AND headliner
            if is_festival and is_headliner:
                df_indexed.loc[idx, TYPE_CLASSIFICATION] = TopBandContext.TYPE_FESTIVAL_HEADLINE
            # Festival (but not headliner)
            elif is_festival:
                df_indexed.loc[idx, TYPE_CLASSIFICATION] = TopBandContext.TYPE_FESTIVAL
            # Headliner (but not festival)
            elif is_headliner:
                df_indexed.loc[idx, TYPE_CLASSIFICATION] = TopBandContext.TYPE_HEADLINE
            # Default to support if neither headline nor support label matched
            else:
                df_indexed.loc[idx, TYPE_CLASSIFICATION] = TopBandContext.TYPE_SUPPORT


def find_most_expensive_ticket(df: DataFrame, key: str) -> tuple[float, str]:
    """
    Find the most expensive ticket from a daily summary DataFrame.

    Returns:
        Tuple of (price, show_description)
    """
    if len(df) > 0 and df[key].max() > 0:
        most_expensive_idx = df[key].idxmax()
        most_expensive_row = df.loc[most_expensive_idx]
        most_expensive = most_expensive_row[key]
        most_expensive_show = f"{most_expensive_row[QUALIFIED_NAME]}"
        return most_expensive, most_expensive_show
    else:
        return 0.0, "[unknown]"


def collect_data_for_user_analysis(
    df_indexed: DataFrame, running_order_headline_last: bool, festival_label: str = "F"
) -> UserAnalysis:
    """
    Collect user-level analysis data from the DataFrame.

    Uses batch_id for most statistics (concerts/shows) but keeps unique dates
    for day count metrics (used for dayplot visualization).
    Handles multi-day festivals by only counting ticket cost once for sequential
    days where TYPE equals festival_label and VENUE is the same.
    """
    logger.debug(f"Collecting user-level analysis data with festival_label={festival_label}")
    # Reset index to work with dates and batch_id as columns
    df_reset = df_indexed.reset_index()

    # Count unique days with shows (for day count and dayplot)
    unique_dates = df_reset[DATE].unique()
    days_with_show = len(unique_dates)

    # Count unique concerts/shows (using batch_id)
    unique_batches = df_reset["batch_id"].unique()
    unique_concerts = len(unique_batches)

    # Count total sets seen (total number of entries)
    sets_seen = len(df_indexed)

    df_pricing = df_indexed[df_indexed[INCLUDE_IN_PRICE] == True]

    total_ticket_cost = df_pricing[PAID_PRICE].sum()
    total_ticket_value = df_pricing[ORIGINAL_PRICE].sum()

    # Calculate total ticket cost without festivals
    df_no_festival = df_pricing[~(get_festival_row_mask(df_indexed))]
    total_ticket_cost_wo_festival = df_no_festival[PAID_PRICE].sum()

    # Find most expensive ticket
    most_expensive, most_expensive_show = find_most_expensive_ticket(df_pricing, PAID_PRICE)

    # Find most expensive ticket without festivals
    most_expensive_wo_festival, most_expensive_show_wo_festival = find_most_expensive_ticket(df_no_festival, PAID_PRICE)

    # Calculate mean ticket costs (using batch_id aggregation)
    mean_ticket_cost = round(df_pricing[PAID_PRICE].mean(), 2) if len(df_indexed) > 0 else 0.0
    mean_ticket_cost_wo_festival = round(df_no_festival[PAID_PRICE].mean(), 2) if len(df_no_festival) > 0 else 0.0

    # Only calculate discounts where the paid price is not zero
    non_zero_paid = df_pricing[df_pricing[PAID_PRICE] != 0]
    df_indexed["discounts_non_zero"] = non_zero_paid[ORIGINAL_PRICE] - non_zero_paid[PAID_PRICE]
    # Find idxmax only once to avoid duplication
    discounts_non_zero = df_indexed["discounts_non_zero"]
    idxmax_non_zero = discounts_non_zero.idxmax()
    highest_discount_non_zero = discounts_non_zero.max()
    highest_discount_band_non_zero = df_indexed.loc[idxmax_non_zero, QUALIFIED_NAME]
    highest_discount_original_price_non_zero = df_indexed.loc[idxmax_non_zero, ORIGINAL_PRICE]


    df_indexed["discounts_all"] = df_pricing[ORIGINAL_PRICE] - df_pricing[PAID_PRICE]
    idxmax_discount_all = df_indexed["discounts_all"].idxmax()
    highest_discount = df_indexed.loc[idxmax_discount_all, "discounts_all"]
    highest_discount_band = df_indexed.loc[idxmax_discount_all, QUALIFIED_NAME]
    highest_discount_original_price = df_indexed.loc[idxmax_discount_all, ORIGINAL_PRICE]

    # most expensive month by sum of PAID_PRICE (we want the month name)
    most_expensive_month = df_indexed[PAID_PRICE].groupby(df_indexed[DATE].dt.month).sum().idxmax()
    most_expensive_month_name = pd.to_datetime(most_expensive_month, unit="M").strftime("%B")

    free_shows_cnt = df_indexed[PAID_PRICE].eq(0).sum()


    # Calculate price per set

    # Select all rows where the first level index matches those in df_no_festival
    df_all_sets_wo_festival = df_indexed.loc[
        df_indexed.index.get_level_values(0).isin(df_no_festival.index.get_level_values(0))
    ]
    sets_wo_festival_cnt = len(df_all_sets_wo_festival)
    price_per_set = round(total_ticket_cost / sets_seen, 2) if sets_seen > 0 else 0.0
    price_per_set_wo_festival = (
        round(total_ticket_cost_wo_festival / sets_wo_festival_cnt, 2) if sets_wo_festival_cnt > 0 else 0.0
    )

    # Calculate total festival cost (festivals counted once for multi-day festivals)
    df_festivals = df_pricing[get_festival_row_mask(df_pricing)]
    total_festival_cost = df_festivals[PAID_PRICE].sum()

    # total countries
    total_countries = len(df_reset[COUNTRY].dropna().unique())
    total_cities = len(df_reset[CITY].dropna().unique())
    total_venues = len(df_reset[VENUE].dropna().unique())
    total_artists = len(df_reset[ARTIST].dropna().unique())

    logger.info(
        f"User analysis summary: {days_with_show} days, {unique_concerts} concerts, {sets_seen} sets, "
        f"€{total_ticket_cost:.2f} total cost, {total_artists} artists, {total_venues} venues, "
        f"{total_cities} cities, {total_countries} countries"
    )

    return UserAnalysis(
        unique_events=unique_concerts,
        unique_events_wo_festival=len(df_no_festival),
        days_with_show=days_with_show,
        days_with_show_wo_festival=len(df_no_festival[DATE].unique()),
        sets_seen=sets_seen,
        sets_seen_wo_festival=sets_wo_festival_cnt,
        total_ticket_cost=round(total_ticket_cost, 2),
        total_ticket_cost_wo_festival=round(total_ticket_cost_wo_festival, 2),
        # TODO: for multi-day festivals the prices are NOT summed up, so this doesn't work rn
        #  hence it is most likely equivalent to marker_most_expensive_wo_festival (hence deactivated)
        # most_expensive=round(most_expensive, 2),
        # most_expensive_show=most_expensive_show,
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

        total_ticket_value=round(total_ticket_value, 2),
        most_expensive_month=most_expensive_month_name,
        free_shows_cnt=free_shows_cnt,

        highest_discount=round(highest_discount, 2),
        highest_discount_band=highest_discount_band,
        highest_discount_original_price=round(highest_discount_original_price, 2),

        highest_discount_non_zero=round(highest_discount_non_zero, 2),
        highest_discount_non_zero_band=highest_discount_band_non_zero,
        highest_discount_non_zero_original_price=round(highest_discount_original_price_non_zero, 2),
       
    )


def get_headline_rows_mask(df_indexed: DataFrame) -> DataFrame:
    return (df_indexed[TYPE_CLASSIFICATION] == TopBandContext.TYPE_HEADLINE) | (
        df_indexed[TYPE_CLASSIFICATION] == TopBandContext.TYPE_FESTIVAL_HEADLINE
    )


def get_festival_row_mask(df_indexed: DataFrame) -> DataFrame:
    return (df_indexed[TYPE_CLASSIFICATION] == TopBandContext.TYPE_FESTIVAL) | (
        df_indexed[TYPE_CLASSIFICATION] == TopBandContext.TYPE_FESTIVAL_HEADLINE
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
        start_date=meta_info.start_date_dt.strftime("%Y-%m-%d"),
        end_date=meta_info.end_date_dt.strftime("%Y-%m-%d"),
        ax=ax,
        color_for_none=moderate_color(color_scheme.gradient_low, saturation_factor=0.5, brightness_factor=0.5),
        cmap=LinearSegmentedColormap.from_list(
            "my_cmap",
            [
                moderate_color(color_scheme.text_color, brightness_factor=0.1, saturation_factor=0.1),
                color_scheme.text_color,
            ],
        ),
        day_kws={"color": color_scheme.text_color},
        month_kws={"color": color_scheme.text_color},
        week_starts_on="Monday",
    )

    # Set calendar background to transparent
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    fig.tight_layout()

    map_svg = user_data_folder / "map.svg"
    plt.savefig(map_svg)
    logger.debug(f"Saved calendar map to {map_svg}")

    svg_text = UserAnalysis.related_svg_solo_export.read_text()

    svg_text = insert_sub_image_map(svg_text, map_svg)

    # Apply user analysis and meta info to SVG
    svg_text = user_analysis.apply_self_to_text(svg_text)
    svg_text = meta_info.apply_self_to_text(svg_text)
    svg_text = color_scheme.apply_self_to_text(svg_text)

    user_svg_path = user_data_folder / "user-high-level.svg"
    user_svg_path.write_text(svg_text)
    logger.info(f"Saved high-level user analysis SVG to {user_svg_path}")

    return [user_svg_path]


def insert_sub_image_map(svg_text: str, map_svg: Path) -> str:
    with open(map_svg) as f:
        lines = f.readlines()[4:]
        lines.insert(
            0,
            '<svg xmlns:xlink="http://www.w3.org/1999/xlink" x="40" y="830" width="760pt" height="600pt" viewBox="0 0 1080 432" xmlns="http://www.w3.org/2000/svg" version="1.1">',
        )
        map_svg_text = "\n".join(lines)

    svg_text = svg_text.replace("<!--MAP-->", map_svg_text)
    return svg_text


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
    return re.sub(r"[^a-zA-Z0-9_\- äöüÄÖÜß.]", "", name)


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

    # All rows of the artist
    artist_df = df[df[ARTIST] == artist]
    # Get all rows whose first level index matches the indices in artist_df (i.e., the batch_ids this artist was at)
    all_artist_related_rows = df[df.index.get_level_values(0).isin(artist_df.index.get_level_values(0))]

    # Collect all venues this artist was seen in and save in a list called venues
    venues = artist_df[VENUE].dropna().tolist()

    # Collect all countries this artist was seen in and save in a list called countries
    countries = artist_df[COUNTRY].dropna().tolist()

    # Collect all ticket prices this artist was seen in and save in a list called prices
    # Query the full related dataframe, take the max paid price per index (some ppl only enter price for headliner)
    # We already have a multi-index, so we can use .max(level=0) directly to get max per batch_id
    # Group by the first level of the multi-index (batch_id), then take the max per group, then tolist
    prices = all_artist_related_rows[PAID_PRICE].dropna().groupby(level=0).max().tolist()

    cities = artist_df[CITY].dropna().tolist()

    # Collect unique visit dates for this artist, headline_per_night, and type_classification
    visit_dates = []
    headline_per_night = []
    show_classification_array = []
    for batch_id, group in artist_df.groupby(level=0):  # level=0 is batch_id
        # Get the date from the DATE column (should be same for all rows in a batch)
        day = group[DATE].iloc[0]
        visit_dates.append(day)
        headline_per_night.append(group[QUALIFIED_NAME].iloc[0])
        # Get the type_classification for this artist in this batch (should be same for all rows of same artist in batch)
        classification = group[TYPE_CLASSIFICATION].iloc[0]
        show_classification_array.append(classification)

    context = TopBandContext(
        position=position_in_ranking,
        name=artist,
        classified_sets=show_classification_array,
        venues=venues,
        cities=cities,
        countries=countries,
        prices=prices,
        visit_dates=visit_dates,
        headline_per_night=headline_per_night,
    )

    return context


def query_for_label(group: DataFrame, label: str, target_col: str = TYPE) -> DataFrame:
    # we use contains since Franka has multiple labels for headline slots in festivals (e.g. "Festival, Main Act")
    return group.loc[group[target_col].str.contains(label, na=False)]


def determine_headliner_for_day(group: DataFrame, festival_label: str, headline_label: str, target_index: int) -> str:

    headline_query = query_for_label(group, headline_label)

    # detect by running order
    if headline_label == "auto":
        return group.iloc[target_index][ARTIST]

    # let's see of we have a headliner
    elif len(headline_query) == 1:
        return headline_query.iloc[target_index][ARTIST]

    # detect by festival label (we dont have a headline tag)
    # we will take the column with the highest amount, so we have the best chance to find the headline
    # and not mess with our price calculations, which builds on headliner rows
    elif len(query_for_label(group, festival_label)) > 0:
        return group.iloc[group[PAID_PRICE].values.argmax()][ARTIST]

    else:
        raise HTTPException(
            400,
            f"Did you set a headline label? "
            f"Can't detect headliner for date {group.iloc[0][DATE]} - Artists: {group[ARTIST].values}.",
        )


context_collectors: dict[str, tuple[Callable, Callable]] = {
    ARTIST: (collect_data_for_artist, BandSeenSetSummary),
    VENUE: (collect_data_for_venue_like, VenueSummary),
    CITY: (collect_data_for_venue_like, VenueSummary),
}


if __name__ == "__main__":

    analyze_concert_csv_file(Path(__file__).parent / "Konzerte - Shows.csv", 2025, "cyber_chris", "Berlin")
