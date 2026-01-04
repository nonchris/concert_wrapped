from typing import Any

import pandas as pd

from concert_data_thing.constants import ARTIST
from concert_data_thing.constants import DATE
from concert_data_thing.constants import EVENT_NAME
from concert_data_thing.constants import PAID_PRICE
from concert_data_thing.constants import TYPE
from concert_data_thing.constants import VENUE
from concert_data_thing.logger import LOGGING_PROVIDER

logger = LOGGING_PROVIDER.new_logger("concert_data_thing.helpers.deduplication")


def _try_grouping_by_column(day_df: pd.DataFrame, column: str) -> list[pd.DataFrame] | None:
    """
    Try grouping a day DataFrame by a column and check if it makes ARTIST unique within each group.

    Args:
        day_df: DataFrame containing all rows for a single date
        column: Column name to group by

    Returns:
        List of grouped DataFrames if grouping makes ARTIST unique, None otherwise
    """
    if column not in day_df.columns or not day_df[column].notna().any():
        return None

    groups = []
    for _, group in day_df.groupby(column, sort=False):
        groups.append(group)

    # Check if grouping makes ARTIST unique within each group
    all_unique = ensure_uniqueness(groups, ARTIST)
    if all_unique:
        logger.debug(f"{column} grouping successful: {len(groups)} batches")
        return groups

    return None


def ensure_uniqueness(groups: list[Any], column: str) -> bool:
    all_unique = all(group[column].nunique() == len(group) for group in groups)
    return all_unique


def split_day_into_batches(day_df: pd.DataFrame, headline_label: str) -> list[pd.DataFrame]:
    """
    Split a day DataFrame into batches when ARTIST values are not unique.

    This function handles cases where multiple concerts occur on the same day.
    It tries different grouping strategies to separate concerts into distinct batches:
    1. Group by VENUE (most common case - different venues = different concerts)
    2. Group by price (PAID_PRICE or ORIGINAL_PRICE) if venue grouping fails
    3. Sequential grouping as fallback (create batches where ARTIST is unique)

    Args:
        day_df: DataFrame containing all rows for a single date

    Returns:
        List of DataFrames, one per batch (concert lineup)
    """
    # Check if ARTIST values are already unique - if so, return single batch
    if day_df[ARTIST].nunique() == len(day_df):
        logger.debug(f"All artists unique for date {day_df[DATE].iloc[0]}, single batch")
        return [day_df]

    # Try grouping strategies in order: VENUE, then price columns
    grouping_columns = [VENUE, PAID_PRICE, EVENT_NAME]
    logger.debug(f"Artists not unique for date {day_df[DATE].iloc[0]}, trying grouping strategies")

    for column in grouping_columns:
        groups = _try_grouping_by_column(day_df, column)
        if groups is not None:
            return groups

    # Strategy 3: Sequential grouping fallback
    # Create batches by iterating through rows and grouping consecutive rows
    # where ARTIST values remain unique within each batch
    # TODO we might be able to estimate the running order/ grouping from the df, but there is no guarantee that user data has it correct
    logger.debug("Doing sequential batching for day {day_df[DATE].iloc[0]}")
    batches = []
    batches_artist_names = []
    headliner_in_batch: list[bool] = []

    do_headline_detect = headline_label != "auto"

    if not do_headline_detect:
        logger.debug("Cannot ensure unique headliners, batching might be incorrect")

    for pos in range(len(day_df)):

        row = day_df.iloc[pos]
        artist = row[ARTIST]
        show_type = row[TYPE]
        # check if we can do headline detect and show is a headline
        is_show_headline = do_headline_detect and show_type == headline_label

        # find batch where artist isn't in yet and where no headliner collision occurs
        for i in range(len(batches)):

            # artist is already in batch
            if artist in batches_artist_names[i]:
                continue

            # check that we don't batch two headliners together
            elif headliner_in_batch[i] and is_show_headline:
                continue

            # yay we found them a home!
            batches[i].append(row)
            batches_artist_names.append(artist)
            if is_show_headline:
                headliner_in_batch[i] = True

            break  # break the cycle

        # If we didn't find a home, create a new batch
        else:
            batches.append([row])
            batches_artist_names.append([artist])
            headliner_in_batch.append(is_show_headline)

    # Rebuild lists of rows to dataframes
    batches = [pd.DataFrame(batch) for batch in batches]

    # Verify all batches have unique artists
    assert ensure_uniqueness(batches, ARTIST), "Batches are not unique, this should never happen"

    logger.debug(f"Sequential grouping successful: {len(batches)} batches")
    return batches
