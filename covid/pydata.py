"""Python-based data munging"""

import re
from warnings import warn

import numpy as np
import pandas as pd


def load_commute_volume(filename, date_range):
    """Loads commute data and clips or extends date range"""
    commute_raw = pd.read_excel(
        filename, index_col="Date", skiprows=5, usecols=["Date", "Cars"]
    )
    commute_raw.index = pd.to_datetime(commute_raw.index, format="%Y-%m-%d")
    commute_raw.sort_index(axis=0, inplace=True)
    commute = pd.DataFrame(
        index=np.arange(date_range[0], date_range[1], np.timedelta64(1, "D"))
    )
    commute = commute.merge(commute_raw, left_index=True, right_index=True, how="left")
    commute[commute.index < commute_raw.index[0]] = commute_raw.iloc[0, 0]
    commute[commute.index > commute_raw.index[-1]] = commute_raw.iloc[-1, 0]
    commute["Cars"] = commute["Cars"] / 100.0
    commute.columns = ["percent"]
    return commute


def load_mobility_matrix(flow_file):
    """Loads mobility matrix from rds file"""
    mobility = pd.read_csv(flow_file)  # list(pyr.read_r(flow_file).values())[0]
    mobility = mobility[
        mobility["From"].str.startswith("E") & mobility["To"].str.startswith("E")
    ]
    mobility = mobility.sort_values(["From", "To"])
    mobility = mobility.groupby(["From", "To"]).agg({"Flow": sum}).reset_index()
    mob_matrix = mobility.pivot(index="To", columns="From", values="Flow")
    mob_matrix[mob_matrix.isna()] = 0.0
    return mob_matrix


def load_population(pop_file):
    pop = pd.read_csv(pop_file, index_col="lad19cd")
    pop = pop[pop.index.str.startswith("E")]
    pop = pop.sum(axis=1)
    pop = pop.sort_index()
    pop.name = "n"
    return pop


def linelist2timeseries(date, region_code, date_range=None):
    """Constructs a daily aggregated timeseries given dates and region code
       Optionally accepts a list expressing a required date range."""

    linelist = pd.DataFrame(dict(date=pd.to_datetime(date), region_code=region_code))

    # 1. clip dates
    if date_range is not None:
        linelist = linelist[
            (date_range[0] <= linelist["date"]) & (linelist["date"] < date_range[1])
        ]
    raw_len = linelist.shape[0]

    # 2. Remove NA rows
    linelist = linelist.dropna(axis=0)  # remove na's
    warn(
        f"Removed {raw_len - linelist.shape[0]} rows of {raw_len} due to missing data \
({100. * (raw_len - linelist.shape[0])/raw_len}%)"
    )

    # 3. Aggregate by date/region and sort on index
    case_counts = linelist.groupby(["date", "region_code"]).size()
    case_counts.sort_index(axis=0, inplace=True)
    case_counts.name = "count"

    case_counts.name = "count"
    return case_counts


def _read_csv(filename, date_format="%m/%d/%Y"):
    data = pd.read_csv(filename, low_memory=False)
    data["specimen_date"] = pd.to_datetime(data["specimen_date"], format=date_format)
    data["lab_report_date"] = pd.to_datetime(
        data["lab_report_date"], format=date_format
    )
    return data


def _merge_ltla(series):
    london = ["E09000001", "E09000033"]
    corn_scilly = ["E06000052", "E06000053"]
    series.loc[series.isin(london)] = ",".join(london)
    series.loc[series.isin(corn_scilly)] = ",".join(corn_scilly)
    return series


def phe_case_data(linelisting_file, date_range=None, date_type="specimen", pillar=None):

    read_file = dict(csv=_read_csv, xlsx=pd.read_excel)
    match_extension = re.match(r"(.*)\.(.*)$", linelisting_file)
    if match_extension is None:
        raise ValueError(
            f"Linelisting filename '{linelisting_file}' is not in name.extension format"
        )
    filetype = match_extension.group(2)
    try:
        ll = read_file[filetype](linelisting_file)
    except KeyError:
        raise ValueError(f"No handler implemented for file type '{filetype}'")

    # Merged regions
    ll["LTLA_code"] = _merge_ltla(ll["LTLA_code"])
    all_regions = ll.loc[~ll["LTLA_code"].isna(), "LTLA_code"].sort_values().unique()

    # Choose pillar
    if pillar is not None:
        ll = ll.loc[ll["pillar"] == pillar]

    date_type_map = {"specimen": "specimen_date", "report": "lab_report_date"}

    date = ll[date_type_map[date_type]]

    ts = linelist2timeseries(date, ll["LTLA_code"], date_range)

    # Re-index spatial timeseries
    # 4. Reindex by day
    one_day = np.timedelta64(1, "D")
    full_dates = pd.date_range(date_range[0], date_range[1] - one_day)
    index = pd.MultiIndex.from_product(
        [full_dates, all_regions], names=["date", "region_code"]
    )
    case_counts = ts.reindex(index)
    case_counts.loc[case_counts.isna()] = 0.0

    return case_counts.reset_index().pivot(
        index="region_code", columns="date", values="count"
    )
