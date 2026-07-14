# -*- coding: utf-8 -*-
"""
============================================================
MRP FORECAST ANALYTICS DASHBOARD (Streamlit)
============================================================
Converted from the original Colab analysis (Inventory.ipynb).

HOW TO RUN (Windows Command Prompt):
    1. Place this file and 'inventory.csv' in the same folder.
    2. cd into that folder.
    3. Run:  streamlit run inventory_dashboard.py

Requirements (install once):
    pip install streamlit pandas numpy matplotlib seaborn
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ------------------------------------------------------------
# PAGE CONFIGURATION (must be the first Streamlit call)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Material Requirements Planning Forecast Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# GLOBAL PLOTTING DEFAULTS
# ------------------------------------------------------------
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"


def fmt_thousands(ax, axis="y"):
    """Comma-format an axis (e.g. 1,250,000)."""
    formatter = mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def render_fig(fig):
    """Display a matplotlib figure in Streamlit and free its memory."""
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ------------------------------------------------------------
# DATA LOADING & CLEANING (cached - runs once, not on every rerun)
# Mirrors CELLS 1, 6 and 7 of the original analysis.
# ------------------------------------------------------------
@st.cache_data(show_spinner="Loading and preparing inventory.csv ...")
def load_data(path: str = "inventory.csv"):
    """Load the raw extract and apply the full cleaning pipeline."""
    # sep=None + engine='python' lets pandas sniff comma vs tab delimiters
    df_raw = pd.read_csv(path, sep=None, engine="python")

    df = df_raw.copy()

    # --- Dates: dd-mm-yyyy format (e.g. 10-10-2015) ---
    date_cols = ["DMND_WEEK_STRT_DATE", "VRSN_WEEK_STRT_DATE"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format="%d-%m-%Y", errors="coerce")

    # --- Numeric columns ---
    df["MRP_FCST_QTY"] = pd.to_numeric(df["MRP_FCST_QTY"], errors="coerce")
    df["LEAD_TIME"] = pd.to_numeric(df["LEAD_TIME"], errors="coerce").astype("Int64")

    # --- IDs stay as strings (you don't sum a CHANNEL_ID) ---
    for col in ["CHANNEL_ID", "PRODUCT_ID", "COMP_ITEM_ID"]:
        df[col] = df[col].astype(str).str.strip()

    # --- Low-cardinality strings -> category ---
    for col in ["CCN", "COMMODITY", "SUPPLY_CHAIN_TYPE", "SCHEDULER_NAME", "PRODUCT_NAME"]:
        df[col] = df[col].astype("category")

    # --- Drop the truncated DW load timestamp (no analytical value) ---
    if "DW_PKG_UPD_DTS" in df.columns:
        df = df.drop(columns=["DW_PKG_UPD_DTS"])

    # --- Derived fields ---
    df["FCST_HORIZON_DAYS"] = (df["DMND_WEEK_STRT_DATE"] - df["VRSN_WEEK_STRT_DATE"]).dt.days
    df["FCST_HORIZON_WKS"] = (df["FCST_HORIZON_DAYS"] / 7).round(1)
    df["DMND_YEAR"] = df["DMND_WEEK_STRT_DATE"].dt.year
    df["DMND_MONTH"] = df["DMND_WEEK_STRT_DATE"].dt.to_period("M").astype(str)
    df["ZERO_FCST"] = df["MRP_FCST_QTY"].eq(0)

    return df_raw, df


# ------------------------------------------------------------
# SECTION 0 - OVERVIEW & DATA QUALITY (EDA)
# ------------------------------------------------------------
def section_overview(df_raw, df):
    st.header("1 · Dataset Overview & Data Preparation")

    st.markdown(
        """
**Business purpose.** Before any chart can be trusted, the raw MRP extract
must be profiled and typed correctly. This section documents the dataset's
structure, the cleaning decisions applied, and the derived fields that
power the analyses that follow.
"""
    )

    # --- KPI cards ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Forecast records", f"{len(df):,}")
    c2.metric("Total forecast (units)", f"{df['MRP_FCST_QTY'].sum():,.0f}")
    c3.metric("Components", f"{df['COMP_ITEM_ID'].nunique():,}")
    c4.metric("Commodities", f"{df['COMMODITY'].nunique():,}")
    c5.metric("Zero-qty rows", f"{df['ZERO_FCST'].mean():.1%}")

    d_min = df["DMND_WEEK_STRT_DATE"].min()
    d_max = df["DMND_WEEK_STRT_DATE"].max()
    st.caption(
        f"Demand horizon: **{d_min:%d %b %Y} → {d_max:%d %b %Y}** · "
        f"Forecast versions: **{df['VRSN_WEEK_STRT_DATE'].nunique()}** snapshots"
    )

    st.divider()

    # --- Data profile expanders ---
    left, right = st.columns(2)

    with left:
        with st.expander("Sample of the data (first 10 rows)", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)

        with st.expander("Missing values & duplicates"):
            missing = (
                df_raw.isna().sum()
                .to_frame("n_missing")
                .assign(pct_missing=lambda x: (x["n_missing"] / len(df_raw) * 100).round(2))
                .sort_values("n_missing", ascending=False)
            )
            st.dataframe(missing, use_container_width=True)
            st.write(f"**Duplicate rows:** {df_raw.duplicated().sum():,}")

    with right:
        with st.expander("Column cardinality (unique values per column)"):
            cardinality = (
                df_raw.nunique()
                .to_frame("n_unique")
                .assign(pct_unique=lambda x: (x["n_unique"] / len(df_raw) * 100).round(2))
                .sort_values("n_unique")
            )
            st.dataframe(cardinality, use_container_width=True)

        with st.expander("Numeric summary (after cleaning)"):
            st.dataframe(
                df[["LEAD_TIME", "MRP_FCST_QTY", "FCST_HORIZON_WKS"]]
                .describe()
                .round(2),
                use_container_width=True,
            )

    with st.expander("Cleaning decisions applied to this dataset"):
        st.markdown(
            """
- **Dates** (`DMND_WEEK_STRT_DATE`, `VRSN_WEEK_STRT_DATE`) parsed with an
  explicit `dd-mm-yyyy` format - never guessed, to avoid silent day/month swaps.
- **IDs** (`CHANNEL_ID`, `PRODUCT_ID`, `COMP_ITEM_ID`) kept as **strings**:
  they look numeric but must never be summed or averaged.
- **Low-cardinality text** converted to `category` dtype for memory and speed.
- **`DW_PKG_UPD_DTS` dropped** - a truncated data-warehouse load timestamp
  with no analytical value.
- **Derived fields:** forecast horizon (demand week − version week),
  calendar helpers, and a zero-forecast flag used throughout this dashboard.
"""
        )


# ------------------------------------------------------------
# SECTION 1 - FORECASTED DEMAND BY COMMODITY
# ------------------------------------------------------------
def section_demand_by_commodity(df):
    st.header("2 · Forecasted Demand by Commodity")

    st.markdown(
        """
**Business purpose.** This chart aggregates the total MRP forecasted quantity
across all weeks, products, and channels in the dataset, grouped by commodity
category (e.g., Battery, LCD, HDD, ODD Mechanical).

**How to read it:** each bar represents the total number of units the MRP
system expects to be required for that commodity over the full planning
horizon covered by this extract. Longer bars = higher planned material
consumption. Exact totals are labeled at the end of each bar.

**Important caveat:** this reflects *forecasted demand*, not stock on hand.
The dataset contains planning quantities (MRP_FCST_QTY), so this view answers
"which commodities drive the most material requirements?" - not "how much
inventory do we currently hold?"

**Why it matters:** commodities at the top of this chart are where forecast
errors are most expensive. They should get the tightest planning cadence and
the most scrutiny in the lead-time and risk analyses that follow.
"""
    )

    demand_by_comm = (
        df.groupby("COMMODITY", observed=True)["MRP_FCST_QTY"]
        .sum()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    demand_by_comm.plot(kind="barh", ax=ax, color="#2f6fb2")
    ax.set_title("Total Forecasted Demand by Commodity")
    ax.set_xlabel("MRP Forecast Qty (units)")
    ax.set_ylabel("")
    fmt_thousands(ax, axis="x")
    for i, v in enumerate(demand_by_comm):
        ax.text(v, i, f" {v:,.0f}", va="center", fontsize=8)
    plt.tight_layout()
    render_fig(fig)

    with st.container(border=True):
        st.subheader("Key Insights")
        st.markdown(
            """
**1. Demand is heavily concentrated in mechanical and commodity components.**
ODD Mechanical alone accounts for ~2.9M forecasted units, more than double the
next commodity (ODM Mechanical, 1.26M). The top four categories
(ODD Mechanical, ODM Mechanical, Memory, CORD) represent roughly 55% of all
forecasted volume. Forecast accuracy efforts should be weighted accordingly:
a 5% error on ODD Mechanical moves more units than a 50% error on most of the
bottom half of this chart.

**2. High-volume ≠ high-value.** The volume leaders (mechanical parts, cords,
mice, keyboards) are largely low-cost commodity items, while lower-volume
categories such as LCD (412K), SSD (189K), and Graphic Card (176K) carry far
higher unit cost and supply risk. This chart should be read alongside
lead-time and cost data - planning priority is a function of
volume × cost × replenishment risk, not volume alone.

**3. The long tail is thin.** The bottom ten categories combined contribute
under 3% of total volume. These items add planning lines and catalog
maintenance effort while moving little material - candidates for simplified
planning policies (min/max or reorder point) rather than weekly forecast
review.

**4. Data hygiene finding: duplicate category labels.** "Carrying case"
(70,409 units) and "Carrying Case" (9,023 units) appear as separate
commodities due to inconsistent capitalization in the source system.
Combined, the category totals ~79K units. The same pattern check should be
applied across the catalog (e.g., ODD vs. ODD Mechanical may warrant
confirmation that the split is intentional). Category-level reporting is only
as reliable as category labels.
"""
        )


# ------------------------------------------------------------
# SECTION 2 - LEAD TIME ANALYSIS
# ------------------------------------------------------------
def section_lead_time(df):
    st.header("3 · Lead Time Analysis by Commodity")

    st.markdown(
        """
**Business purpose.** Two complementary views of procurement lead times, in weeks.

**Left - Average lead time per commodity.** A quick ranking of which commodity
categories take longest to replenish, sorted from slowest to fastest.
Commodities on the left of this chart are the ones where demand surprises are
hardest to recover from.

**Right - Lead time vs forecasted volume.** Each point is a commodity,
positioned by its average lead time and its total forecasted volume
(log scale). This view answers the question that matters for risk: *does the
volume live in slow or fast commodities?*

**Note:** lead times here are weighted by forecast records, meaning components
that appear in more product/channel/week combinations influence the picture
more heavily. This reflects lead-time exposure across the plan, rather than a
simple catalog average.
"""
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    lt_avg = (
        df.groupby("COMMODITY", observed=True)["LEAD_TIME"]
        .mean()
        .sort_values(ascending=False)
    )
    lt_avg.plot(kind="bar", ax=axes[0], color="#c05a2e")
    axes[0].set_title("Average Lead Time by Commodity")
    axes[0].set_ylabel("Weeks")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=90, labelsize=8)

    comm_view = (
        df.groupby("COMMODITY", observed=True)
        .agg(lead_time=("LEAD_TIME", "mean"), total_fcst=("MRP_FCST_QTY", "sum"))
        .reset_index()
    )
    sns.scatterplot(
        data=comm_view, x="lead_time", y="total_fcst",
        s=80, color="#c05a2e", ax=axes[1],
    )
    for _, r in comm_view.iterrows():
        axes[1].annotate(
            r["COMMODITY"], (r["lead_time"], r["total_fcst"]),
            fontsize=7, xytext=(4, 2), textcoords="offset points",
        )
    axes[1].set_yscale("log")
    axes[1].set_title("Lead Time vs Forecasted Volume by Commodity")
    axes[1].set_xlabel("Avg Lead Time (weeks)")
    axes[1].set_ylabel("Total Forecast Qty (log)")

    plt.tight_layout()
    render_fig(fig)

    with st.container(border=True):
        st.subheader("Key Insights")
        st.markdown(
            """
**1. Lead times are tiered planning parameters, not measured values.**
Lead times cluster at fixed levels - 8, 13, and 17–24 weeks - rather than
varying continuously. This reflects how the planning system is maintained:
standard lead-time tiers assigned per commodity. The practical implication is
that true supplier variability is invisible in this extract; the values
represent planning assumptions, and any supplier performing worse than the
assumed tier is a hidden risk.

**2. The highest-volume commodities sit in the mid lead-time band.**
ODD Mechanical (~2.9M units), ODM Mechanical, and Memory - the three biggest
demand drivers - all carry ~13-week lead times. Roughly a quarter's worth of
pipeline sits behind each unit of these forecasts: a demand signal error
today cannot be corrected on the supply side for three months. These
commodities justify the tightest forecast review cadence in the portfolio.

**3. The longest lead times belong to low-volume items - with two exceptions
worth watching.** Mobile Computing Cart (24 wks), Controller Card (22 wks),
and Projector (17 wks) are slow but small, limiting their aggregate impact.
However, **Graphic Card (~176K units at 19 weeks)** and **Nic Card
(~81K units at 22 weeks)** combine meaningful volume with the slowest
replenishment in the portfolio. These two are the standout structural risks
in this view and should be prioritized for safety stock review or supplier
lead-time reduction.

**4. The fast movers are well-positioned.** Wireless, HDD, SSD, Mouse, and
Docking all sit at 8 weeks with substantial volumes (150K–600K units). Their
short pipelines mean forecast errors are recoverable within the planning
cycle - these categories can tolerate leaner buffers and more reactive
planning.

**5. The quiet outlier: CORD.** At 1.1M units with a 10-week lead time, CORD
carries top-4 volume with below-average replenishment risk - an example of a
category where high volume does not translate into high planning risk.
Contrast with Graphic Card: six times less volume but nearly double the
pipeline.
"""
        )


# ------------------------------------------------------------
# SECTION 3 - DEMAND OVER TIME
# ------------------------------------------------------------
def section_demand_over_time(df):
    st.header("4 · Forecasted Demand Over Time")

    st.markdown(
        """
**Business purpose.**

**Top - Total weekly forecasted demand.** Every forecast record is aggregated
by demand week to show the overall shape of the planning horizon:
seasonality, ramp-ups for new products, end-of-life declines, and any abrupt
spikes or cliffs that warrant investigation.

**Bottom - Weekly demand for the top 5 commodities (stacked area).** The same
timeline, broken out by the five commodity categories with the highest total
volume. The overall height of the stack tracks their combined demand, and
each colored band shows how much a single commodity contributes in a given
week.

**How to read it:** look for shifts in the *mix*, not just the total. A stable
total can hide one commodity ramping down while another ramps up - a pattern
typical of product transitions, where new platforms replace old ones
component by component.

**Caution on the edges:** the first and last few weeks of the chart may show
artificially low demand. This usually reflects the boundaries of the data
extract (partial forecast coverage), not a real collapse in demand, and
should not be read as a trend.
"""
    )

    weekly = (
        df.groupby("DMND_WEEK_STRT_DATE")["MRP_FCST_QTY"]
        .sum()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    weekly.plot(ax=ax, color="#2f6fb2", lw=1.5)
    ax.set_title("Total Forecasted Demand per Week")
    ax.set_ylabel("Units")
    ax.set_xlabel("Demand Week")
    fmt_thousands(ax)
    plt.tight_layout()
    render_fig(fig)

    top5 = (
        df.groupby("COMMODITY", observed=True)["MRP_FCST_QTY"]
        .sum().nlargest(5).index
    )
    weekly_comm = (
        df[df["COMMODITY"].isin(top5)]
        .groupby(["DMND_WEEK_STRT_DATE", "COMMODITY"], observed=True)["MRP_FCST_QTY"]
        .sum()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    weekly_comm.plot.area(ax=ax, alpha=0.85, cmap="tab10")
    ax.set_title("Weekly Forecasted Demand - Top 5 Commodities")
    ax.set_ylabel("Units")
    ax.set_xlabel("Demand Week")
    fmt_thousands(ax)
    ax.legend(title="Commodity", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    render_fig(fig)

    with st.container(border=True):
        st.subheader("Key Insights")
        st.markdown(
            """
**1. The overall arc - read with caution.** Weekly forecasted demand rises
from ~80K units in October 2015, peaks between April and July 2016 at
300–350K units per week, then declines steadily to near zero by April 2017.
Before interpreting this as a business cycle, note the structural caveat
below (#4) - part of this shape is likely an artifact of how forecast
extracts are built.

**2. Peak planning season: Q2 2016.** The highest sustained demand sits in
April–July 2016, with the single largest week (~350K units) in mid-July. If
this window reflects real demand concentration, it aligns with typical
PC-industry cycles (back-to-school build ahead of Q3 and corporate refresh
activity), and it defines the period where component supply - especially the
13-week lead-time commodities - is under the most pressure. Materials for a
July peak must be committed by April.

**3. Demand is volatile week to week.** Swings of 50–150K units between
consecutive weeks are common (e.g., the drop from ~317K to ~135K and rebound
to ~350K around July 2016). Week-level volatility of this magnitude, if real,
argues for planning at a smoothed (3–4 week rolling) level rather than
chasing individual weekly signals.

**4. Structural caveat: the ramp-and-fade shape is partly an extract
artifact.** This dataset aggregates multiple forecast versions
(VRSN_WEEK_STRT_DATE). Weeks in the middle of the horizon are covered by many
forecast snapshots and accumulate quantity from each, while weeks at the
edges are covered by few - inflating the center and deflating the extremes.
The near-zero tail in early 2017 most likely reflects fewer forecast versions
reaching that far ahead, not a collapse in expected demand.
Version-controlled views (single-snapshot or latest-version-only) are
required before this curve can be read as a true demand profile.
"""
        )


# ------------------------------------------------------------
# SECTION 4 - SUPPLY CHAIN TYPE MIX
# ------------------------------------------------------------
def section_supply_chain_mix(df):
    st.header("5 · Supply Chain Type Mix: CTO vs BTO vs BTP vs BTS")

    st.markdown(
        """
**Business purpose.** The dataset spans four fulfillment models, ordered here
from most order-driven to most forecast-driven:

- **CTO (Configure to Order):** the customer defines the configuration;
  nothing is built until the order lands. Maximum flexibility, hardest to
  plan at the component level.
- **BTO (Build to Order):** predefined configurations built only when an
  order arrives.
- **BTP (Build to Plan):** production is triggered by the demand plan rather
  than firm orders - components and builds are committed against the forecast
  before orders exist.
- **BTS (Build to Stock):** standard units built to forecast and held as
  finished inventory. Easiest to execute, highest inventory risk.

**Left - Records by type.** The share of forecast lines each model generates -
a proxy for planning complexity.

**Right - Forecasted units by type.** The same split measured in actual
units. Comparing the two panels reveals asymmetry between planning effort and
volume.

**Heatmap - Commodity × supply chain type.** Where volume actually lives at
the intersection of component category and fulfillment model. Dark cells are
the heart of the plan. A commodity concentrated in BTS or BTP can be buffered
with planned stock; the same commodity flowing mostly through CTO demands
short, reliable supplier lead times because there is no finished-goods buffer
to hide behind.

**Why it matters:** planning strategy should differ by cell, not by commodity
alone. The forecast-driven models (BTP, BTS) carry inventory risk when the
forecast is wrong; the order-driven models (CTO, BTO) carry responsiveness
risk when lead times are long. This matrix shows which risk dominates each
part of the portfolio.
"""
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    sct_rows = df["SUPPLY_CHAIN_TYPE"].value_counts()
    axes[0].pie(
        sct_rows, labels=sct_rows.index, autopct="%1.1f%%",
        colors=sns.color_palette("Set2"), startangle=90,
    )
    axes[0].set_title("Records by Supply Chain Type")

    sct_qty = (
        df.groupby("SUPPLY_CHAIN_TYPE", observed=True)["MRP_FCST_QTY"]
        .sum()
        .sort_values(ascending=False)
    )
    sct_qty.plot(kind="bar", ax=axes[1], color=sns.color_palette("Set2"))
    axes[1].set_title("Forecasted Units by Supply Chain Type")
    axes[1].set_ylabel("Units")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=0)
    fmt_thousands(axes[1])

    plt.tight_layout()
    render_fig(fig)

    ct = pd.crosstab(
        df["COMMODITY"], df["SUPPLY_CHAIN_TYPE"],
        values=df["MRP_FCST_QTY"], aggfunc="sum",
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        ct, annot=True, fmt=",.0f", cmap="Blues",
        linewidths=0.5, ax=ax, cbar_kws={"label": "Units"},
        annot_kws={"fontsize": 7},
    )
    ax.set_title("Forecasted Units: Commodity × Supply Chain Type")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    render_fig(fig)

    with st.container(border=True):
        st.subheader("Key Insights")
        st.markdown(
            """
**1. This is a CTO-dominant operation - in both effort and volume.**
Configure-to-Order accounts for 58% of forecast records and roughly 8.3M of
the ~11.6M total forecasted units (~72%). Notably, CTO's unit share *exceeds*
its record share - the configure-to-order engine isn't a low-volume
complexity tax on the side; it is the core of the business. Component-level
forecast accuracy is therefore the critical capability: with no
finished-goods buffer, every forecast error in CTO lands directly on
component availability and order lead times.

**2. Order-driven models carry ~90% of volume.** CTO and BTO combined
represent about 83% of records and roughly 90% of forecasted units. This
portfolio holds very little finished-goods inventory risk - the dominant risk
is *responsiveness*: long component lead times (the 13-week tier identified
earlier) against demand that materializes order by order. Supply flexibility,
supplier commitments, and component safety stock matter far more here than
finished-goods stocking policy.

**3. BTS is small but disproportionately efficient to plan.** Build-to-Stock
generates 13% of forecast lines for roughly 1M units (~9% of volume). These
are the standard, predictable products where classic forecast-then-build
logic applies - worth protecting, but not where planning effort should
concentrate.

**4. BTP is marginal - and worth questioning.** At 4% of records and under
200K units (<2% of volume), Build-to-Plan is the smallest flow in the
portfolio. A flow this thin invites a process question: does it represent a
genuine strategic segment (e.g., ramp builds for new platforms), or legacy
classification that could be consolidated into BTS/BTO to simplify planning
governance?

**5. The pie and the bar tell a consistent story - which is itself a
finding.** Record share and unit share track each other closely across all
four types (no model consumes wildly more planning effort than its volume
justifies). The planning workload is well-proportioned to where the business
actually is.
"""
        )


# ------------------------------------------------------------
# SECTION 5 - CHANNEL CONCENTRATION
# ------------------------------------------------------------
def section_channels(df):
    st.header("6 · Channel Concentration: Top 15 Channels")

    st.markdown(
        """
**Business purpose.** Total forecasted units aggregated by demand channel,
showing only the 15 largest channels - the full channel list is typically
long-tailed, and the tail adds noise rather than insight at this level.

**How to read it:** each bar is a channel's total forecasted volume over the
full planning horizon. The channel IDs are internal codes; what matters at
this stage is the *shape* of the distribution rather than the identity of
each code.

**The Pareto check (below the chart):** across all channels in the dataset,
this calculates how many are needed to cover 80% of total forecasted demand.
A small number signals a concentrated demand base - efficient to plan, but
exposed if a key channel shifts. A large number signals fragmented demand -
more resilient, but harder to forecast accurately since each channel's signal
is thinner.

**Why it matters:** channel concentration determines where forecasting effort
pays off. The top channels deserve dedicated review cadence and tighter
collaboration; the long tail is usually better served by statistical
forecasting and aggregate buffers than by manual attention.
"""
    )

    top_channels = (
        df.groupby("CHANNEL_ID")["MRP_FCST_QTY"]
        .sum()
        .nlargest(15)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    top_channels.plot(kind="barh", ax=ax, color="#4a8f5c")
    ax.set_title("Top 15 Channels by Forecasted Demand")
    ax.set_xlabel("Units")
    ax.set_ylabel("Channel ID")
    fmt_thousands(ax, axis="x")
    plt.tight_layout()
    render_fig(fig)

    # Pareto check - computed live and displayed as a metric
    ch_total = df.groupby("CHANNEL_ID")["MRP_FCST_QTY"].sum().sort_values(ascending=False)
    cum_share = ch_total.cumsum() / ch_total.sum()
    n80 = int((cum_share <= 0.80).sum() + 1)

    c1, c2 = st.columns(2)
    c1.metric("Channels covering 80% of demand", f"{n80} of {len(ch_total)}")
    c2.metric("Share of top 2 channels", f"{cum_share.iloc[1]:.1%}")

    with st.container(border=True):
        st.subheader("Key Insights")
        st.markdown(
            f"""
**1. Extreme concentration: two channels dominate everything.** Channel 1865
(~5.5M units) and channel 2190 (~2.9M units) together account for roughly
8.4M of the ~11.6M total forecasted units - about 70–72% of the entire plan
flowing through just two demand channels. This is far beyond a typical 80/20
distribution; it is closer to a two-pillar demand structure with a long
decorative tail.

**2. The drop-off is a cliff, not a slope.** Third place (channel 5000960,
~1.25M units) is less than half of second place, and from rank 4 onward no
channel exceeds ~500K units. The bottom half of even this top-15 view
contributes almost nothing visible at this scale - the tail beyond rank 15 is
operationally negligible in volume terms.

**3. Concentration cuts both ways.** Practically, forecast quality for
channels 1865 and 2190 *is* forecast quality for the company - a
collaborative planning process with the owners of these two demand streams
would cover most of the plan with minimal effort. The flip side is fragility:
any structural shift in either channel (a program ending, a customer segment
moving, a regional realignment) reshapes the entire component plan.
Concentration risk at this level typically deserves explicit contingency
planning, not just acknowledgment.

**4. Interpretation caveat: channel IDs may not be commercial channels.** The
ID structure suggests two families - short legacy codes (1865, 2190, 5364)
and a 5000xxx series - which may represent planning entities, regions, or
fulfillment sites rather than sales channels in the commercial sense. The
concentration finding stands either way, but *what* is concentrated (a
customer base vs. a fulfillment node) changes the business response.
Confirming the channel dimension against a source-system lookup is
recommended before this view drives decisions.

**5. Pareto figure:** **{n80} of {len(ch_total)} channels account for 80% of
forecasted demand** (computed live from the dataset above).
"""
        )


# ------------------------------------------------------------
# SECTION 6 - RISK QUADRANT & WATCHLIST
# ------------------------------------------------------------
def section_risk_quadrant(df):
    st.header("7 · Risk Quadrant: Lead Time vs Forecasted Demand")

    st.markdown(
        """
**Business purpose.** Every point is a single component, positioned by its
procurement lead time (x-axis, weeks) and total forecasted demand across the
planning horizon (y-axis, log scale). Colors indicate commodity. Items with
zero total forecast are excluded. Where an item shows more than one lead time
in the data, the maximum is used - a deliberately conservative choice for
risk assessment.

The dashed red lines mark the 75th percentile on each axis, dividing
components into four risk profiles:

- **Top-right - high demand, long lead time:** the critical quadrant. A
  forecast miss here cannot be recovered quickly, and the volume at stake is
  large. These items warrant safety stock, supplier commitments, or dual
  sourcing.
- **Top-left - high demand, short lead time:** high volume but recoverable;
  replenishment can react within the planning cycle.
- **Bottom-right - low demand, long lead time:** slow movers with long
  pipelines. Individually small, but prone to dead stock and end-of-life
  write-offs.
- **Bottom-left - low demand, short lead time:** minimal planning attention
  required.

**Method note:** thresholds are relative (75th percentile), so the quadrants
always highlight the riskiest quarter of the portfolio rather than relying on
fixed cutoffs that go stale as the business changes.
"""
    )

    item_view = (
        df.groupby("COMP_ITEM_ID")
        .agg(
            lead_time=("LEAD_TIME", "max"),
            total_fcst=("MRP_FCST_QTY", "sum"),
            commodity=("COMMODITY", "first"),
        )
        .reset_index()
    )
    item_view = item_view[item_view["total_fcst"] > 0]  # drop dead rows

    lt_thr = float(item_view["lead_time"].quantile(0.75))
    qty_thr = float(item_view["total_fcst"].quantile(0.75))

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.scatterplot(
        data=item_view, x="lead_time", y="total_fcst",
        hue="commodity", alpha=0.6, s=35, ax=ax,
    )
    ax.set_yscale("log")
    ax.axvline(lt_thr, color="red", ls="--", lw=1)
    ax.axhline(qty_thr, color="red", ls="--", lw=1)
    ax.set_title("Risk Quadrant: Lead Time vs Total Forecasted Demand (log scale)")
    ax.set_xlabel("Lead Time (weeks)")
    ax.set_ylabel("Total Forecast Qty (log)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    render_fig(fig)

    st.caption(
        f"Quadrant thresholds (75th percentile): lead time ≥ **{lt_thr:.0f} weeks** · "
        f"total forecast ≥ **{qty_thr:,.0f} units**"
    )

    # --- The watchlist: high demand AND long lead time ---
    watchlist = (
        item_view[
            (item_view["lead_time"] >= lt_thr)
            & (item_view["total_fcst"] >= qty_thr)
        ]
        .sort_values("total_fcst", ascending=False)
        .rename(columns={
            "COMP_ITEM_ID": "Component",
            "lead_time": "Lead Time (wks)",
            "total_fcst": "Total Forecast (units)",
            "commodity": "Commodity",
        })
    )

    st.subheader(f"⚠️ Watchlist - {len(watchlist)} items with high lead time AND high demand")
    st.markdown(
        "The top-right quadrant extracted as a ranked table - the specific part "
        "numbers that combine high volume with slow replenishment. This is the "
        "shortlist a planner would bring to a supplier review."
    )
    st.dataframe(watchlist, use_container_width=True, height=400)

    st.download_button(
        label="Download full watchlist (CSV)",
        data=watchlist.to_csv(index=False).encode("utf-8"),
        file_name="risk_watchlist.csv",
        mime="text/csv",
    )

    with st.container(border=True):
        st.subheader("Key Insights")
        st.markdown(
            """
**1. The vertical stripes confirm it: lead time is a tiered parameter.**
Components line up at fixed lead-time values (8, 10, 13, 15, 17–19, 22, 24
weeks) rather than spreading continuously - item-level confirmation of the
finding from the commodity view. The risk analysis is therefore about *which
tier an item was assigned to*, not measured supplier behavior.

**2. The danger zone is real and populated.** The top-right quadrant
(≥17 weeks lead time, above the demand threshold) contains a meaningful
cluster of components - including items in the 10⁴–10⁵ unit range at 17–19
weeks (visible among LCD, Keyboard, Graphic Card, and Speaker families, plus
Nic Card items out at 22 weeks). These are parts where four to five months of
pipeline sit behind substantial volume: a forecast miss discovered today was
baked in last quarter. The watchlist above names them individually.

**3. The bulk of high-volume items sit at 13 weeks or less - moderate, not
safe.** The densest high-demand columns are at 8, 10, and 13 weeks. Thirteen
weeks is still a full quarter of exposure; the portfolio's center of gravity
is "recoverable with effort," not "reactive." Only the 8-week columns
(Wireless, HDD, SSD, Docking families) offer genuinely fast correction
cycles.

**4. Long lead time correlates with low volume - mostly.** The 22- and
24-week columns are sparsely populated and skew low-demand (Controller Card,
Mobile Computing Cart): slow parts the business has largely kept small. The
exceptions - high-volume items stranded in slow tiers - are precisely what
the quadrant isolates, and they are few enough to manage individually rather
than by policy.

**5. Within-column spread is the hidden story.** Items at the *same* lead
time span five orders of magnitude in demand (from single units to 10⁵–10⁶).
Uniform planning rules per lead-time tier would therefore be badly
miscalibrated: the top of each column merits item-level attention and
supplier commitments, while the bottom of the same column is candidate
material for catalog cleanup - components forecasted at one to ten units
across an entire year, which cost more to plan than they move.
"""
        )


# ------------------------------------------------------------
# SECTION 7 - ZERO-FORECAST CONCENTRATION (DATA QUALITY)
# ------------------------------------------------------------
def section_zero_forecast(df):
    st.header("8 · Data Quality Check: Zero-Forecast Concentration")

    st.markdown(
        """
**Business purpose.** Before drawing conclusions from the demand charts, this
view measures how much of each commodity's data actually carries a signal.
Each bar shows the percentage of forecast records where the planned quantity
is exactly zero.

Zero rows are common in MRP extracts and are not errors - they appear when a
component is set up in the planning system but has no demand in a given
week/channel/version combination (phase-in items not yet ramped, phase-out
items winding down, or configurations kept active for service coverage).

**How to read it:** a commodity with a high zero-rate has its demand
concentrated in a small subset of its records. Its averages and trends are
driven by that active subset, so apparent volatility may reflect sparse data
rather than volatile demand. Commodities with low zero rates have broad,
consistent demand across their records - their charts can be read with more
confidence.

**Why it matters:** this is the reliability legend for the rest of the
dashboard. It tells the reader which commodity-level findings rest on dense
data and which should be treated as directional. It also flags potential
catalog hygiene issues - a commodity where most records are zero may be
carrying many inactive part numbers that inflate planning workload without
contributing volume.
"""
    )

    zero_by_comm = (
        df.groupby("COMMODITY", observed=True)["ZERO_FCST"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    (zero_by_comm * 100).plot(kind="bar", ax=ax, color="#8a6fb2")
    ax.set_title("% of Forecast Records That Are Zero, by Commodity")
    ax.set_ylabel("% zero rows")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    for i, v in enumerate(zero_by_comm * 100):
        ax.text(i, v + 0.8, f"{v:.0f}%", ha="center", fontsize=7)
    plt.tight_layout()
    render_fig(fig)

    with st.container(border=True):
        st.subheader("Key Insights")
        st.markdown(
            """
**1. Zero rows are the norm, not the exception.** Every commodity in the
portfolio carries a substantial share of zero-quantity forecast records -
from ~16% (Security Lock) up to ~67% (Speaker), with most categories sitting
between 40% and 60%. Roughly half of the typical commodity's forecast lines
carry no demand signal at all. This is structural to how MRP explodes
forecasts across product, channel, and week combinations - but it means every
average and trend in this dashboard is driven by the active subset of
records.

**2. The noisiest commodities are the highest-volume ones.** Speaker (67%),
MOUSE (~59%), Memory (~59%), ODD Mechanical (~59%), and ADPT (~59%) top the
zero-rate ranking - and three of those are also among the biggest demand
drivers in the portfolio. This is the opposite of the intuitive guess. The
mechanism: universal, high-attach components are set up across nearly every
product and channel combination, so they accumulate enormous numbers of
placeholder lines, most of which are zero in any given week. High volume
concentrated in a minority of active lines.

**3. The cleanest data belongs to the niche items.** Security Lock (~16%),
Projector (~22%), and Cable (~26%) show the lowest zero rates - narrow-purpose
components set up only where genuine demand exists. Their charts can be read
at close to face value, while conclusions about the 55%+ group should be
validated against active records only.

**4. Planning workload implication: the system is maintaining a large volume
of empty lines.** If roughly half of 106K forecast records are zero, planners
and systems are processing tens of thousands of lines per cycle that carry no
signal. Two practical responses: (a) filter analytical views to active
records to sharpen signal, and (b) review whether long-zero component/channel
combinations can be deactivated in the planning master - catalog hygiene that
reduces noise at the source.

**5. Reliability guide for this dashboard.** As a rule of thumb applied to
the preceding sections: findings about Cable, Projector, Network Card, and
Security Lock rest on dense data; findings about Speaker, MOUSE, Memory, and
the mechanical categories describe real volume but sparse-and-spiky record
patterns - their week-level movements deserve more smoothing and less literal
reading.
"""
        )


# ------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------
def main():
    # --- Load data (with a friendly error if the CSV is missing) ---
    try:
        df_raw, df = load_data("inventory.csv")
    except FileNotFoundError:
        st.error(
            "**inventory.csv not found.** Place the CSV file in the same folder "
            "as this script, then refresh the page."
        )
        st.stop()
    except Exception as e:
        st.error(f"Could not load inventory.csv: {e}")
        st.stop()

    # --- Sidebar: branding + navigation ---
    with st.sidebar:
        st.title("📦 Material Requirements Planning Forecast Analytics")
        st.caption("Component demand, lead-time exposure, and supply risk")
        st.divider()

        sections = {
            "1 · Overview & Data Preparation": section_overview,
            "2 · Demand by Commodity": section_demand_by_commodity,
            "3 · Lead Time Analysis": section_lead_time,
            "4 · Demand Over Time": section_demand_over_time,
            "5 · Supply Chain Type Mix": section_supply_chain_mix,
            "6 · Channel Concentration": section_channels,
            "7 · Risk Quadrant & Watchlist": section_risk_quadrant,
            "8 · Zero-Forecast Data Quality": section_zero_forecast,
        }
        choice = st.radio("Analysis sections", list(sections.keys()), index=0)

        st.divider()
        st.caption(
            f"**{len(df):,}** forecast records · "
            f"**{df['COMP_ITEM_ID'].nunique():,}** components · "
            f"**{df['COMMODITY'].nunique()}** commodities"
        )
        st.caption("Built with Streamlit · Humberto Hernandez R.")

    # --- Header shown on every page ---
    st.title("MRP Forecast Analytics Dashboard")
    st.markdown(
        """
An end-to-end analysis of an MRP component forecast extract (~106K records):
demand concentration by commodity and channel, lead-time exposure,
fulfillment-model mix (CTO/BTO/BTP/BTS), component-level supply risk, and the
data-quality checks that qualify every finding. Use the sidebar to navigate
through the analysis in its intended order - each section states its business
purpose, presents the evidence, and closes with actionable insights.
"""
    )
    st.divider()

    # --- Render the selected section ---
    if choice == "1 · Overview & Data Preparation":
        section_overview(df_raw, df)
    else:
        sections[choice](df)


if __name__ == "__main__":
    main()