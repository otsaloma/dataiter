---
name: fuzz
description: End-to-end fuzzing of dataiter — work independently 30–60
  minutes doing realistic data analysis on a new 50–100M row dataset,
  record API surprises, broken functionality, complicated or slow
  operations into ISSUES.md, then work through them one by one. Use
  when asked to fuzz dataiter or repeat the end-to-end exercise.
---

Work independently for 30–60 minutes doing data manipulation, data
science and data engineering with dataiter, including joins, filtering
etc. Like fuzzing, but end-to-end: do realistic full tasks, such as
calculating dashboard-type figures from raw data, not isolated little
operations. Download some suitable data from the internet. Keep a record
of any API surprises, broken functionality, complicated operations and
slow operations. Then write your findings as numbered items into
`ISSUES.md` and present them to me, and we'll see if there's something
in dataiter that we should fix.

Regarding API surprises, note that Pandas is a pile of shit and our goal
is not to clone Pandas. Compare broadly across data frame
implementations in Python and R: Pandas, Polars, Arrow, dplyr. Is there
a consensus on how certain operations are done? Missing a consensus,
favor dplyr most as reference, it's the best designed API among those.

Regarding slow operations, note that we don't want to optimize
performance for just the single use-case of your current data. Look for
a change that speeds up an operation for your current data, but then
test other kind of data as well to make sure your suggestion would not
introduce regressions elsewhere. For example, if you find a join slow
for wide data with multiple integer join keys, test your suggestion also
for long data, string keys, datetime keys, single keys etc.

Start by reading `doc/_build/html/llms-full.txt` in full and work based
on that, since that's what we recommend users point their coding agents
at. If it doesn't exist, build the documentation first with `uv run make
doc`. Do not read the dataiter source code before writing and running
your analysis code! Real users will work based on the API documentation
and we want to make sure that alone enables writing code that works
right. Once you find an actual problem, such as slow performance, then
read the source code to understand what happens and to help you draft a
suggestion to fix the problem.

If you want to place your working files under this repo, put them under
`tmp/fuzz/{dataset}`. `tmp` is gitignored and persistent across our
sessions until eventually wiped by `make clean`.

Previously you have used the following datasets. Try something different
this time. Find preferably something with 50–100M rows. Focus primarily on
tabular data and the `DataFrame` class.

- Amazon Reviews 2023 (Electronics and Cell Phones ratings, metadata)
- Backblaze hard drive stats (daily SMART snapshots)
- Binance spot aggregated trades and 1-minute klines (October 2025)
- BLS QCEW employment and wages (annual and quarterly files, titles)
- CFPB HMDA loan/application register (Washington 2018–2023) and filers
- Chicago crimes
- CMS Medicare Physician & Other Practitioners
- Citi Bike GBFS station feeds
- Citi Bike trips
- Complete Journey retail purchases, products, households, campaigns
- EIA-930 hourly electricity balance and balancing-authority interchange
- EPA AQS air quality (hourly ozone, daily PM2.5 and AQI, monitors)
- Eurostat weekly deaths by NUTS region, population and GEO codelist
- FEC campaign finance bulk data (contributions, committees, candidates)
- GDELT daily event exports (2024-01-01 through 2025-03-01)
- GHCN-Daily weather (1900 and 1920 year files, stations, inventory)
- GH Archive GitHub events (2024-01-08 through 2024-01-21, hourly files)
- Helsinki city bikes
- IMDb non-commercial datasets
- Inside Airbnb listings, calendar and reviews (Amsterdam, Berlin, etc.)
- Instacart Online Grocery 2017 (orders, baskets, products, departments)
- JHU CSSE COVID-19 time series (US county and global, UID lookup table)
- Lichess puzzle database
- MBTA GTFS schedule
- mledoze/countries (country names, regions and currencies)
- MovieLens 25M
- MTA subway hourly ridership (data.ny.gov) and subway stations
- nflverse NFL play-by-play, games, players, rosters and participation
- NHTSA vehicle safety complaints, recalls and defect investigations
- NOAA Storm Events details, fatalities and locations
- NYC 311 service requests
- NYC Citywide Payroll
- NYC DOF parking violations, violation codes and camera violations
- NYC motor vehicle collisions (crashes, vehicles, person sample)
- NYC taxi
- Ookla open data fixed broadband performance tiles
- Open-Meteo hourly weather
- OpenPowerlifting
- OpenSky Network flight lists (April 2019 and 2020) and aircraft
- opentransportdata.swiss Ist-Daten v2 (September 2026), service points
- OurAirports airports and countries
- PUDL EPA CEMS hourly emissions (2022–2023), EIA plants, generators, EPA–EIA crosswalk
- Seattle Public Library checkouts (monthly and events), data dictionary
- SF Fire Department calls for service and fire incidents (data.sf.gov)
- SSA baby names (national and state counts, card-holder totals)
- Stack Overflow data dump via ClickHouse S3
- Stanford Open Policing traffic stops (Nashville, San Diego, etc.)
- UCI Gas Sensor Array Temperature Modulation (13 measurement runs)
- UCI Individual Household Electric Power Consumption (2006–2010)
- UCI Online Retail II (two-year retail transactions)
- UCI PAMAP2 Physical Activity Monitoring
- UK DfT road safety (STATS19) collisions, vehicles and casualties
- UK DVSA anonymised MOT tests (2022 results, failure items, lookups)
- UK Land Registry price paid
- US Census ACS PUMS
- US Census LEHD LODES (California commuting flows, jobs and crosswalk)
- US Census national county codes (2020)
- US Census ZCTA gazetteer
- US flights
- USGS earthquake catalog (2016–2025 events, M6+ GeoJSON, contributors)
- Wikipedia clickstream
- World Bank country population (datasets/population on GitHub)

When done, add the datasets you used to the above list in this file
(`.agents/skills/fuzz/SKILL.md`).

## Working Through the Findings

Re-read `ISSUES.md` first, I might have edited it. Then go through the
items in order, one at a time: (1) Present the issue and your
suggestion, or options with a recommendation, and wait for my decision.
(2) Implement it, including tests and a `NEWS.md` entry if user-visible.
(3) Present the result with a proposed commit message and commit once I
approve. (4) Remove the item from `ISSUES.md`, but keep the numbering.
Add any new issues found along the way as new items instead of expanding
the current one. Finally, remove `ISSUES.md` once empty and commit this
file if not yet committed.
