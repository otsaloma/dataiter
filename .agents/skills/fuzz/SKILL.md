---
name: fuzz
description: End-to-end fuzzing of dataiter — work independently 30–60
  minutes doing realistic data analysis on a new 1–10M row dataset,
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
ISSUES.md and present them to me, and we'll see if there's something in
dataiter that we should fix.

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

Previously you have used the following datasets. Try something different
this time. Find preferably something with 1–10M rows. Focus primarily on
tabular data and the `DataFrame` class.

- Backblaze hard drive stats (daily SMART snapshots)
- BLS QCEW employment and wages (annual and quarterly single files, area, industry, ownership and aggregation level titles)
- Chicago crimes
- Citi Bike GBFS station feeds
- Citi Bike trips
- Complete Journey retail purchases, promotions, products, household demographics, campaigns and coupons (completejourney R data)
- EPA AQS air quality (hourly ozone, daily PM2.5, daily AQI by county, sites, monitors)
- Eurostat weekly deaths by NUTS region, regional population and GEO codelist
- FEC campaign finance bulk data (individual and committee contributions, committees, candidates)
- Helsinki city bikes
- IMDb non-commercial datasets
- Lichess puzzle database
- MBTA GTFS schedule
- MovieLens 25M
- MTA subway hourly ridership (data.ny.gov) and subway stations
- NOAA Storm Events details, fatalities and locations
- NYC 311 service requests
- NYC Citywide Payroll
- NYC DOF parking violations issued (fiscal year), parking violation codes and Open Parking and Camera Violations (payments)
- NYC taxi
- Ookla open data fixed broadband performance tiles
- Open-Meteo hourly weather
- OpenPowerlifting
- Seattle Public Library checkouts by title (monthly), physical item checkouts (events) and ILS data dictionary
- SF Fire Department calls for service and fire incidents (data.sf.gov)
- UK DfT road safety (STATS19) collisions, vehicles and casualties
- UK Land Registry price paid
- US Census ACS PUMS
- US Census ZCTA gazetteer
- US flights
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
