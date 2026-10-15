---
name: fuzz
description: End-to-end fuzzing of dataiter — work independently 30–60
  minutes doing realistic data analysis on a new 1–10M row dataset,
  record API surprises, broken functionality, complicated or slow
  operations into ISSUES.md and present the findings. Use when asked to
  fuzz dataiter or repeat the end-to-end exercise.
---

Work independently for 30–60 minutes doing data manipulation, data
science and data engineering with dataiter, including joins, filtering
etc. Like fuzzing, but end-to-end: do realistic full tasks, such as
calculating dashboard-type figures from raw data, not isolated little
operations. Use files under data or download more from the internet.
Keep a record of any API surprises, broken functionality, complicated
operations and slow operations. Then write your findings as numbered
items into ISSUES.md and present them to me, and we'll see if there's
something in dataiter that we should fix.

Start by reading `doc/_build/html/llms-full.txt` and work based on that,
since that's what we recommend users point their coding agents at. If it
doesn't exist, build the documentation first with `uv run make doc`.
Then read the source code too as you see fit.

Previously you have used the following datasets. Try something different
this time. Find preferably something with 1–10M rows. Focus primarily on
tabular data and the `DataFrame` class.

- Citi Bike GBFS station feeds
- Citi Bike trips
- Helsinki city bikes
- NYC taxi
- Open-Meteo hourly weather
- UK Land Registry price paid
- US flights

When done, add the datasets you used to the above list in this file
(`.agents/skills/fuzz/SKILL.md`).
