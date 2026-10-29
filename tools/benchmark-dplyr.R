#!/usr/bin/env Rscript
# Benchmark dplyr equivalents of the data frame benchmarks in tools/benchmark.py.
# Usage: Rscript tools/benchmark-dplyr.R [-o OUTPUT] [PATTERN...]

suppressPackageStartupMessages({
    library(dplyr)
    library(nanoparquet)
    library(readr)
    library(tidyr)
    library(uuid)
})

script = sub("--file=", "", grep("--file=", commandArgs(), value=TRUE))
PATH = file.path(dirname(script), "..", "data", "vehicles.csv")
set.seed(0)
VEHICLES = read_csv(PATH, show_col_types=FALSE, guess_max=Inf)

CACHE = list()

data_frame = function(nrow=1000000) {
    key = as.character(nrow)
    if (is.null(CACHE[[key]])) {
        n = nrow %/% nrow(VEHICLES)
        data = head(bind_rows(rep(list(VEHICLES), n + 1)), nrow)
        CACHE[[key]] <<- data[sample(nrow(data)), ]
    }
    CACHE[[key]]
}

RANDOM = tibble(g=sample(0:99999, 1000000, replace=TRUE), a=rnorm(1000000, 10, 2))

B = list()

B$data_frame_aggregate_128 = function() {
    data = data_frame()
    start = Sys.time()
    data |>
        group_by(make) |>
        summarise(n=n(), hwy=mean(hwy), cty=mean(cty))
    Sys.time() - start
}

B$data_frame_aggregate_3264 = function() {
    data = data_frame()
    start = Sys.time()
    data |>
        group_by(make, model) |>
        summarise(n=n(), hwy=mean(hwy), cty=mean(cty), .groups="drop")
    Sys.time() - start
}

B$data_frame_aggregate_14668 = function() {
    data = data_frame()
    start = Sys.time()
    data |>
        group_by(make, model, year) |>
        summarise(n=n(), hwy=mean(hwy), cty=mean(cty), .groups="drop")
    Sys.time() - start
}

# dplyr evaluates R expressions per group, so there's no separate fast path.
B$data_frame_aggregate_100000_lambda = function() {
    start = Sys.time()
    RANDOM |>
        group_by(g) |>
        summarise(a_mean=mean(a), a_std=sd(a))
    Sys.time() - start
}

B$data_frame_aggregate_100000_short = B$data_frame_aggregate_100000_lambda

B$data_frame_filter_is_in_1000 = function() {
    data = data_frame()
    models = head(unique(data$model), 1000)
    start = Sys.time()
    data |> filter(model %in% models)
    Sys.time() - start
}

B$data_frame_full_join = function() {
    data = data_frame()
    meta = data |> distinct(make, model)
    meta = bind_rows(meta, mutate(meta, model="X"))
    meta$random = runif(nrow(meta))
    start = Sys.time()
    data |> full_join(meta, by=c("make", "model"), relationship="many-to-many")
    Sys.time() - start
}

left_join_by = function(by) {
    data = data_frame()
    meta = data |> distinct(across(all_of(by)))
    meta$random = runif(nrow(meta))
    start = Sys.time()
    data |> left_join(meta, by=by)
    Sys.time() - start
}

B$data_frame_left_join = function() left_join_by(c("make", "model"))
B$data_frame_left_join_int = function() left_join_by("id")
B$data_frame_left_join_mixed = function() left_join_by(c("make", "year"))

B$data_frame_left_join_to_larger = function() {
    meta = data_frame()
    meta$uuid = UUIDgenerate(n=nrow(meta))
    data = meta |>
        select(uuid) |>
        head(300)
    start = Sys.time()
    data |> left_join(meta, by="uuid")
    Sys.time() - start
}

B$data_frame_modify_grouped = function() {
    data = data_frame()
    start = Sys.time()
    data |>
        group_by(make, model) |>
        mutate(hwy_share=hwy / sum(hwy)) |>
        ungroup()
    Sys.time() - start
}

B$data_frame_pivot_longer = function() {
    data = data_frame() |>
        distinct(make, model, year, .keep_all=TRUE) |>
        pivot_wider(id_cols=c(model, year), names_from=make, values_from=hwy)
    start = Sys.time()
    data |> pivot_longer(!c(model, year), names_to="make", values_to="hwy")
    Sys.time() - start
}

B$data_frame_pivot_wider = function() {
    data = data_frame() |> distinct(make, model, year, .keep_all=TRUE)
    start = Sys.time()
    data |> pivot_wider(id_cols=c(model, year), names_from=make, values_from=hwy)
    Sys.time() - start
}

B$data_frame_rbind_2 = function() {
    data = data_frame(500000)
    start = Sys.time()
    bind_rows(rep(list(data), 2))
    Sys.time() - start
}

B$data_frame_rbind_100 = function() {
    data = data_frame(10000)
    start = Sys.time()
    bind_rows(rep(list(data), 100))
    Sys.time() - start
}

B$data_frame_rbind_100000 = function() {
    data = data_frame(10)
    start = Sys.time()
    bind_rows(rep(list(data), 100000))
    Sys.time() - start
}

B$data_frame_read_csv = function() {
    start = Sys.time()
    read_csv(PATH, show_col_types=FALSE, guess_max=Inf)
    Sys.time() - start
}

B$data_frame_read_parquet = function() {
    path = tempfile(fileext=".parquet")
    on.exit(unlink(path))
    write_parquet(data_frame(), path)
    start = Sys.time()
    read_parquet(path)
    Sys.time() - start
}

B$data_frame_sort = function() {
    data = data_frame()
    start = Sys.time()
    data |> arrange(make, model, year)
    Sys.time() - start
}

B$data_frame_sort_categorical = function() {
    data = data_frame()
    start = Sys.time()
    data |> arrange(make)
    Sys.time() - start
}

B$data_frame_sort_unique = function() {
    data = data_frame()
    data$uuid = UUIDgenerate(n=nrow(data))
    start = Sys.time()
    data |> arrange(uuid)
    Sys.time() - start
}

B$data_frame_unique = function() {
    data = data_frame()
    start = Sys.time()
    data |> distinct(make, model, year, .keep_all=TRUE)
    Sys.time() - start
}

B$data_frame_write_csv = function() {
    data = data_frame()
    path = tempfile(fileext=".csv")
    on.exit(unlink(path))
    start = Sys.time()
    write_csv(data, path)
    Sys.time() - start
}

B$data_frame_write_parquet = function() {
    data = data_frame()
    path = tempfile(fileext=".parquet")
    on.exit(unlink(path))
    start = Sys.time()
    write_parquet(data, path)
    Sys.time() - start
}

args = commandArgs(trailingOnly=TRUE)
output = NULL
if (length(args) >= 2 && args[1] %in% c("-o", "--output")) {
    output = args[2]
    args = args[-(1:2)]
}
patterns = if (length(args) > 0) args else "_"
benchmarks = Filter(function(x) any(sapply(patterns, grepl, x, fixed=TRUE)), names(B))
keys = sapply(strsplit(benchmarks, "_"), function(x) {
    paste(ifelse(grepl("^[0-9]+$", x), paste0(strrep("0", 9 - nchar(x)), x), x), collapse="_")
})
benchmarks = benchmarks[order(keys, method="radix")]
width = max(nchar(benchmarks)) + 2
results = list()
for (i in seq_along(benchmarks)) {
    benchmark = benchmarks[i]
    dots = strrep(".", width - nchar(benchmark) - 1)
    cat(sprintf("%2d/%d. %s %s ", i, length(benchmarks), benchmark, dots))
    elapsed = 1000 * min(sapply(1:5, function(i) as.numeric(B[[benchmark]](), units="secs")))
    cat(sprintf("%5.0f ms\n", elapsed))
    results[[i]] = tibble(name=benchmark, elapsed=round(elapsed))
}
if (!is.null(output)) {
    cat(sprintf("Writing %s...\n", output))
    results = bind_rows(results) |> mutate(version=paste("dplyr", packageVersion("dplyr")))
    write_csv(results, output)
}
