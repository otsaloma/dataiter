# AGENTS.md

## API Mirroring

The `DataFrame` and `ListOfDicts` classes have a mostly matching API.
When adding a method or an argument to one, also add it to the other if
it makes sense there as well.

## Validation, Testing

After making changes, run `direnv exec . make check` and `direnv exec .
make test` (or `direnv exec . pytest ...` to test only a subset). direnv
ensures these use the virtualenv present in `venv`.

## API Documentation

The API documentation currently requires new methods, properties and
attributes to be manually added to the index at the top of `doc/*.rst`
files. Run `direnv exec . make doc-check` to check these are up to date.
