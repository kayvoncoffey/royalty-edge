# royalty_edge

Quantitative pricing of music and IP royalty catalogs listed on Royalty Exchange.
Phase 1: acquisition and storage. Phase 2: clearing-multiple regression.
Phase 3: decay model. Phase 4: bid engine.

## Status

Parsers are written and pinned against live payloads captured 2026-08-11.
`pytest` passes. No harvest has been run yet.

## API (confirmed by recon)

Index, 15 rows per page, 2,528 listings across 169 pages:

```
GET https://auctions.royaltyexchange.com/inventory/listings/
    ?filter{state.in}=filled&filter{state.in}=pending&filter{state.in}=closed
    &only_favorited=0&sort[]=-deal_date&sort[]=-published_date
    &page=N&page_size=15
```

The response carries its own `next` cursor. Follow it rather than incrementing
`page`, so a mid-crawl page-size change cannot silently skip listings.

Detail payloads carry the offer ladder, a complete quarterly earnings panel,
and four annual breakdowns. Detail is where all the value is; the index is
just the spine.

## Layout

```
src/royalty_edge/
  fetch/client.py      polite, content-addressed, resumable HTTP client
  parse/listing.py     index + detail adapters, term taxonomy, reconciliation
  parse/dollar_age.py  Dollar Age recomputation, sync normalization
  db/schema.sql        three-layer DuckDB schema
  db/load.py           L1 writes, L2 rebuild, quality flags
fixtures/              captured payloads; the API contract
tests/                 regression tests pinned to fixtures
```

## Run order

```bash
pip install -e ".[dev]"
pytest
python -m royalty_edge.cli discover      # walk the index, fill fetch_queue
python -m royalty_edge.cli harvest       # payloads to landing/, no DB lock
python -m royalty_edge.cli load          # payloads -> L1 -> L2 rebuild
```

Harvest and load are separate because DuckDB takes an exclusive file lock; a
long scrape holding it would block every notebook you have open.

## Invariants the loader enforces

- The quarterly panel must sum to `lifetime_amount`, and its last four
  quarters to `ltm`. A listing that fails this does not enter the decay sample.
- `fixed_return` listings are flagged `exclude`. They are contracted advance
  payments, not decaying royalty streams, and their "multiple" is not
  comparable to a perpetuity multiple.
- Unsold listings stay in `dim_listing` with `sold = FALSE`. They are the
  censoring information that makes the clearing-price regression interpretable.
- Multiples are recomputed from price and LTM in one place, gross and
  fee-loaded. The platform's displayed multiple is stored, never trusted.

## Open empirical questions

- `dollar_age` has a median of 9.4 in the 30-row recon sample but a maximum of
  145, which cannot be a weighted average song age. Either the construction
  differs by listing type or some catalogs are genuinely ancient.
  `v_dollar_age_check` decides it once the full pull lands.
- `is_open_auction` is `False` on all 30 recon rows, yet listing 6792 has a
  38-bid ascending ladder with proxy increments. The flag means something
  other than "this listing had an auction". Resolve before using it to split
  price-formation mechanisms.
- `marketplace_median_multiplier` is published to bidders before they bid, and
  is disabled on a minority of listings. Those disabled listings are a natural
  experiment on anchoring.
