# Member 1 report writing handoff

Completed in report/:

- Section 2.2: exact SGD/global-DoG updates, initialization/update order, theoretical scope, averaging, state cost and implementation checks.
- Section 4.1: formal quality, seed variability, target attainment, runtime/search cost, quadratic conditioning/scaling evidence and limitations.
- Section 3.3: frozen common protocol and Member 1 search/settings; other members' optimizer rows remain explicitly pending.
- Member 1 synthesis/contribution rows and Introduction scope updated; other members' data/results were not invented.
- Tables generated from main_results.csv and diagnostic_results.csv; formal aggregates independently checked against 18 raw summaries, diagnostic relative gaps against 30 raw summaries.
- Two compact report figures generated from saved logs; original overview/full diagnostic plots retained.
- Reference implementation citation added with version/commit, distinct from the DoG paper.

Rebuild:

```sh
.venv/bin/python scripts/build_member1_report_assets.py
sh report/build_report.sh
```

The generator changes report assets only, not training evidence. Numerical audit:
report/generated/member1_numeric_audit.json. The report compiled successfully to 13
pages with no undefined references/citations or overfull boxes. A contact sheet of the
whole report and enlarged Member 1 pages were reviewed; previews stay in report/build/qa/.

The whole report is still a group draft: other members' sections/tables, author
identities, Group ID, repository URL and final abstract/conclusions still need their
owners' completion. No experiments or frozen training sources were changed by writing.
