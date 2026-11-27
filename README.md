# COMP6704 optimizer study

Four paired studies of unconstrained neural-network optimization on CPU Macs:
SGD/DoG, Momentum/Schedule-Free SGD, RMSProp/ADOPT with beta1=0, Adam/AdEMAMix.
The tasks are Fashion-MNIST (MLP), CIFAR-10 subset (CNN), Tiny Shakespeare (character GRU).

## What is ready

- Member ownership, report draft/template, paper references and individual prompts.
- A frozen shared protocol, experiment phases and output schema.
- Member 4's complete Adam/AdEMAMix implementation, including shared data, model and training modules.
- A runnable plan/dispatch command and portable ZIP packager.

Member 2's final Momentum/Schedule-Free reproduction is available in
[studies/member2](studies/member2/README.md), with frozen settings, 45 measured run records,
and figure/table export. Member 3's completed RMSProp/ADOPT-0 study is in
[members/member3](members/member3/README.md), with implementation checks, frozen configurations,
measured logs, figures, and report source. Member 4 runs through the shared dispatcher.
Member 1's completed SGD/DoG study is available in
[members/member1](members/member1/README.md), including independent optimizers,
frozen configurations, 18 formal runs, quadratic experiments, and reproducible
tables and figures.

## Member 1 code and measured results

Read [Member 1 reproduction instructions](members/member1/README.md).
Formal results are in
[main_results.csv](members/member1/results/member1/tables/main_results.csv).

Install Member 1's recorded dependencies in a Python environment:

```sh
python -m pip install -r members/member1/requirements-lock.txt
```

Run from the repository root:

```sh
python scripts/run_member.py --member 1 --phase check
python scripts/run_member.py --member 1 --phase main
python scripts/run_member.py --member 1 --phase export
```

The main command recognizes the supplied completed batch. Export rebuilds
its tables and figures from saved logs.

## Member 3 code and measured results

Read [Member 3 reproduction instructions](members/member3/README.md),
[results](members/member3/results/member3/tables/), and
[report integration notes](members/member3/docs/MEMBER3_HANDOFF.md).
Use its own runner from `members/member3/`; the root Member 3 dispatcher remains a scaffold.
The module preserves its recorded 300-call protocol and does not replace Member 4 shared settings.

## Start

```sh
python3 scripts/run_member.py --member 1 --phase plan
python3 scripts/run_member.py --member 4 --phase check
```

Read [team assignments](docs/分工3.0.md), [experiment contracts](docs/EXPERIMENT_CONTRACT.md), and your member prompt:

- [Member 1](docs/prompts/member1_prompt.md): SGD / DoG
- [Member 2](docs/prompts/member2_prompt.md): Momentum / Schedule-Free SGD
- [Member 3](docs/prompts/member3_prompt.md): RMSProp / momentum-free ADOPT
- [Member 4](docs/prompts/member4_prompt.md): Adam / AdEMAMix

[Report template PDF](report/main.pdf) · [Branch workflow](docs/BRANCHING.md)
Each member implements and checks their phase handler before running the commands in their prompt.
Proposed dependencies are in requirements.in; after a real environment is tested, record exact versions
in requirements-lock.txt and each run's environment.json. A GPU is not required.

## Layout

```text
comp6704-group_project/
  README.md
  members/member1/           completed SGD/DoG study and experiment evidence
  members/member3/           completed standalone RMSProp/ADOPT-0 study
  studies/member2/           completed Momentum/Schedule-Free reproduction
  requirements.in
  configs/shared_protocol.json
  configs/member1.json ... member4.json
  src/common/                 shared data, model and training implementations
  src/members/member1.py ... member4.py
  scripts/run_member.py       planner and phase dispatcher
  scripts/package_project.py ZIP builder
  data/raw/                  ignored; no datasets in release ZIP
  data/manifests/             split indices and checksums to generate
  results/member1/ ... member4/
    raw/ tables/ figures/ notes/
  docs/分工3.0.md
  docs/COURSE_REQUIREMENTS.md
  docs/EXPERIMENT_CONTRACT.md
  docs/BRANCHING.md
  docs/prompts/member1_prompt.md ... member4_prompt.md
  report/                    report source and compiled template
    main.tex references.bib neurips_2025.sty sections/ figures/
    build_report.sh .latexmkrc main.pdf
```

All commands below run from the repository root. The editable report is in `report/`,
and each member's handoff is in `docs/prompts/`. Course requirements are summarized in
`docs/COURSE_REQUIREMENTS.md`; the original course handout is not publicly distributed.

## Build and package

Build: `sh report/build_report.sh`.
LaTeX auxiliaries stay under `report/build/`; `report/main.pdf` is the final copy.

```sh
python3 scripts/package_project.py
```

Default output is dist/comp6704-group_project_scaffold.zip. Use --output for another destination.
The packager includes report source/PDF, configs, measured logs if later present, tables, figures,
docs and code; excludes raw datasets, model checkpoints, private environment files, .git and build.
It records SHA-256 hashes in ARCHIVE_MANIFEST.json and checks archive integrity.
It creates a local archive only, never a remote repository or commit.
Before a final group release, integrate the remaining member submissions and the final group report,
and update their tested reproduction instructions. Member 3 has its own verified ZIP builder;
see its README for the module-specific command.
