# COMP6704 optimizer study

Four paired studies of unconstrained neural-network optimization on CPU Macs:
SGD/DoG, Momentum/Schedule-Free SGD, RMSProp/ADOPT with beta1=0, Adam/AdEMAMix.
The tasks are Fashion-MNIST (MLP), CIFAR-10 subset (CNN), Tiny Shakespeare (character GRU).

## What is ready

- Member ownership, report draft/template, paper references and individual prompts.
- Shared draft configurations, experiment phases and output schema.
- A runnable plan/dispatch command and portable ZIP packager.

The optimizer implementations, shared data/model/training modules, actual experiment runs and
result export logic remain member work. `src/members/memberN.py` deliberately raises
NotImplementedError. Do not describe this scaffold as a finished reproducible experiment.

## Start

```sh
python3 scripts/run_member.py --member 1 --phase plan
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
comp6704-optimizer-study/
  README.md
  requirements.in
  configs/shared_protocol.json
  configs/member1.json ... member4.json
  src/common/                 shared interface specification; implementations to follow
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

Default output is dist/comp6704_project_scaffold.zip. Use --output for another destination.
The packager includes report source/PDF, configs, measured logs if later present, tables, figures,
docs and code; excludes raw datasets, model checkpoints, private environment files, .git and build.
It records SHA-256 hashes in ARCHIVE_MANIFEST.json and checks archive integrity.
It creates a local archive only, never a remote repository or commit.
Before a final release, update completion status and include actual tested reproduction instructions.
