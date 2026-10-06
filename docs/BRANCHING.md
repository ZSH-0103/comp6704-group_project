# Proposed Git branches and integration rules

The repository starts with main. Create the following member branches when beginning each module.

| Branch | Scope |
| --- | --- |
| main | Reviewed common protocol and integrated report |
| member1-sgd-dog | Member 1 optimizer pair, own results and owned report files |
| member2-schedule-free | Member 2 pair and owned files |
| member3-rmsprop-adopt | Member 3 pair, data/model specification and owned files |
| member4-ademamix | Member 4 pair, metrics and owned files |

Everyone can work on their own module, theory and controlled diagnostic immediately.
Before main runs, agree one frozen shared protocol. Shared changes require communicating the
changed field and its consequences; do not silently fork data splits or architectures.
Each member provides a reviewed row for shared tables; the named integrator edits the shared file.
Use stable table labels, not hard-coded page numbers or transient table numbers.

Report source: report/. Keep only one editable report copy per checkout.
Do not put build outputs, downloaded datasets, .venv or checkpoints in Git.
