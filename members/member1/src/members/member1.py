"""Member 1: checks, smoke tests, validation tuning and quadratic diagnostics."""

def run_phase(phase, config, repo_root):
    if phase == "main":
        from .member1_main import run_main
        return run_main(repo_root)
    if phase == "export":
        from .member1_export import export_results
        return export_results(repo_root)
    if phase == "tune":
        from .member1_tuning import run_tuning
        return run_tuning(repo_root)
    if phase == "smoke":
        from .member1_smoke import run_smoke
        return run_smoke(config, repo_root)
    if phase == "check":
        from .member1_checks import run_checks
        return run_checks(repo_root / config["output_root"] / "notes")
    if phase == "diagnostic":
        from .member1_diagnostics import run_experiment
        return run_experiment(repo_root)
    raise NotImplementedError(
        "Member 1 must implement phase " + phase +
        " in src/members/member1.py; see docs/prompts/member1_prompt.md."
    )
