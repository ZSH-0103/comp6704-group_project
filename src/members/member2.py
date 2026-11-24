"""Final Momentum/Schedule-Free experiments using the recorded library runtime."""
from studies.member2.run import HOME, check, execute, export


def run_phase(phase, config, repo_root):
    if phase == 'check':
        check()
    elif phase == 'export':
        export(HOME/'published', HOME/'generated')
    elif phase in {'main', 'diagnostic', 'smoke'}:
        execute(phase, config)
    elif phase == 'tune':
        raise ValueError('Final rates are frozen in configs/member2.json. Reproduction does not retune; see studies/member2/README.md.')
    else:
        raise ValueError(phase)
