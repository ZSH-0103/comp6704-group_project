"""Plan or dispatch member work; training is implemented in src/members/memberN.py."""
import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--member', type=int, required=True, choices=range(1, 5))
    p.add_argument('--phase', choices=['plan', 'check', 'smoke', 'tune', 'main', 'diagnostic', 'export'], default='plan')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    config = json.loads((ROOT / f'configs/member{args.member}.json').read_text())
    shared = json.loads((ROOT / config['shared_protocol']).read_text())
    config['shared'] = shared
    if args.phase == 'plan' or args.dry_run:
        print(json.dumps({'member': args.member, 'phase': args.phase,
            'implementation_status': 'See member source; initial scaffold is not implemented',
            'optimizers': config['optimizers'], 'tasks': config['tasks'],
            'main_runs': len(config['tasks']) * len(config['optimizers']) * len(shared['main_seeds']),
            'output_root': str(ROOT / config['output_root']),
            'protocol_status': shared['status']}, indent=2))
        return 0
    if args.phase == 'main' and (shared['status'] != 'FROZEN' or config['status'] != 'FROZEN'):
        p.error('Main experiments require FROZEN shared and member configurations after pilot/tuning.')
    sys.path.insert(0, str(ROOT))
    module = importlib.import_module(f'src.members.member{args.member}')
    try:
        module.run_phase(args.phase, config, ROOT)
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
