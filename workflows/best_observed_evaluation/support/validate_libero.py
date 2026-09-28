"""Strict one-suite parser; never report a stitched four-suite run."""
import importlib.util
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
suite = sys.argv[2]
source = Path(__file__).with_name('summarize_libero.py')
module_spec = importlib.util.spec_from_file_location('libero_suite_parser', source)
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
result = module.parse_suite(root / 'eval', suite, module.SUITES[suite])
server_log = root / 'eval/evaluation/libero' / suite / 'server/server.log'
server = server_log.read_text(errors='replace')
spec = re.search(r'Release policy specification: (\{[^\n]+\})', server)
assert spec, 'missing server policy specification'
loaded = json.loads(spec.group(1))
protocol = (root / 'protocol.txt').read_text()
expected_policy = re.search(r'^policy=(.+)$', protocol, re.M).group(1)
assert loaded['policy_checkpoint'] == expected_policy
assert loaded['config'] == 'phy_libero_full_finetune'
assert loaded['client_mode'] == 'single_client_serial'
assert loaded['effective_horizon'] == module.SUITES[suite]['infer_h']
assert loaded['replan'] == module.SUITES[suite]['replan']
assert 'Finished restoring checkpoint' in server
assert re.search(r'missing:\s+set\(\)', server) and re.search(r'unexpected:\s+\[\]', server)
result['server_policy_checkpoint'] = expected_policy
result['reference'] = re.search(r'^reference=(.+)$', protocol, re.M).group(1)
result['best_observed_recorded_successes'] = int(re.search(r'^recorded_successes=(\d+)/500$', protocol, re.M).group(1))
result['historical_reproduction_guaranteed'] = False
(root / 'result.json').open('x').write(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ('suite', 'episodes', 'successes', 'validation_status')}), flush=True)
