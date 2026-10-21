#!/usr/bin/env python3
"""Run the isolated context comparison, then restore the dashboard DS41."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/home/neron/projects/xolotl/docs/v41-ctx262k-benchmark')
DASH = Path('/home/neron/services/dashboard')
PY = '/home/neron/shared-venv-gpu/bin/python'
STATE = ROOT / 'campaign.json'
state = {'started': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'phase': 'precheck'}

def save(**values):
    state.update(values)
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(STATE)
    print(json.dumps(values), flush=True)

def console(*args, check=True):
    return subprocess.run(['python3', str(DASH/'llm_console.py'), *args], cwd=DASH,
                          text=True, capture_output=True, check=check)

save()
status = json.loads(console('status', '--json').stdout)
running = status.get('running')
if running and (running.get('model') != 'deepseek-v4.1-flash' or running.get('engine') != 'ds41'):
    raise RuntimeError('Refusing to displace an unrelated model: ' + str(running))
benchmark_rc = 1
restore_rc = 1
try:
    if running:
        save(phase='stopping_dashboard', original_running=running)
        print(console('stop').stdout, flush=True)
    save(phase='benchmark', requested_ctx=262144, arena='engine default 60 GiB')
    # Runner includes all preflight gates in the same command as model launch.
    proc = subprocess.run(['bash', str(ROOT/'run_ctx262k.sh')], cwd=ROOT)
    benchmark_rc = proc.returncode
    save(phase='validating', benchmark_exit=benchmark_rc)
    if benchmark_rc == 0:
        verify = subprocess.run([PY, str(ROOT/'verify_outputs.py')], cwd=ROOT)
        benchmark_rc = verify.returncode
        save(verification_exit=verify.returncode)
finally:
    save(phase='restoring_dashboard')
    # The harness owns shutdown of both transient engine scopes. Never load beside an orphan.
    engines = subprocess.run(['pgrep', '-x', 'ds41_generate'], capture_output=True, text=True)
    if engines.returncode == 0:
        save(restore_error='Benchmark engine still alive; refusing a second load', remaining_pids=engines.stdout.strip())
    else:
        launch = console('launch', '--model', 'deepseek-v4.1-flash', '--engine', 'ds41',
                         '--gpus', '1', '--ctx', '262144', '--port', '8080', '--host', '0.0.0.0', check=False)
        print(launch.stdout, launch.stderr, flush=True)
        if launch.returncode == 0:
            deadline = time.monotonic() + 900
            last_error = ''
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8080/v1/models', timeout=5) as response:
                        data = json.load(response)
                    model = next(x for x in data.get('data', []) if x.get('id') == 'deepseek-v4.1-flash-strata-gsq')
                    if model.get('meta', {}).get('n_ctx') != 262144:
                        raise RuntimeError('Unexpected restored context: ' + str(model))
                    restored = json.loads(console('status', '--json').stdout)
                    assert restored['running']['healthy'] and restored['running']['ctx'] == 262144
                    with urllib.request.urlopen('http://127.0.0.1:7777/health', timeout=5) as response:
                        assert response.status == 200
                    save(restored=restored['running'], bge_health=200)
                    restore_rc = 0
                    break
                except Exception as error:
                    last_error = repr(error)
                unit = subprocess.run(['systemctl', '--user', 'is-active', 'battlestation-llm.service'],
                                      capture_output=True, text=True).stdout.strip()
                if unit not in ('active', 'activating', 'reloading'):
                    break
                time.sleep(5)
            if restore_rc:
                save(restore_error=last_error or 'readiness deadline/unit failure')
                print(console('log', '--lines', '100', check=False).stdout, flush=True)
                print(console('stop', check=False).stdout, flush=True)
        else:
            save(restore_error='console launch failed', launch_exit=launch.returncode)
    save(phase='complete' if benchmark_rc == 0 and restore_rc == 0 else 'failed',
         benchmark_exit=benchmark_rc, restore_exit=restore_rc,
         finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
sys.exit(0 if benchmark_rc == 0 and restore_rc == 0 else 1)
