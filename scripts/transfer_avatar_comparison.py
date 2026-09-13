"""Detached direct server1 -> server2 rsync; credentials exist only in process memory."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data/home/kxrgzn/lzh/MyDualTalk/runs/avatar_comparison_20260913_s2')
SOURCE = '/home/s21_yhr/lzh/MyDualTalk/'
EXPORT = SOURCE + 'runs/avatar_comparison_20260913_s2_export/'


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def main():
    auth = json.load(sys.stdin)
    ROOT.mkdir(parents=True, exist_ok=True)
    if '--launch' in sys.argv:
        if (ROOT / 'transfer_status.json').exists():
            raise ValueError('Transfer already registered; inspect its status before restarting')
        with (ROOT / 'transfer.log').open('wb') as log:
            child = subprocess.Popen([sys.executable, '-u', '-B', str(Path(__file__).resolve())],
                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            child.stdin.write(json.dumps(auth).encode())
            child.stdin.close()
        print(json.dumps(dict(pid=child.pid, log=str(ROOT / 'transfer.log'))))
        return
    known = ROOT / 'server1_known_hosts'
    known.write_text(auth['known_hosts'])
    os.chmod(known, 0o600)
    askpass = ROOT / 'askpass.py'
    askpass.write_text('#!/usr/bin/python3\nimport os\nprint(os.environ["AVATAR_SOURCE_PASSWORD"])\n')
    os.chmod(askpass, 0o700)
    env = dict(os.environ, SSH_ASKPASS=str(askpass), SSH_ASKPASS_REQUIRE='force', DISPLAY='codex',
               AVATAR_SOURCE_PASSWORD=auth['password'])
    del auth
    ssh_args = ['ssh', '-T', '-o', 'ConnectTimeout=20', '-o', 'StrictHostKeyChecking=yes',
                '-o', 'UserKnownHostsFile=' + str(known)]
    (ROOT / 'source').mkdir(exist_ok=True)
    for group in ('bootstrap', 'smoke', 'evaluation', 'training'):
        started = time.time()
        write_json(ROOT / 'transfer_status.json', dict(status='running', stage=group, pid=os.getpid(), started=started))
        if group == 'smoke':
            while True:
                ready = subprocess.run(ssh_args + ['s21_yhr@180.201.150.24', 'test', '-f', EXPORT + 'export_status.json'],
                                       env=env, stdin=subprocess.DEVNULL, capture_output=True)
                if ready.returncode == 0:
                    break
                if ready.returncode != 1:
                    raise RuntimeError('Could not check source export readiness')
                time.sleep(20)
            listing = (ROOT / 'bootstrap_files.txt').read_bytes() + b'runs/avatar_comparison_20260913_s2_export/transfer_manifest.json\n'
        else:
            listing = subprocess.check_output(ssh_args + ['s21_yhr@180.201.150.24', 'cat', EXPORT + group + '_files.txt'],
                                              env=env, stdin=subprocess.DEVNULL)
        filelist = ROOT / (group + '_files.txt')
        filelist.write_bytes(listing)
        command = ['rsync', '-a', '--partial', '--info=progress2', '--stats', '--files-from=' + str(filelist),
                   '-e', ' '.join(ssh_args), 's21_yhr@180.201.150.24:' + SOURCE, str(ROOT / 'source') + '/']
        with (ROOT / ('transfer_' + group + '.log')).open('wb') as log:
            subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(ROOT / (group + '_transfer_complete.json'), dict(completed=time.time(), elapsed_seconds=time.time()-started))
    write_json(ROOT / 'transfer_status.json', dict(status='complete', completed=time.time(), pid=os.getpid()))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if ROOT.exists():
            write_json(ROOT / 'transfer_status.json', dict(status='failed', error=str(error), time=time.time()))
        raise
