"""Local launcher: use existing SSH helpers, never write or log their credentials."""
import json
import os
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, SSH_ASKPASS=str(root / '.codex_askpass_s2.cmd'),
               SSH_ASKPASS_REQUIRE='force', DISPLAY='codex')
    remote = '/data/home/kxrgzn/lzh/MyDualTalk/runs/avatar_comparison_20260913_s2/tools/'
    subprocess.run(['scp', '-q', '-o', 'StrictHostKeyChecking=yes',
                    str(root / 'scripts/transfer_avatar_comparison.py'), 'kxrgzn@172.19.128.133:' + remote], env=env, check=True)
    known = subprocess.check_output(['ssh-keygen', '-F', '180.201.150.24'], text=True)
    if not known.strip():
        raise ValueError('Server1 must have a previously verified SSH host key')
    password = subprocess.check_output(['cmd.exe', '/c', str(root / '.codex_askpass_s1.cmd')], text=True).strip()
    result = subprocess.run(['ssh', '-T', '-o', 'StrictHostKeyChecking=yes', 'kxrgzn@172.19.128.133',
                            '/data/home/kxrgzn/anaconda3/envs/lzh_DDG/bin/python', '-u', '-B',
                            remote + 'transfer_avatar_comparison.py', '--launch'],
                           env=env, input=json.dumps(dict(password=password, known_hosts=known)),
                           text=True, capture_output=True)
    del password
    print(result.stdout)
    if result.returncode:
        print(result.stderr)
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
