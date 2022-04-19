import json
import getpass
import argparse
import subprocess
from .core import *

def get_branch_name():
    try:
        branch_name = subprocess.check_output(
            ['git', 'symbolic-ref', '--short', '-q', 'HEAD']).decode("utf-8")
        branch_name = branch_name.strip()
        return branch_name
    except subprocess.CalledProcessError as e:
        out_bytes = e.output
        code = e.returncode
        return None

def main():
    parser = argparse.ArgumentParser(description='Code sync command line application.')
    parser.add_argument('--server-config', type=str, default="~/.config/csync/servers.json")
    parser.add_argument('--server', '-s', type=str, default=None)
    parser.add_argument('--username', '-u', type=str, default=None)
    parser.add_argument('--keyfile', '-k', type=str, default="~/.ssh/id_rsa")
    parser.add_argument('--local-path', '-l', type=str, default=".")
    parser.add_argument('--remote-path', '-r', type=str, default=None)
    parser.add_argument('--remote-port', type=int, default=22)
    parser.add_argument('--no-ignore-hidden-file', action="store_true", default=False)
    parser.add_argument('--ignore-filetypes', type=str, default="['.pyc']")
    parser.add_argument('--file-limited-size', type=str, default="1024**2")
    parser.add_argument('--no-branch-name', action="store_true", default=False)
    
    args = parser.parse_args()

    ignore_filetypes = eval(args.ignore_filetypes)
    file_limited_size = eval(args.file_limited_size)

    with open(os.path.expanduser(args.server_config), "r") as f:
        server_config = json.load(f)
    servers = server_config["servers"]
    
    try:
        remote_ip  = servers[args.server]
    except Exception:
        raise ValueError(f"server can not be {args.server}")

    ssh, remote_home = connect_to(remote_ip=remote_ip, remote_port=args.remote_port,
    username=args.username or getpass.getuser(), keyfile=os.path.expanduser(args.keyfile))

    local_path = os.path.abspath(args.local_path)
    if args.remote_path is None:
        relpath = os.path.relpath(local_path, os.path.expanduser("~"))
        remote_path = os.path.join(remote_home, relpath)
    else:
        remote_path = args.remote_path

    if not args.no_branch_name:
        remote_path = os.path.join(remote_path, get_branch_name() + "-dev")
    
    sync(local_path, remote_ip, ssh, remote_path,  
    not args.no_ignore_hidden_file, ignore_filetypes,file_limited_size)


if __name__ == "__main__":
    main()
