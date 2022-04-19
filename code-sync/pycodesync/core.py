import os
import tqdm
import paramiko

remote_dirs_cache = set()

def is_ignore_filetype(filename: str, ignore_filetypes):
    for ignore_filetype in ignore_filetypes:
        if filename.endswith(ignore_filetype):
            return True
    return False

def sftp_exists(sftp, path, cache=remote_dirs_cache):
    if path in cache:
        return True
    try:
        sftp.stat(path)
        cache.add(path)
        return True
    except FileNotFoundError:
        return False

def get_root_dirs(dir):
    rev = []
    p = dir
    while p != "/":
        rev.append(p)
        p = os.path.dirname(p)
    return reversed(rev)

def sftp_mkdirs(sftp, path):
    for p in get_root_dirs(path):
        if not sftp_exists(sftp, p):
            sftp.mkdir(p)

def get_remote_home(ssh):
    stdin, stdout, stderr = ssh.exec_command('echo $HOME')
    remote_home = stdout.readlines()[0].strip()
    return remote_home

def connect_to(remote_ip, remote_port, username, keyfile):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(remote_ip, remote_port, username, key_filename=keyfile)

    remote_home = get_remote_home(ssh)
    
    return ssh, remote_home

def sync(local_path, remote_ip, ssh, remote_path, 
         ignore_hidden_file=True, ignore_filetypes=[],
         file_limited_size=float("inf")):
    sftp = paramiko.SFTPClient.from_transport(ssh.get_transport())

    candidates = []
    for root, dirs, files in os.walk(local_path):
        if ignore_hidden_file and root.startswith("."):
            continue

        for f in files:
            filename = os.path.join(root, f)
            if ignore_hidden_file and f.startswith("."):
                continue
            if is_ignore_filetype(filename, ignore_filetypes):
                continue
            if os.path.getsize(filename) < file_limited_size:
                candidates.append(filename)
    
    print(f"local:{local_path} -> {remote_ip}:{remote_path}")

    bar_format = "{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"

    for p in tqdm.tqdm(candidates, bar_format=bar_format):
        relpath = os.path.relpath(p, local_path)
        rp = os.path.join(remote_path, relpath)
        sftp_mkdirs(sftp, os.path.dirname(rp))
        sftp.put(p, rp)