# Jumper

## Build
```
docker build -t chenyaofo/frp-ssh:0.1.0 .
```

## RUN
```
docker run -d -v /home/chenyaofo/code/jumper/frpc.ini:/frpc.ini -v /home/chenyaofo/code/jumper/authorized_keys:/home/jupmer/.ssh/authorized_keys --restart=always chenyaofo/frp-ssh:0.1.0
```

```
docker run -d -v /mnt/cephfs/home/chenyaofo/workspace/jumper/frpc.ini:/frpc.ini -v /mnt/cephfs/home/chenyaofo/workspace/jumper/authorized_keys:/home/jupmer/.ssh/authorized_keys --restart=always chenyaofo/frp-ssh:0.1.0
```
