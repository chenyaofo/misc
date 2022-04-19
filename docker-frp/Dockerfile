from ubuntu:20.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    TZ=Asia/Shanghai

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    GIT_CLONE="git clone --depth 10" && \
    cat /etc/apt/sources.list && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL wget tzdata openssh-server && \
    apt-get clean

RUN ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime && \
    echo ${TZ} > /etc/timezone && \
    dpkg-reconfigure --frontend noninteractive tzdata

RUN echo "**** setup openssh environment ****" && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/g' /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/g' /etc/ssh/sshd_config

RUN useradd -m -d /home/jupmer -u 1000 jumper 

RUN mkdir -p /home/jupmer/frp && \
    cd /home/jupmer/frp && \
    wget --no-check-certificate https://github.91chifun.workers.dev/https://github.com//fatedier/frp/releases/download/v0.37.1/frp_0.37.1_linux_amd64.tar.gz && \
    tar -zxvf frp_0.37.1_linux_amd64.tar.gz --strip-components 1 && \
    rm -rf frp_0.37.1_linux_amd64.tar.gz && \
    cd .. && \
    mkdir -p /home/jupmer/.ssh && \
    chown -R jumper:jumper /home/jupmer/.ssh

CMD chown jumper:jumper /home/jupmer/.ssh/authorized_keys && service ssh start && /home/jupmer/frp/frpc -c /frpc.ini
