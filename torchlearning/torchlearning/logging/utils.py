import os
import psutil
import platform
import torchlearning
from torchlearning.utils import get_host_ip
from torchlearning.utils.gpu_utils import GPUsMonitor


def get_os_full_description():
    if os.path.exists('/etc/lsb-release'):
        lines = open('/etc/lsb-release').read().split('\n')
        for line in lines:
            if line.startswith('DISTRIB_DESCRIPTION='):
                name = line.split('=')[1]
                if name[0] == '"' and name[-1] == '"':
                    return name[1:-1]
    return None


def bytes2human(n):
    symbols = ('KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return "%sB" % n


def to_line(nt, target_transform):
    strs = []
    for name in nt._fields:
        value = getattr(nt, name)
        if name != 'percent':
            value = target_transform(value)
            strs.append("{} {}".format(name.capitalize(), value))
    return ", ".join(strs)


def report_machine_info(logger=torchlearning.logger):
    if get_os_full_description() is not None:
        logger.info(
            "OS: {}({} {} {})".format(get_os_full_description(), platform.system(), platform.release(),
                                      platform.machine()))
    else:
        logger.info(
            "OS: {} {} {}".format(platform.system(), platform.release(), platform.machine()))

    logger.info("IP: {}".format(get_host_ip()))

    logger.info(
        "CPU : {}.".format(to_line(psutil.cpu_times_percent(interval=1), target_transform=lambda v: "{}%".format(v))))
    logger.info("Memory: {}.".format(to_line(psutil.virtual_memory(), target_transform=bytes2human)))
    logger.info("Swap: {}.".format(to_line(psutil.swap_memory(), target_transform=bytes2human)))
    with GPUsMonitor() as gm:
        if gm.nvml_init:
            for i, gpu in enumerate(gm.gpus):
                logger.info("GPU {}: {}, {}MB/{}MB, {}W/{}W, {}%".format(
                    i, gpu.name,
                    gpu.memory.used // (1014 ** 2), gpu.memory.total // (1014 ** 2),
                    int(gpu.power), int(gpu.power_litmitation),
                    gpu.utilization_rate
                ))
