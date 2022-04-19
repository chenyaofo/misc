import os
import psutil
import py3nvml.py3nvml as py3nvml


def get_docker_container_id(pid: int):
    cgroup = "/proc/{}/cgroup".format(pid)
    if os.path.exists(cgroup):
        with open(cgroup, "r") as f:
            for line in iter(f.readline, ""):
                if "docker" in line:
                    line = line.replace("\n", "")
                    *_, docker_path = line.split(":")
                    *_, id_ = docker_path.split("/")
                    return id_


class GPUMemory(object):
    def __init__(self, total, used, free):
        self.total = total
        self.used = used
        self.free = free


class GPUInfo(object):
    def __init__(self, handler):
        # the name like "Titan Xp"
        self.name = py3nvml.nvmlDeviceGetName(handler)

        # gpu memory (bytes)
        memory_info = py3nvml.nvmlDeviceGetMemoryInfo(handler)
        self.memory = GPUMemory(memory_info.total, memory_info.used, memory_info.free)

        # temperature (int)
        self.temperature = py3nvml.nvmlDeviceGetTemperature(handler, py3nvml.NVML_TEMPERATURE_GPU)

        # current power(float)
        self.power = py3nvml.nvmlDeviceGetPowerUsage(handler) / 1000.0
        # power limitation(float)
        self.power_litmitation = py3nvml.nvmlDeviceGetEnforcedPowerLimit(handler) / 1000.0

        # every process have two fields, process id and used gpu memory(bytes)
        self.processes = list()
        for p in py3nvml.nvmlDeviceGetComputeRunningProcesses(handler):
            info = dict(pid=p.pid, memory=p.usedGpuMemory, container_id=get_docker_container_id(p.pid))
            if psutil.pid_exists(p.pid):
                p_ = psutil.Process(pid=p.pid)
                info.update(p_.as_dict(attrs=["name", "username"]))
            self.processes.append(info)

        # gpu utilization rate(%)
        self.utilization_rate = py3nvml.nvmlDeviceGetUtilizationRates(handler).gpu

        # gpu fan speed(%)
        self.fan_speed = py3nvml.nvmlDeviceGetFanSpeed(handler)


class GPUsMonitor(object):
    def __init__(self):
        self.nvml_init = True
        try:
            py3nvml.nvmlInit()
        except Exception as e:
            self.nvml_init = False
        if self.nvml_init:
            self.driver_version = py3nvml.nvmlSystemGetDriverVersion()
            self.n_gpus = py3nvml.nvmlDeviceGetCount()

            self.update()

    def update(self):
        if self.nvml_init:
            self.gpus = list()
            for i in range(self.n_gpus):
                handler = py3nvml.nvmlDeviceGetHandleByIndex(i)
                self.gpus.append(GPUInfo(handler))

    def close(self):
        if self.nvml_init:
            py3nvml.nvmlShutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
