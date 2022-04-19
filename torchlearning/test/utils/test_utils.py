from torchlearning.utils import get_host_ip

class TestGetHostIP():
    def test(self):
        ip = get_host_ip()
        print(ip)