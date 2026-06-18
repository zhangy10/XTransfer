import collections
import psutil
import pynvml
from pynvml import *
import time
import pickle


Resource_flag = False

pid = os.getpid()
print("RP" + str(pid))
p = psutil.Process(pid)
stime = time.time()


def get_cpu_mem(rd):
    cpu_percent = p.cpu_percent()
    mem_percent = p.memory_percent()
    mem_rss = p.memory_info()
    # m = threading.Timer(1.0, get_cpu_mem)
    # m.start()

    rd["cpu_percent"].append(cpu_percent)
    rd["mem_percent"].append(mem_percent)
    rd["mem_rss"].append(mem_rss)
    return cpu_percent, mem_percent, mem_rss[0]


def get_GPU(rd):
    nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        occupied_percent = info.used / info.total
        gpuUtilRate = nvmlDeviceGetUtilizationRates(handle).gpu

        rd["gpu_name_%d" % (i)] = info
        rd["gpu_occupied_%d" % (i)].append(info.used)
        rd["gpu_util_rate_%d" % (i)].append(gpuUtilRate)
    # t=threading.Timer(1.0,get_GPU)
    # t.start()
    nvmlShutdown()
    # print(data)
    return info.used, occupied_percent


def run_new(useGpu, rd):
    cpu = get_cpu_mem(rd)
    if useGpu:
        gpu = get_GPU(rd)
    # print('data_len', len(rd["mem_rss"]))


def save(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def ResourceProfile_new(useGpu, filename, interval=1):
    resource_data = collections.defaultdict(list)
    while True:
        global Resource_flag
        if Resource_flag:
            save(os.path.join(filename, "resource.pkl"), resource_data)
            print("Save resource %s" % filename)
            break

        run_new(useGpu, resource_data)
        time.sleep(interval)


def RP(useGPU, filename, interval):
    global Resource_flag
    Resource_flag = False
    p1 = threading.Thread(target=ResourceProfile_new, args=(useGPU, filename, interval))
    # p1.setDaemon(True)
    p1.start()


def stop_RP(t=1.2):
    global Resource_flag
    Resource_flag = True
    time.sleep(t)


def BindingCore(cpu_list):
    p.cpu_affinity(cpu_list)


"""
使用方法
1.调用RP( useGPU  , filename)函数, 其中filename 是log文件存储的位置
2. 调用stop_RP函数,结束监测
"""
if __name__ == "__main__":
    # print(p.cpu_percent())
    # print(psutil.cpu_count(logical=False))
    # print(psutil.cpu_percent(percpu=True))
    # print(p.cpu_affinity())
    from utils import load_dict

    # data = load_dict('C:/Users/Zber/Desktop/Face/resource.pkl')

    for i in range(2):
        RP(True, 'E:/test_res{}'.format(i), interval=1)
        time.sleep(3)
        stop_RP()

    # print(p.cpu_percent())
    # print(psutil.cpu_count(logical=False))
    # print(psutil.cpu_percent(percpu=True, interval=0.1))
    # print(psutil.cpu_percent(percpu=True, interval=0.1))
    # RP( useGpu=True , filename = 'resource_log.pkl')
    #
    # t1 = time.time()
    # i = 0
    # while True:
    #     time.sleep(1)
    #     i += 1
    #     if i >= 5:
    #         stop_RP()
    #         break
    #
    # # test
    # file = open('resource_log.pkl', 'rb')
    # data = pickle.load(file)
    # print(data)
