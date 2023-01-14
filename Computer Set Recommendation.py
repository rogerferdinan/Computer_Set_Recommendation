import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import timeit
import time


timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""

def CMF(price):
    cpu_list = pd.read_csv("cpu.csv")
    mobo_list = pd.read_csv("mobo.csv")
    vga_list = pd.read_csv("vga.csv")

    cpu_list = cpu_list[cpu_list["harga_cpu"] <= price]
    mobo_list = mobo_list[mobo_list["harga_mobo"] <= price]
    vga_list = vga_list[vga_list["harga_vga"] <= price]

    cpu_mobo_list = cpu_list[["nama_cpu", "harga_cpu", "socket"]].merge(
                                mobo_list[["nama_mobo", "harga_mobo", "socket"]],
                                on=['socket'],
                                how='inner'
                                )
    
    mobo_vga_list = mobo_list[["nama_mobo", "harga_mobo"]].merge(
                                vga_list[["nama_vga", "harga_vga"]],
                                how='cross'
                                )

    cpu_mobo_vga_list = cpu_mobo_list[["nama_cpu","harga_cpu", "nama_mobo", "harga_mobo"]].merge(
                                        mobo_vga_list[["nama_mobo", "nama_vga", "harga_vga"]], 
                                                on=['nama_mobo'],
                                                how='inner'
                                        )
    cpu_mobo_vga_list["total_harga"] = cpu_mobo_vga_list['harga_cpu'] + cpu_mobo_vga_list["harga_mobo"] + cpu_mobo_vga_list["harga_vga"]

    cpu_mobo_vga_list = cpu_mobo_vga_list[(cpu_mobo_vga_list["total_harga"] <= price)]
    cpu_mobo_vga_list = cpu_mobo_vga_list.sort_values(by=["total_harga"], ascending=False)
    return cpu_mobo_vga_list.head(1)["total_harga"].values

def CMSFR(price):
    cpu_list = pd.read_csv("cpu.csv")
    mobo_list = pd.read_csv("mobo.csv")
    vga_list = pd.read_csv("vga.csv")

    cpu_mobo_list = cpu_list[["nama_cpu", "harga_cpu", "socket"]].merge(
                                mobo_list[["nama_mobo", "harga_mobo", "socket"]],
                                on=['socket'],
                                how='inner'
                                )
    
    mobo_vga_list = mobo_list[["nama_mobo", "harga_mobo"]].merge(
                                vga_list[["nama_vga", "harga_vga"]],
                                how='cross'
                                )

    cpu_mobo_vga_list = cpu_mobo_list[["nama_cpu","harga_cpu", "nama_mobo", "harga_mobo"]].merge(
                                        mobo_vga_list[["nama_mobo", "nama_vga", "harga_vga"]], 
                                                on=['nama_mobo'],
                                                how='inner'
                                        )
    cpu_mobo_vga_list['total_harga'] = cpu_mobo_vga_list['harga_cpu'] + cpu_mobo_vga_list["harga_mobo"] + cpu_mobo_vga_list["harga_vga"]
    cpu_mobo_vga_list["cpu_ratio"] = cpu_mobo_vga_list["harga_cpu"] / cpu_mobo_vga_list["total_harga"]
    cpu_mobo_vga_list["vga_ratio"] = cpu_mobo_vga_list["harga_vga"] / cpu_mobo_vga_list["total_harga"]
    
    # Logarithm Regression
    vga_log_reg = np.polyfit(np.log(cpu_mobo_vga_list["total_harga"]), cpu_mobo_vga_list["vga_ratio"], 1)
    expected_vga_ratio = vga_log_reg[0] * np.log(price) + vga_log_reg[1]
    cpu_log_reg = np.polyfit(np.log(cpu_mobo_vga_list["total_harga"]), cpu_mobo_vga_list["cpu_ratio"], 1)
    expected_cpu_ratio = cpu_log_reg[0] * np.log(price) + cpu_log_reg[1]

    # new_x = np.arange(start=cpu_mobo_vga_list["total_harga"].min(), stop=cpu_mobo_vga_list["total_harga"].max())
    # plt.title("VGA Price Ratio Regression")
    # plt.scatter(cpu_mobo_vga_list["total_harga"], cpu_mobo_vga_list["vga_ratio"])
    # plt.plot(new_x, vga_log_reg[0] * np.log(new_x) + vga_log_reg[1], "g")
    # plt.legend(["VGA Price Ratio", "Logarithm Regression"])
    # plt.show()

    # plt.clf()
    # plt.title("CPU Price Ratio Regression")
    # plt.scatter(cpu_mobo_vga_list["total_harga"], cpu_mobo_vga_list["cpu_ratio"])
    # plt.plot(new_x, cpu_log_reg[0] * np.log(new_x) + cpu_log_reg[1], "r")
    # plt.legend(["CPU Price Ratio", "Logarithm Regression"])
    # plt.show()

    cpu_mobo_vga_list = cpu_mobo_vga_list[(cpu_mobo_vga_list["total_harga"] <= price) & (cpu_mobo_vga_list["vga_ratio"] <= expected_vga_ratio) & (cpu_mobo_vga_list["cpu_ratio"] <= expected_cpu_ratio)]
    cpu_mobo_vga_list = cpu_mobo_vga_list.sort_values(by=["total_harga", "vga_ratio", "cpu_ratio"], ascending=False)
    return cpu_mobo_vga_list.head(1)["total_harga"].values

def CMFR(price):
    cpu_list = pd.read_csv("cpu.csv")
    mobo_list = pd.read_csv("mobo.csv")
    vga_list = pd.read_csv("vga.csv")

    # cpu_list = cpu_list[cpu_list["harga_cpu"] <= price]
    # mobo_list = mobo_list[mobo_list["harga_mobo"] <= price]
    # vga_list = vga_list[vga_list["harga_vga"] <= price]

    cpu_mobo_list = cpu_list[["nama_cpu", "harga_cpu", "socket"]].merge(
                                mobo_list[["nama_mobo", "harga_mobo", "socket"]],
                                on=['socket'],
                                how='inner'
                                )
    
    mobo_vga_list = mobo_list[["nama_mobo", "harga_mobo"]].merge(
                                vga_list[["nama_vga", "harga_vga"]],
                                how='cross'
                                )

    cpu_mobo_vga_list = cpu_mobo_list[["nama_cpu","harga_cpu", "nama_mobo", "harga_mobo"]].merge(
                                        mobo_vga_list[["nama_mobo", "nama_vga", "harga_vga"]], 
                                                on=['nama_mobo'],
                                                how='inner'
                                        )
    cpu_mobo_vga_list['total_harga'] = cpu_mobo_vga_list['harga_cpu'] + cpu_mobo_vga_list["harga_mobo"] + cpu_mobo_vga_list["harga_vga"]
    cpu_mobo_vga_list["cpu_ratio"] = cpu_mobo_vga_list["harga_cpu"] / cpu_mobo_vga_list["total_harga"]
    cpu_mobo_vga_list["vga_ratio"] = cpu_mobo_vga_list["harga_vga"] / cpu_mobo_vga_list["total_harga"]
    
    # Logarithm Regression
    vga_log_reg = np.polyfit(np.log(cpu_mobo_vga_list["total_harga"]), cpu_mobo_vga_list["vga_ratio"], 1)
    expected_vga_ratio = vga_log_reg[0] * np.log(price) + vga_log_reg[1]
    cpu_log_reg = np.polyfit(np.log(cpu_mobo_vga_list["total_harga"]), cpu_mobo_vga_list["cpu_ratio"], 1)
    expected_cpu_ratio = cpu_log_reg[0] * np.log(price) + cpu_log_reg[1]

    new_x = np.arange(start=cpu_mobo_vga_list["total_harga"].min(), stop=cpu_mobo_vga_list["total_harga"].max())
    plt.clf()
    plt.title("CMFR VGA Ratio Regression")
    plt.scatter(cpu_mobo_vga_list["total_harga"], cpu_mobo_vga_list["vga_ratio"])
    plt.plot(new_x, vga_log_reg[0] * np.log(new_x) + vga_log_reg[1], "g")
    plt.legend(["VGA Price Ratio", "Logarithm Regression"])
    plt.show()

    plt.clf()
    plt.title("CMFR CPU Ratio Regression")
    plt.scatter(cpu_mobo_vga_list["total_harga"], cpu_mobo_vga_list["cpu_ratio"])
    plt.plot(new_x, cpu_log_reg[0] * np.log(new_x) + cpu_log_reg[1], "g")
    plt.legend(["CPU Price Ratio", "Logarithm Regression"])
    plt.show()

    cpu_mobo_vga_list = cpu_mobo_vga_list[(cpu_mobo_vga_list["total_harga"] <= price) & (cpu_mobo_vga_list["vga_ratio"] <= expected_vga_ratio) & (cpu_mobo_vga_list["cpu_ratio"] <= expected_cpu_ratio)]
    cpu_mobo_vga_list = cpu_mobo_vga_list.sort_values(by=["total_harga","vga_ratio", "cpu_ratio"], ascending=False)
    return cpu_mobo_vga_list.head(1)["total_harga"].values

def PMFR(price):
    cpu_list = pd.read_csv("cpu.csv")
    mobo_list = pd.read_csv("mobo.csv")
    vga_list = pd.read_csv("vga.csv")

    # cpu_list = cpu_list[cpu_list["harga_cpu"] <= price]
    # mobo_list = mobo_list[mobo_list["harga_mobo"] <= price]
    # vga_list = vga_list[vga_list["harga_vga"] <= price]

    cpu_mobo_list = cpu_list[["nama_cpu", "harga_cpu", "socket"]].merge(
                                mobo_list[["nama_mobo", "harga_mobo", "socket"]],
                                on=['socket'],
                                how='inner'
                                )
    cpu_mobo_list["total_harga"] = cpu_mobo_list["harga_cpu"] + cpu_mobo_list["harga_mobo"]
    cpu_mobo_list["cpu_ratio"] = cpu_mobo_list["harga_cpu"] / cpu_mobo_list["total_harga"]
    cpu_log_reg = np.polyfit(np.log(cpu_mobo_list["total_harga"]), cpu_mobo_list["cpu_ratio"], 1)
    expected_cpu_ratio = cpu_log_reg[0] * np.log(price) + cpu_log_reg[1]

    cpu_new_x = np.arange(start=cpu_mobo_list["total_harga"].min(), stop=cpu_mobo_list["total_harga"].max())
    plt.clf()
    plt.title("PMFR CPU Ratio Regression")
    plt.scatter(cpu_mobo_list["total_harga"], cpu_mobo_list["cpu_ratio"])
    plt.plot(cpu_new_x, cpu_log_reg[0] * np.log(cpu_new_x) + cpu_log_reg[1], "g")
    plt.legend(["CPU Price Ratio", "Logarithm Regression"])
    plt.show()

    mobo_vga_list = mobo_list[["nama_mobo", "harga_mobo"]].merge(
                                vga_list[["nama_vga", "harga_vga"]],
                                how='cross'
                                )
    mobo_vga_list["total_harga"] = mobo_vga_list["harga_vga"] + mobo_vga_list["harga_mobo"]
    mobo_vga_list["vga_ratio"] = mobo_vga_list["harga_vga"] / mobo_vga_list["total_harga"]
    vga_log_reg = np.polyfit(np.log(mobo_vga_list["total_harga"]), mobo_vga_list["vga_ratio"], 1)
    expected_vga_ratio = vga_log_reg[0] * np.log(price) + vga_log_reg[1]

    vga_new_x = np.arange(start=mobo_vga_list["total_harga"].min(), stop=mobo_vga_list["total_harga"].max())
    plt.clf()
    plt.title("PMFR VGA Ratio Regression")
    plt.scatter(mobo_vga_list["total_harga"], mobo_vga_list["vga_ratio"])
    plt.plot(vga_new_x, vga_log_reg[0] * np.log(vga_new_x) + vga_log_reg[1], "g")
    plt.legend(["CPU Price Ratio", "Logarithm Regression"])
    plt.show()

    cpu_mobo_list = cpu_mobo_list[cpu_mobo_list["cpu_ratio"] <= expected_cpu_ratio]
    mobo_vga_list = mobo_vga_list[mobo_vga_list["vga_ratio"] <= expected_vga_ratio]

    cpu_mobo_vga_list = cpu_mobo_list[["nama_cpu","harga_cpu", "nama_mobo", "harga_mobo"]].merge(
                                        mobo_vga_list[["nama_mobo", "nama_vga", "harga_vga"]], 
                                                on=['nama_mobo'],
                                                how='inner'
                                        )
    cpu_mobo_vga_list['total_harga'] = cpu_mobo_vga_list['harga_cpu'] + cpu_mobo_vga_list["harga_mobo"] + cpu_mobo_vga_list["harga_vga"]
    cpu_mobo_vga_list = cpu_mobo_vga_list[(cpu_mobo_vga_list["total_harga"] <= price)]
    cpu_mobo_vga_list = cpu_mobo_vga_list.sort_values(by=["total_harga"], ascending=False)
    return cpu_mobo_vga_list.head(1)["total_harga"].values

if __name__ == "__main__":
    pricelist = np.arange(start=1000, step=50, stop=11000)

    CMF_mean_time = np.zeros(pricelist.shape[0])
    CMSFR_mean_time = np.zeros(pricelist.shape[0])
    CMFR_mean_time = np.zeros(pricelist.shape[0])
    PMFR_mean_time = np.zeros(pricelist.shape[0])
    testing_data = np.zeros(pricelist.shape[0])
    
    CMSFR_error = 0
    CMFR_error = 0
    PMSFR_error = 0

    for i, price in enumerate(pricelist):
        CMF_time = timeit.timeit(f"CMF({price})", "from __main__ import CMF", number=1)
        testing_data[i] = CMF_time[1]
        CMF_mean_time[i] = CMF_time[0]

        CMSFR_time = timeit.timeit(f"CMSFR({price})", "from __main__ import CMSFR", number=1)
        CMSFR_mean_time[i] = CMSFR_time[0]
        if CMSFR_time[1] != testing_data[i]:
            CMSFR_error += 1
        
        CMFR_time = timeit.timeit(f"CMFR({price})", "from __main__ import CMFR", number=1)
        CMFR_mean_time[i] = CMFR_time[0]
        if CMFR_time[1] != testing_data[i]:
            CMFR_error += 1
        
        PMFR_time = timeit.timeit(f"PMFR({price})", "from __main__ import PMFR", number=1)
        PMFR_mean_time[i] = PMFR_time[0]
        if PMFR_time[1] != testing_data[i]:
            PMSFR_error += 1
        

    print(f"CMF: {CMF_mean_time.mean():.2f} s")
    print(f"CMSFR: {CMSFR_mean_time.mean():.2f} s")
    print(f"CMFR: {CMFR_mean_time.mean():.2f} s")
    print(f"PMSFR: {PMFR_mean_time.mean():.2f} s")

    print(f"CMSFR acc: {(pricelist.shape[0]-CMSFR_error)/pricelist.shape[0]*100}%")
    print(f"CMFR acc: {(pricelist.shape[0]-CMFR_error)/pricelist.shape[0]*100}%")
    print(f"PMSFR acc: {(pricelist.shape[0]-PMSFR_error)/pricelist.shape[0]*100} %")