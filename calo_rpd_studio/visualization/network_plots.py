"""Engineering result plots for bus voltage and branch loading."""
import matplotlib.pyplot as plt
def voltage_profile_figure(bus_numbers,vm):
    fig,ax=plt.subplots(figsize=(7.2,4.6));ax.plot(bus_numbers,vm,marker='o',label='Voltage magnitude');ax.set_title('Bus voltage profile');ax.set_xlabel('Bus');ax.set_ylabel('Voltage (p.u.)');return fig,ax
def branch_loading_figure(branch_numbers,loading):
    fig,ax=plt.subplots(figsize=(7.2,4.6));ax.bar(branch_numbers,loading);ax.axhline(100,linestyle='--',label='Thermal limit');ax.set_title('Branch loading');ax.set_xlabel('Branch index');ax.set_ylabel('Loading (%)');return fig,ax
