"""Scenario-response plot."""
import matplotlib.pyplot as plt
def scenario_figure(names,values):
    fig,ax=plt.subplots(figsize=(7.2,4.6));ax.plot(range(len(values)),values,marker='o');ax.set_title('Scenario objective response');ax.set_xlabel('Scenario index');ax.set_ylabel('Objective');return fig,ax
