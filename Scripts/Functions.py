from scipy.optimize import curve_fit
from matplotlib import colors
import matplotlib.pyplot as plt
import pycircstat as pcs
import seaborn as sns
import pandas as pd
import numpy as np
import os

type_colors_A = {'ITL46': 'lightcoral',
                 'ITL35': 'indianred',
                 'ITL23': 'brown',
                 'PVALB': 'red',
                 'LAMP5PAX6Other': 'tomato',
                 'SST': 'coral',
                 'VIP': 'sandybrown'}

type_colors_B = {'ITL46': 'limegreen',
                 'ITL35': 'green',
                 'ITL23': 'darkgreen',
                 'PVALB': 'steelblue',
                 'LAMP5PAX6Other': 'dodgerblue',
                 'SST': 'deepskyblue',
                 'VIP': 'cadetblue'}

theta = np.linspace(0,2*np.pi,13)

def Read_Tables(files,frequency,amplitude):
    dfs = []
    for i in files:
        dfs.append(pd.read_csv('../Results/Result_Tables/' + i + '.csv'))
    columns = dfs[0].columns
    for i in range(len(dfs)):
        dfs[i] = dfs[i].groupby('ES_frequency(hz)').get_group(frequency)
        dfs[i] = dfs[i].groupby('ES_current(uA)').get_group(amplitude)
    df = dfs[0].iloc[:,:3].copy()
    for i in range(len(dfs)):
        df.loc[:,files[i]] = dfs[i].loc[:,'Amp_Median(mV)']
    return (df)

def Get_cell_type(dfcells, cell_type):
    dfcell = dfcells.get_group(cell_type)
    dfcell.loc[:,'Cell_Rotation(deg)'] = dfcell['Cell_Rotation(deg)']*np.pi/180
    newdf = dfcell.copy()
    newdf.sort_values(by=['Cell_Rotation(deg)'], inplace = True)
    return newdf.copy()

def Make_figure(dfcells, sp):
    fig = plt.figure(figsize=(12,3.5*(int(len(dfcells.groups.keys())/5)+1.5)))
    fig.suptitle(sp, fontsize = 16)
    return fig, []

def Add_subplot(fig, dfcells, i, axis):
    axis.append(fig.add_subplot(int(len(dfcells.groups.keys())/5)+1, 5, i+1, projection = 'polar'))
    axis[-1].set_xticks(np.pi/180. * np.linspace(45,  405, 4, endpoint=False))
    axis[-1].set_xticklabels(['315°','45°','135°','225°']) 
    return axis

def Show_figure():
    plt.tight_layout()
    plt.show()
    plt.close()
    
def Plot_snowman(dfcell, files, sweep_colors, sweeps, grp, swp, axis, theta, title):
    radii = list(dfcell[files[grp]])
    radii.append(radii[0])
    axis.plot(theta,radii,sweep_colors[sweeps[grp]][swp],linewidth = 2, label = sweeps[grp])
    axis.set_yticks([])
    axis.set_title(title)
    axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
  fancybox=True, shadow=True, ncol=1)
    return radii[:-1]

def Fix_Lamp_Label(axis):
    old = axis.get_xticklabels()
    new = [lbl.get_text() if 'LAMP' not in lbl.get_text() else 'LAMP5\nPAX6\nOther' for lbl in old]
    axis.set_xticks(axis.get_xticks())
    axis.set_xticklabels(new)  
    axis.set_xlabel(axis.get_xlabel(), fontsize=16) 
    axis.set_ylabel(axis.get_ylabel(), fontsize=16) 

def Plot_sensitivity_histogram(dftype, bins, i, color, fig, ys, title):
    radians = np.asarray(dftype['angledrift'])
    mean_phase_rad = pcs.descriptive.mean(np.array(radians))
    mean_phase_angle = mean_phase_rad*(180 / np.pi)
    mean_pvalue_z=pcs.rayleigh(np.array(radians))
    mean_vector_length = pcs.descriptive.vector_strength(np.array(radians))
    ax = fig.add_subplot(2, 7, i+1, projection='polar')
    ax.set_ylim(0,1.2)
    Y, X = np.histogram(dftype['angledrift']*180/np.pi, bins=bins)
    ys.append(Y)
    Xp =(X[1:] + X[:-1]) / 2
    Xp = Xp * np.pi / 180
    normY = np.true_divide(Y, (np.max(Y)))
    bars = ax.bar(Xp, normY,  width=0.4, edgecolor = 'black', color=color, alpha=0.8, linewidth=1.2)
    ax.set_axisbelow(True)
    ax.set_ylim([0,1])
    ax.set_yticks([])
    ax.set_xticks(np.pi/180. * np.linspace(45,  405, 4, endpoint=False))
    if i<7:
        ax.set_title(title.replace('LAMP5PAX6Other','LAMP5\nPAX6\nOther'))
    else:
        ax.set_xlabel(str(round((Y[1]+Y[3])*100/sum(Y)))+'%')
    return ax, ys

def sincircle(x, topness, sideness, amplitude, angledrift):
    Y = amplitude * np.abs(np.sin(x + angledrift) + topness) + sideness
    return Y

def getfitset(xdata, ydata, funct):
    if np.isnan(ydata).any():
        x,y,popt,pcov = None,None,None,None
    else:
        try:
            peakpoin = xdata[ydata.index(max(ydata))]
            p0=[0, 0, max(ydata), peakpoin]
            popt, pcov = curve_fit(funct, xdata, ydata, p0, bounds=((-1.0, 0.0, 0.0, 0.0), (1.0, max(ydata)/3, max(ydata), np.pi*2)))
            if popt[0]<0:
                p0=[-popt[0], popt[1], popt[2], (popt[3]+np.pi)%(2*np.pi)]
                popt, pcov = curve_fit(funct, xdata, ydata, p0, bounds=((0.0, 0.0, 0.0, 0.0), (1.0, max(ydata)/3, max(ydata), np.pi*2)))
            x = np.linspace(0,np.pi*2,13)
            y = funct(x, *popt)
        except:
            x,y,popt,pcov = None,None,None,None
    return x,y,popt,pcov

def Compare(files, sweeps, sweep_colors, morphologies_paths = None, 
                                                Theoretical = None, Fit = False, frequency = 1,amplitude = 10):
    df = Read_Tables(files,frequency,amplitude)                                  # frequency Hz, amplitude uA
    allresults = []
    if Fit:
        allresults.append(['Type', 'ID', 'angledrift', 'topness', 'sideness', 'amplitude'])
    for sp in sweep_colors[sweeps[0]]:
        dft = df.groupby('Cell_Type').get_group(sp)
        dfcells = dft.groupby('Cell_ID')
        fig, axs = Make_figure(dfcells, sp)
        correlations_per_cell = []
        for i, cell in enumerate(dfcells.groups.keys()):
            if morphologies_paths is not None:
                for morph in morphologies_paths:
                    if str(cell) in morph:
                        morph_path = morph
                        break
                m = pd.read_csv(morph, delimiter = ' ', header = None)
                m = m[m.iloc[:,1] == 2]
                if len(m)<10000:
                    continue     
            dfcell = Get_cell_type(dfcells, cell)
            axs = Add_subplot(fig, dfcells, i, axs)
            for grp in range(len(files)):
                if Fit:                
                    radii = list(dfcell[files[grp]])
                    radii.append(radii[0])
                    axs[-1].plot(theta,radii,sweep_colors[sweeps[grp]][sp],linewidth = 2, label = sweeps[0])
                    x,y,popt,pcov = getfitset(np.linspace(0, 2 * np.pi, 13), radii, sincircle)
                    if x is not None:
                        axs[-1].plot(theta,y,sweep_colors[sweeps[-1]][sp],linewidth = 2, label = sweeps[-1])
                        allresults.append([sp, list(dfcell['Cell_ID'])[0], popt[3], popt[0], popt[1], popt[2]])
                    axs[-1].set_yticks([])
                    axs[-1].set_title(cell)
                    axs[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                     fancybox=True, shadow=True, ncol=1)
                else:
                    if Theoretical is not None:
                        radii = Plot_snowman(dfcell, files, sweep_colors, sweeps, grp, sp, axs[-1], theta, cell)
                        maximu = np.max(radii)
                    else:
                        Plot_snowman(dfcell, files, sweep_colors, sweeps, grp, sp, axs[-1], theta, cell)
            if Theoretical is not None:
                radii = list(Theoretical.get_group(cell).loc[:,'estimate'])[::3]
                radii.append(radii[0])
                radii = np.array(radii)*maximu/np.max(radii)
                axs[-1].plot(theta,radii,sweep_colors['Theoretical'][sp],linewidth = 2, label = 'Theoretical')
                axs[-1].set_yticks([])
                axs[-1].set_title(cell)
                axs[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=1)
                correlations_per_cell.append(np.corrcoef(list(dfcell[files[0]]), radii[:-1])[0][1])
            elif Fit == False:
                correlations_per_cell.append(np.corrcoef(list(dfcell[files[0]]), list(dfcell[files[1]]))[0][1])
        if Fit == False:
            allresults.append(correlations_per_cell)
        Show_figure()
    return allresults

def Plot_correlations(correlations_per_type, type_colors_A = type_colors_A, type_colors_B = type_colors_B):
    correlations = pd.DataFrame({k : pd.Series(v) for k, v in zip(type_colors_A.keys(),correlations_per_type)})
    correlations = correlations.melt(var_name='Cell type', value_name='Correlation coefficient')
    ax = sns.stripplot(data=correlations, x='Cell type', y='Correlation coefficient', 
                       hue = 'Cell type', palette = type_colors_B, size = 5)
    ax = sns.boxplot(data=correlations, x='Cell type', y='Correlation coefficient',
                     showfliers=False, whiskerprops={'linewidth':1}, palette=type_colors_A)
    Fix_Lamp_Label(ax)
    ax.legend().set_visible(False)
    plt.show()

def Plot_sensitivity(allresults, type_colors_A = type_colors_A, type_colors_B = type_colors_B):    
    df = pd.DataFrame(allresults[1:],columns = allresults[0])

    fig = plt.figure(figsize=(12,8))
    figcnt = 1
    for i in ['angledrift', 'topness', 'sideness', 'amplitude']:
        ax = fig.add_subplot(2, 2, figcnt)
        sns.stripplot(ax = ax, data=df, x='Type', y=i, hue = 'Type', palette = type_colors_B, linewidth=1)
        Fix_Lamp_Label(ax)
        ax.get_legend().set_visible(False)
        figcnt+=1
    #plt.savefig('./Plots/FitsAndSwarmsPassive/Swarms/nocomp'+i+'.png')
    plt.show()

    fig = plt.figure(figsize=(14,4))   
    dft = df.groupby('Type')
    yyys=[]
    for i, sp in enumerate(type_colors_A):
        dftype = dft.get_group(sp)
        bins = [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360]
        color =  type_colors_A[sp]
        ax, _ = Plot_sensitivity_histogram(dftype, bins, i, color, fig, [], sp)

        bins = [-45, 45, 135, 225, 315]
        color =  type_colors_B[sp]
        ax, yyys = Plot_sensitivity_histogram(dftype, bins, i+7, color, fig, yyys,sp)
    exc = sum(yyys[:3])
    inh = sum(yyys[3:])
    Y = yyys[-1]
    ax.set_xlabel(str(round((Y[1]+Y[3])*100/sum(Y)))+
                  '%\nTotal exc: '+str(round((exc[1]+exc[3])*100/sum(exc)))+
                  '%\nTotal inh: '+str(round((inh[1]+inh[3])*100/sum(inh)))+'%')
    #plt.savefig('./Plots/FitsAndSwarmsPassive/XvsY.png')
    plt.show()
    
def Compare_multy_freq_amp(files, sweeps, sweep_colors, frequencies = [1,3,5,8,10,28,30,50,100,140,300], amplitudes = [10,100,500]):
    for amplitude in amplitudes: # uA
        print("ES AMPLITUDE = " + str(amplitude) + 'uA')
        freq_results = []
        freq_results.append(['freq', 'Exc', 'Inh'])
        freq_results_per_class = []
        col_names = ['freq']
        for i in type_colors_A.keys():
            col_names.append(i)
        freq_results_per_class.append(col_names)
        for frequency in frequencies:
            allresults = []
            allresults.append(['Type', 'ID', 'angledrift', 'topness', 'sideness', 'amplitude'])
            df = Read_Tables(files,frequency,amplitude)
            for sp in type_colors_A:
                dft = df.groupby('Cell_Type').get_group(sp).groupby('Cell_ID')
                for i, kk in enumerate(dft.groups.keys()):
                    dfcell = dft.get_group(kk)
                    dfcell.loc[:,'Cell_Rotation(deg)'] = dfcell['Cell_Rotation(deg)']*np.pi/180
                    dfcell = dfcell.sort_values(by=['Cell_Rotation(deg)'])
                    for grp in range(len(files)):
                        radii = list(dfcell[files[grp]])
                        radii.append(radii[0])
                        x,y,popt,pcov = getfitset(np.linspace(0, 2 * np.pi, 13), radii, sincircle)
                        if x is not None:
                            allresults.append([sp, list(dfcell['Cell_ID'])[0], popt[3], popt[0], popt[1], popt[2]])

            df = pd.DataFrame(allresults[1:],columns = allresults[0])
            dft = df.groupby('Type')
            yyys = []
            yyyy = []
            yyyy.append(frequency)
            for i, kk in enumerate(type_colors_A):
                dftype = dft.get_group(kk)
                radians = np.asarray(dftype['angledrift'])
                mean_phase_rad = pcs.descriptive.mean(np.array(radians))
                mean_phase_angle = mean_phase_rad*(180 / np.pi)
                mean_pvalue_z=pcs.rayleigh(np.array(radians))
                mean_vector_length = pcs.descriptive.vector_strength(np.array(radians))
                Y, X = np.histogram(dftype['angledrift']*180/np.pi, bins=[-45, 45, 135, 225, 315])
                yyys.append(Y)
                yyyy.append(round((Y[1]+Y[3])*100/sum(Y)))
            exc = sum(yyys[:3])
            inh = sum(yyys[3:])
            freq_results.append([frequency,round((exc[1]+exc[3])*100/sum(exc)),round((inh[1]+inh[3])*100/sum(inh))])
            freq_results_per_class.append(yyyy)


        fig = plt.figure(figsize=(12,4))
        ax = fig.add_subplot(1,2,1) 
        df = pd.DataFrame(freq_results[1:], columns = freq_results[0])
        ax.plot(df.iloc[:,0], df.iloc[:,1], marker = 'o', label = 'Excitatory types')
        ax.plot(df.iloc[:,0], df.iloc[:,2], marker = 'o', label = 'Inhibitory types')
        ax.set_xlabel('ES frequency (Hz)', fontsize = 16)
        ax.set_ylabel('X sensitive cells (%)', fontsize = 16)
        ax.set_xscale('log')
        ax.set_ylim(0,80)
        ax.legend()

        ax = fig.add_subplot(1,2,2) 
        df = pd.DataFrame(freq_results_per_class[1:], columns = freq_results_per_class[0])
        for i in type_colors_A.keys():
            ax.plot(df.iloc[:,0], df.loc[:,i], marker = 'o', label = i, color = type_colors_B[i])
        ax.set_xlabel('ES frequency (Hz)', fontsize = 16)
        ax.set_ylabel('X sensitive cells (%)', fontsize = 16)
        ax.set_xscale('log')
        ax.set_ylim(0,80)
        ax.legend(ncol = 3)
        plt.show()
    