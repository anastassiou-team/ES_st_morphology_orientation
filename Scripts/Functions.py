from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import norm
from collections import Counter
from statannot import add_stat_annotation
from matplotlib import colors
from matplotlib.colors import ListedColormap
from neurom import viewer
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pycircstat as pcs
import seaborn as sns
import warnings
import pandas as pd
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=(SettingWithCopyWarning))
import numpy as np
import os, neurom

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

custom_colors = {'Passive axon\ncm = 4':'maroon',
                 'Passive axon\ncm = 1':'red',
                 'Passive axon\ncm = 0.5':'tomato',
                 'Active axon dendrite\nparameters':'darkviolet',
                 'Active axon soma\nparameters':'chartreuse',
                 'HoF 1':'darkviolet',
                 'HoF 2':'cadetblue',
                 'HoF 3':'deepskyblue',
                 'HoF 4':'dodgerblue',
                 'HoF 5':'steelblue'}

def Read_Tables(files, frequency, amplitude, phase = False, maximal = False, sweeps = None):
    dfs = []
    for s,i in enumerate(files):
        if sweeps is not None:
            if 'MMR' in sweeps[s] or 'VL' in sweeps[s]:
                dfs.append(pd.read_pickle('../Results/Result_Tables/' + i + '.pkl'))
            else:
                dfs.append(pd.read_csv('../Results/Result_Tables/' + i + '.csv'))
        else:
            dfs.append(pd.read_csv('../Results/Result_Tables/' + i + '.csv'))
    columns = dfs[0].columns
    for i in range(len(dfs)):
        dfs[i] = dfs[i].groupby('ES_frequency(hz)').get_group(frequency)
        dfs[i] = dfs[i].groupby('ES_current(uA)').get_group(amplitude)
        if phase:
            if not maximal:
                dfs[i] = dfs[i].groupby('Cell_Rotation(deg)').get_group(0)
    common_cells = None
    if len(dfs)>1:
        cell_sets = []
        max_set = 0
        for i in dfs:
            cell_sets.append(set(i.loc[:,'Cell_ID'].astype(int)))
            if max_set < len(cell_sets[-1]):
                max_set = len(cell_sets[-1])
        common_cells = cell_sets[0]
        for s in cell_sets[1:]:
            common_cells = common_cells.intersection(s)
        for i in range(len(dfs)):
            dfs[i] = dfs[i][dfs[i]['Cell_ID'].astype(int).isin(common_cells)]
        if max_set != len(common_cells):
            print ("WARNING: Reduced cells from "+str(max_set)+" to "+str(len(common_cells))+" due to mismatch.")     
    
    df = dfs[0].iloc[:,:3].copy().reset_index(drop = True)
    for i in range(len(dfs)):
        assert list(df.loc[:,'Cell_ID'])==list(dfs[i].loc[:,'Cell_ID'].astype(int))
        if phase:
            df.loc[:,files[i]] = (dfs[i].loc[:,'Phs@40000(deg)'].reset_index(drop = True)+270)%360
            if maximal:
                df.loc[:,'Amplitude_'+files[i]] = dfs[i].loc[:,'Amp_Median(mV)'].reset_index(drop = True)
        else: 
            if sweeps is not None:
                if 'VL' in sweeps[i]:
                    df.loc[:,files[i]] = dfs[i].loc[:,'mean_vector_length'].reset_index(drop = True)
                elif 'MMR' in sweeps[i]:
                    df.loc[:,files[i]] = dfs[i].loc[:,'MMR'].reset_index(drop = True)
                else:
                    df.loc[:,files[i]] = dfs[i].loc[:,'Amp_Median(mV)'].reset_index(drop = True)
            else:
                df.loc[:,files[i]] = dfs[i].loc[:,'Amp_Median(mV)'].reset_index(drop = True)
    return (df)

def Get_cell_type(dfcells, cell_type):
    dfcell = dfcells.get_group(cell_type)
    dfcell.loc[:,'Cell_Rotation(deg)'] = dfcell['Cell_Rotation(deg)'].astype('int')*np.pi/180
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
    
def fix_angles(angles, data):
    sorted_indices = sorted(range(len(angles)), key=lambda i: angles[i])
    angles_sorted = [angles[i] for i in sorted_indices]
    data_sorted = [data[i] for i in sorted_indices]
    if angles_sorted != sorted(angles):
        raise ValueError("Sort fail.")
    diffs = []
    for i in range(len(angles) - 1):
        diffs.append(angles[i + 1] - angles[i])
    diff_counter = Counter(diffs)
    most_common_diff, _ = diff_counter.most_common(1)[0]
    if np.isclose(most_common_diff, np.pi / 6):
        step_size = np.pi / 6
        total_steps = 13
    elif np.isclose(most_common_diff, np.pi / 18):
        step_size = np.pi / 18
        total_steps = 37
    else:
        raise ValueError("Step size is neither 30° nor 10°.")
    complete_angles = np.arange(0, 2 * np.pi, step_size)
    filled_angles = [np.nan] * len(complete_angles)
    filled_data = [np.nan] * len(complete_angles)
    for i, angle in enumerate(angles):
        closest_index = np.argmin(np.abs(complete_angles - angle))
        filled_data[closest_index] = data[i]
    return list(complete_angles), list(filled_data)
    
def Plot_snowman(dfcell, files, sweep_colors, sweeps, grp, swp, axis, title):
    radii = list(dfcell[files[grp]])
    thetaR = list(dfcell['Cell_Rotation(deg)'])
    thetaR, radii = fix_angles(thetaR, radii)
    radii.append(radii[0])
    thetaR.append(thetaR[0])
    axis.plot(thetaR,radii,sweep_colors[sweeps[grp]][swp],linewidth = 2, label = sweeps[grp])
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

def Plot_sensitivity_histogram(dftype, bins, i, color, fig, ys, title, quad = False):
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
    elif quad:
        ax.set_xlabel('X+: '+str(round((Y[0])*100/sum(Y)))+'%\n'+
                      'Y+: '+str(round((Y[1])*100/sum(Y)))+'%\n'+
                      'X-: '+str(round((Y[2])*100/sum(Y)))+'%\n'+
                      'Y-: '+str(round((Y[3])*100/sum(Y)))+'%')
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
    df = Read_Tables(files,frequency,amplitude, sweeps = sweeps)                                  # frequency Hz, amplitude uA
    allresults = []
    if Fit:
        allresults.append(['Type', 'ID', 'angledrift', 'topness', 'sideness', 'amplitude'])
    for sp in sweep_colors[sweeps[0]]:
        if sp not in df.groupby('Cell_Type').groups.keys():
            continue
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
                    thetaR = list(dfcell['Cell_Rotation(deg)'])
                    thetaR, radii = fix_angles(thetaR, radii)
                    radii.append(radii[0])
                    thetaR.append(thetaR[0])
                    if len(thetaR)!=13:
                        radii = Plot_snowman(dfcell, files, sweep_colors, sweeps, grp, sp, axs[-1], cell)
                        max_angle = thetaR[radii.index(max(radii))]
                        allresults.append([sp, list(dfcell['Cell_ID'])[0], max_angle, np.nan, np.nan, np.nan])
                    else:
                        axs[-1].plot(thetaR,radii,sweep_colors[sweeps[grp]][sp],linewidth = 2, label = sweeps[0])
                        x,y,popt,pcov = getfitset(np.linspace(0, 2 * np.pi, 13), radii, sincircle)
                        if x is not None:
                            axs[-1].plot(thetaR,y,sweep_colors[sweeps[-1]][sp],linewidth = 2, label = sweeps[-1])
                            allresults.append([sp, list(dfcell['Cell_ID'])[0], popt[3], popt[0], popt[1], popt[2]])
                        axs[-1].set_yticks([])
                        axs[-1].set_title(cell)
                        axs[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                         fancybox=True, shadow=True, ncol=1)
                else:
                    if Theoretical is not None:
                        radii = Plot_snowman(dfcell, files, sweep_colors, sweeps, grp, sp, axs[-1], cell)
                        mean_scale = np.nanmean(radii)
                    else:
                        Plot_snowman(dfcell, files, sweep_colors, sweeps, grp, sp, axs[-1], cell)
            if Theoretical is not None:
                if len(radii)==12:
                    radiiR = list(Theoretical.get_group(cell).loc[:,'estimate'])[::3]
                    thetaR = list(Theoretical.get_group(cell).loc[:,'rotation']*np.pi/180)[::3]
                else:
                    radiiR = list(Theoretical.get_group(cell).loc[:,'estimate'])
                    thetaR = list(Theoretical.get_group(cell).loc[:,'rotation']*np.pi/180)
                thetaR, radiiR = fix_angles(thetaR, radiiR)
                radiiR.append(radiiR[0])
                thetaR.append(thetaR[0])
                radiiR = np.array(radiiR)*mean_scale/np.nanmean(radiiR)
                axs[-1].plot(thetaR,radiiR,sweep_colors['Theoretical'][sp],linewidth = 2, label = 'Theoretical')
                axs[-1].set_yticks([])
                axs[-1].set_title(cell)
                axs[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=1)  
                corr_mask = ~np.isnan(np.array(radiiR[:-1])) & ~np.isnan(np.array(radii))     
                arr_stack = np.vstack((np.array(radiiR)[:-1][corr_mask], np.array(radii)[corr_mask]))              
                correlations_per_cell.append(np.corrcoef(arr_stack)[0][1])
            elif Fit == False:
                if len(files)==2:
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

def Plot_sensitivity(allresults, quad = False, type_colors_A = type_colors_A, type_colors_B = type_colors_B):    
    df = pd.DataFrame(allresults[1:],columns = allresults[0])

    fig = plt.figure(figsize=(12,8))
    if df['topness'].isna().all():
        ax = fig.add_subplot(1,1,1)
        sns.stripplot(ax = ax, data=df, x='Type', y='angledrift', hue = 'Type', palette = type_colors_B, linewidth=1)
        Fix_Lamp_Label(ax)
        ax.set_ylabel('Peak amplitude angle (deg)')
    else:
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
        ax, yyys = Plot_sensitivity_histogram(dftype, bins, i+7, color, fig, yyys,sp, quad = quad)
    exc = sum(yyys[:3])
    inh = sum(yyys[3:])
    Y = yyys[-1]
    if quad:
        ax.set_xlabel('X+: '+str(round((Y[0])*100/sum(Y)))+'%\n'+
                      'Y+: '+str(round((Y[1])*100/sum(Y)))+'%\n'+
                      'X-: '+str(round((Y[2])*100/sum(Y)))+'%\n'+
                      'Y-: '+str(round((Y[3])*100/sum(Y)))+'%\n'+
                      'Exc X+: '+str(round((exc[0])*100/sum(exc)))+'%\n'+
                      'Exc Y+: '+str(round((exc[1])*100/sum(exc)))+'%\n'+
                      'Exc X-: '+str(round((exc[2])*100/sum(exc)))+'%\n'+
                      'Exc Y-: '+str(round((exc[3])*100/sum(exc)))+'%\n'+
                      'Inh X+: '+str(round((inh[0])*100/sum(inh)))+'%\n'+
                      'Inh Y+: '+str(round((inh[1])*100/sum(inh)))+'%\n'+
                      'Inh X-: '+str(round((inh[2])*100/sum(inh)))+'%\n'+
                      'Inh Y-: '+str(round((inh[3])*100/sum(inh)))+'%')
    else:
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
                        thetaR = list(dfcell['Cell_Rotation(deg)'])
                        thetaR, radii = fix_angles(thetaR, radii)
                        radii.append(radii[0])
                        thetaR.append(thetaR[0])
                        if len(thetaR)!=13:
                            max_angle = thetaR[radii.index(max(radii))]
                            allresults.append([sp, list(dfcell['Cell_ID'])[0], max_angle, np.nan, np.nan, np.nan])
                        else:
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

def Plot_phases_violin (files, sweeps, sweep_colors, frequency = 1, amplitude = 10):
    df = Read_Tables(files, frequency, amplitude, phase=True)

    df = df[df["Cell_Type"] != 'L56NP']
    df = df[df["Cell_Type"] != 'ITL6']

    dfb = pd.DataFrame({'Cell_Type': df["Cell_Type"],'Value': df["B"]})

    dfc = pd.DataFrame({'Cell_Type': df["Cell_Type"],'Value': df["C"]})

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1)

    df_melted = dfb.melt(id_vars=['Cell_Type'], var_name='Value', value_name='Phase')
    grouped_data = [df_melted[df_melted['Cell_Type'] == cell_type]['Phase'] for cell_type in dfb['Cell_Type'].unique()]
    violin_parts = ax.violinplot(dataset=grouped_data,side = 'low', showmeans=True, showextrema=True, showmedians=False, bw_method=0.3)

    for i, pc in enumerate(violin_parts['bodies']):
        cell_type = df_melted['Cell_Type'].unique()[i]
        color = type_colors_A[cell_type]
        pc.set_facecolor(color) 
        pc.set_edgecolor('black') 
        pc.set_alpha(0.7) 
    violin_parts['cbars'].set_edgecolors('black')
    violin_parts['cmins'].set_edgecolors('black')
    violin_parts['cmaxes'].set_edgecolors('black')
    violin_parts['cmeans'].set_edgecolors('black')

    df_melted = dfc.melt(id_vars=['Cell_Type'], var_name='Value', value_name='Phase')
    grouped_data = [df_melted[df_melted['Cell_Type'] == cell_type]['Phase'] for cell_type in dfc['Cell_Type'].unique()]
    violin_parts = ax.violinplot(dataset=grouped_data,side = 'high', showmeans=True, showextrema=True, showmedians=False, bw_method=0.3)

    for i, pc in enumerate(violin_parts['bodies']):
        cell_type = df_melted['Cell_Type'].unique()[i]
        color = type_colors_B[cell_type] 
        pc.set_facecolor(color) 
        pc.set_edgecolor('black') 
        pc.set_alpha(0.7) 
    violin_parts['cbars'].set_edgecolors('black')
    violin_parts['cmins'].set_edgecolors('black')
    violin_parts['cmaxes'].set_edgecolors('black')
    violin_parts['cmeans'].set_edgecolors('black')

    ax.set_xticks(range(1, len(dfb['Cell_Type'].unique()) + 1))
    ax.set_xticklabels(type_colors_B.keys())
    ax.set_xlabel('Cell Type', fontsize = 16)
    ax.set_ylabel('Phase', fontsize = 16)
    Fix_Lamp_Label(ax)
    plt.tight_layout()
    plt.show()

def Plot_phases (files, sweeps, sweep_colors, frequency = 1, amplitude = 10, maximal = False):
    df = Read_Tables(files, frequency, amplitude, phase=True, maximal = maximal)

    df = df[df["Cell_Type"] != 'L56NP']
    df = df[df["Cell_Type"] != 'ITL6']
    if maximal:
        dfb = df.loc[df.groupby('Cell_ID')['Amplitude_'+files[0]].idxmin()]
        dfc = df.loc[df.groupby('Cell_ID')['Amplitude_'+files[1]].idxmin()]
    else:
        dfb = df.copy()
        dfc = df.copy()
    fig = plt.figure(figsize=(14, 6))
    
    ax = fig.add_subplot(1, 2, 1)
    sns.swarmplot(ax = ax, data=dfb, x = 'Cell_Type', y = files[0], hue = 'Cell_Type', palette = type_colors_A,size = 2.3, order = type_colors_A.keys())
    ax.set_title(sweeps[0], fontsize = 18)
    ax.legend().set_visible(False)
    ax.set_ylabel('Phase (deg)')
    ax.set_ylim(0,360)
    Fix_Lamp_Label(ax)
        
    ax = fig.add_subplot(1, 2, 2)
    sns.swarmplot(ax = ax, data=dfc, x = 'Cell_Type', y = files[1], hue = 'Cell_Type', palette = type_colors_B,size = 2.3, order = type_colors_A.keys())
    ax.set_title(sweeps[1], fontsize = 18)
    ax.legend().set_visible(False)
    ax.set_ylabel('Phase (deg)')
    ax.set_ylim(0,360)
    Fix_Lamp_Label(ax)
    
    plt.tight_layout()
    plt.show()
    
    
def Compare_theoretical(cell_list, frequency = 1, amplitude = 10, axconf = 'NA_', 
                        files = [50, 100, 300, 500, 800, 1200, 1800, 2500, 2500],
                        crops = [ 'Point at 50 um', 'Point at 100 um', 'Point at 300 um', 
                                  'Point at 500 um', 'Point at 800 um', 'Point at 1200 um', 
                                  'Point at 1800 um', 'Point at 2500 um', 'Plane at 2500 um']):
    file_axon = '../Results/Result_Tables/PO_TH_' + axconf
    leg_colors = {'Point at 50 um': 'lightsalmon',
                  'Point at 100 um': 'coral',
                  'Point at 300 um': 'chocolate',
                  'Point at 500 um': 'peru',
                  'Point at 800 um': 'brown',
                  'Point at 1200 um': 'firebrick',
                  'Point at 1800 um': 'maroon',
                  'Point at 2500 um': 'red',
                  'Plane at 2500 um': 'blue'}
            
    for sp in type_colors_A:
        dft = cell_list.groupby('Cell_Type').get_group(sp)
        dfcells = dft.groupby('Cell_ID')

        theta = list(np.linspace(0,2*np.pi,36))
        theta.append(theta[0])
        fig, axs = Make_figure(dfcells, sp)
        for i, kk in enumerate(dfcells.groups.keys()):
            axs = Add_subplot(fig, dfcells, i, axis=axs)
            for grp in range(len(files)):
                if 'Plane' in crops[grp]:
                    estimate = pd.read_csv(file_axon.replace('PO','PL') + str(files[grp]) + '.csv').groupby('cell')
                else:
                    estimate = pd.read_csv(file_axon + str(files[grp]) + '.csv').groupby('cell')
                est = list(estimate.get_group(kk).sort_values(by=['rotation']).reset_index(drop = True).loc[:,'estimate'])
                est.append(est[0])
                axs[-1].plot(theta, est,leg_colors[crops[grp]],linewidth = 2, alpha = 0.5, label = crops[grp])
            axs[-1].set_yscale('log')
        Show_figure()
    fig, ax = plt.subplots(figsize=(5, 3))
    legend_handles = [mlines.Line2D([0], [0], color=color, lw=4) for color in leg_colors.values()]
    ax.legend(legend_handles, leg_colors.keys(), loc='center')
    ax.axis('off')
    plt.show()
    
def Plot_isi_distributions(target, results):
    tar = target.groupby('cell_type')
    res = results.groupby('ES_frequency(hz)').get_group(8).groupby('ES_current(uA)').get_group(0)
    res['Cell_Rotation(deg)'] = res['Cell_Rotation(deg)'].astype(int)
    res['Cell_ID'] = res['Cell_ID'].astype(int)
    res = res.groupby('Cell_Rotation(deg)').get_group(0).groupby('Cell_Type')

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()

    index_flag = 0
    for idx, ctype in enumerate(type_colors_A):
        if 'LAMP' in ctype:
            index_flag += 1
            continue
        means = list(tar.get_group(ctype)['Target'])
        stds = np.array(tar.get_group(ctype)['TargetStD'])/2
        isis = res.get_group(ctype)['ISIs']
        isi = []
        for i in isis:
            for j in i:
                isi.append(j)
        
        if 'ITL' in ctype:
            x = np.linspace(0, 55, 1000)
            rng = [0, 55]
        elif 'PVALB':
            x = np.linspace(0, 300, 1000)
            rng = [0, 250]
        else:
            x = np.linspace(0, 200, 1000)
            rng = [0, 200]
        
        overall_distribution = np.zeros_like(x)
        
        for mu, sigma in zip(means, stds):
            overall_distribution += norm.pdf(x, mu, sigma)
        overall_distribution  /= np.trapz(overall_distribution, x)
        axs[idx - index_flag].hist(isi, bins=40, range=rng, density=True, alpha=0.5, label = "Simulation spikes",
                      color=type_colors_B[ctype], edgecolor='black') 
        axs[idx - index_flag].plot(x, overall_distribution, label='Target distribution', color='k', linewidth=2)
        
        axs[idx - index_flag].grid(True)
        axs[idx - index_flag].set_xlim(rng)
        if 'IT' in ctype:
            axs[idx - index_flag].set_ylim(0,0.2)
        else:
            axs[idx - index_flag].set_ylim(0,0.034)
        axs[idx - index_flag].set_xlabel('Firing rate (Hz)')
        axs[idx - index_flag].set_ylabel('Number of spikes normalized')
        axs[idx - index_flag].set_title(ctype)
        axs[idx - index_flag].legend()

    plt.tight_layout()
    plt.show()
    
def Phase_Artifact (FiringRate,ESRate, bins = 15):
    PA = ((1/FiringRate)%(1/ESRate))*360/(1/ESRate)
    modes = 1
    phase = PA
    while phase%360>(360/bins):
        modes += 1
        phase += PA
    drift = phase%(360/bins)
    return ((1/FiringRate)%(1/ESRate))*360/(1/ESRate), modes, drift
    
    
def Plot_entrainment (data_frame, targets = None, MMR = False, subtract_ref = False, bins = 15):
    all_results = []
    if targets is not None:
        tar = targets.set_index('cell_id')[['PM', 'PS']]
    res = data_frame.copy()
    res['Cell_Rotation(deg)'] = res['Cell_Rotation(deg)'].astype(int)
    res['Cell_ID'] = res['Cell_ID'].astype(int)
    rotation  =  0 # deg
    gftex = res.groupby('Cell_Type')
    for typeex, gftexf in gftex:
        gfa = gftexf.groupby('Cell_Rotation(deg)').get_group(rotation)
        gfa = gfa.groupby('Cell_ID')
        for cell, gfc in gfa:
            fig = plt.figure(figsize = (8,6))
            if targets is not None:
                fig.suptitle(gfc['Cell_Type'].iloc[0] + ' ' + str(gfc['Cell_ID'].iloc[0]) +
                ' \nFiring rate: ' + str(round(tar.loc[cell,'PM'],2)) + ' +- ' + str(round(tar.loc[cell,'PS'],2)) + ' Hz', fontsize = 16)
            else:
                fig.suptitle(gfc['Cell_Type'].iloc[0] + ' ' + str(gfc['Cell_ID'].iloc[0]), fontsize = 16)
            color = type_colors_B[str(gfc['Cell_Type'].iloc[0])]
            for i,freq in enumerate([8,28,140]):
                for j,ampl in enumerate([0,10,100,500]):
                    ax = fig.add_subplot(3,4,1+j+i*4, projection='polar')
                    gft = gfc.groupby('ES_frequency(hz)').get_group(freq).groupby('ES_current(uA)').get_group(ampl)
                    if bins==15:
                        X, Y = gft['VL_X'].iloc[0], gft['VL_Y'].iloc[0]
                        if subtract_ref:
                            if ampl==0:
                                reference = Y
                            else:
                                Y -= reference
                    else:
                        Phases = gft['Phases'].iloc[0]
                        radians = np.asarray(Phases)*np.pi/180
                        mean_phase_rad = pcs.descriptive.mean(np.array(radians))
                        mean_phase_angle = mean_phase_rad*(180 / np.pi)
                        mean_pvalue_z=pcs.rayleigh(np.array(radians))
                        mean_vector_length = pcs.descriptive.vector_strength(np.array(radians))
                        Y, X = np.histogram(Phases, bins=np.linspace(0,360,bins+1))
                        Xp =(X[1:] + X[:-1]) / 2
                        X = Xp * np.pi / 180
                        Y = np.true_divide(Y, (np.max(Y)))
                        if subtract_ref:
                            if ampl==0:
                                reference = Y
                            else:
                                Y -= reference
                    bars = ax.bar(X, Y,  width=0.4, edgecolor = 'black', color=color, alpha=0.8, linewidth=1.2)
                    ax.set_axisbelow(True)
                    thetaticks = np.arange(0,360,90)
                    radius = [0,gft['mean_vector_length'].iloc[0]]
                    if MMR:
                        radius = [0,gft['MMR'].iloc[0]]
                        th = gft['MMR_X'].iloc[0][list(gft['MMR_Y'].iloc[0]).index(max(list(gft['MMR_Y'].iloc[0])))]
                        theta = [th,th]
                        ax.plot(theta,radius,type_colors_A[str(gfc['Cell_Type'].iloc[0])],linewidth = 2)
                        
                        ax.set_yticks([])
                        ax.set_xticks(np.pi/180. * np.linspace(45,  405, 2, endpoint=False))
                        ax.set_xlabel('MMR: {:.3f}'.format(gft['MMR'].iloc[0]) +  
                                          '\nAngle:  {:.3f}'.format(th*180/np.pi))
                    else:
                        theta = [gft['mean_phase_angle'].iloc[0]*np.pi/180,gft['mean_phase_angle'].iloc[0]*np.pi/180]
                        ax.plot(theta,radius,type_colors_A[str(gfc['Cell_Type'].iloc[0])],linewidth = 2)
                        ax.set_yticks([])
                        ax.set_xticks(np.pi/180. * np.linspace(45,  405, 2, endpoint=False))
                        ax.set_xlabel('VL: {:.3f}'.format(gft['mean_vector_length'].iloc[0]) + 
                                          '\nPval: {:.3f}'.format(gft['mean_pvalue_z'].iloc[0][0]) + 
                                          '\nAngle:  {:.3f}'.format(gft['mean_phase_angle'].iloc[0]))
                    if i==0:
                        ax.set_title(str(ampl) + ' uA', fontsize = 14)
                    if j==0:
                        ax.set_ylabel(str(freq) + ' Hz', fontsize = 14)
                    elif targets is not None and j==1:
                        PA, modes, drift = Phase_Artifact(tar.loc[cell,'PM'],freq, bins = bins)
                        ax.set_ylabel('PA: '+ str(round(PA,1)) + 'deg\nSteps: '+ str(modes) +'\n', fontsize = 14)
                        all_results.append([PA, modes, drift, gft['MMR'].iloc[0]])
            fig.tight_layout()
            if MMR:
                path = 'MMR/' 
            else:
                path = 'VL/'
            path += gfc['Cell_Type'].iloc[0] + '_' + str(gfc['Cell_ID'].iloc[0]) + '.png'
            plt.savefig('../Results/Result_Plots/Single_Cell_Entrainment_Roseplots/' + path)         
            
            plt.show()
            plt.close()
    return pd.DataFrame(all_results)
            
def Plot_entrainment_across (data_frame, show_all = True, mean = 'None', metric_tag = 'mean_vector_length'):
    
        
    dff = data_frame.groupby('Cell_Rotation(deg)').get_group('0')
    if metric_tag == 'ISIs':
        if mean == 'median':
            dff['ISIs'] = [np.median(i) for i in dff['ISIs']]
        else:
            dff['ISIs'] = [np.mean(i) for i in dff['ISIs']]
    for freq in [8,28,140]:
        df = dff.groupby('ES_frequency(hz)').get_group(freq)
        df = df.groupby('Cell_Type')
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, group in df:
            if mean == 'mean':
                x = ['0','10','100','500']
                y = list(group.groupby(['ES_current(uA)'])[metric_tag].mean())
                if not show_all:
                    x = [x[0],x[-1]]
                    y = [y[0],y[-1]]
                y = y-np.mean(y)
                ax.plot(x, y, label=name, color = type_colors_B[name])
                ax.set_ylabel('Mean '+ metric_tag, fontsize = 16)
            elif mean == 'median':
                x = ['0','10','100','500']
                y = list(group.groupby(['ES_current(uA)'])[metric_tag].median())
                if not show_all:
                    x = [x[0],x[-1]]
                    y = [y[0],y[-1]]
                y = y-np.mean(y)
                ax.plot(x, y, label=name, color = type_colors_B[name])
                ax.set_ylabel('Median '+ metric_tag, fontsize = 16)
            else:
                dfc = group.groupby('Cell_ID')
                leg_flag = 1
                for n, gr in dfc:
                    gr['ES_current(uA)'] = gr['ES_current(uA)'].astype(str)  # Correctly convert to string
                    if leg_flag:
                        ax.plot(gr['ES_current(uA)'], gr[metric_tag]-np.mean(gr[metric_tag]), label=name, color = type_colors_B[name])
                        leg_flag = 0
                    else:
                        ax.plot(gr['ES_current(uA)'], gr[metric_tag]-np.mean(gr[metric_tag]), label=None)
                ax.set_ylabel('All '+ metric_tag, fontsize = 16)
        if metric_tag == 'ISIs':
            ax.set_ylim(-5,5)
        ax.set_xlabel('ES_current(uA)', fontsize = 16)
        ax.set_title('ES_frequency: ' + str(freq) + ' Hz', fontsize = 16)
        ax.legend()
        plt.show()

def MMR (PhaseList):
    instances = np.array(np.round(PhaseList), dtype = int)
    bin_size = 36  
    gaussian = []
    for i in range(bin_size*2+1):
        gaussian.append(np.exp(-0.5*(i-bin_size)**2/(bin_size/4)**2)/(bin_size*np.sqrt(2*np.pi)/4))
    gaussian = np.array(gaussian)/gaussian[bin_size]
    waveshaped = []
    for i in instances:
        waveshape = np.zeros(360)
        indexes = np.arange(0,len(gaussian))
        indexes = indexes+i-bin_size
        for j in np.arange(0,len(gaussian)):
            if indexes[j]<0:
                indexes[j] +=360
            elif indexes[j]>359:
                indexes[j] -=360
        for j, ind in enumerate(indexes):
            waveshape[ind] = gaussian [j]
        waveshaped.append(waveshape)
    sumation = np.sum(waveshaped, axis = 0)
    sumation = sumation/np.max(sumation)
    return sumation, 1-np.mean(sumation)
        
def Plot_Displacement(time_displace = False, entrainment = False, preentrain = False, prestrength = 2, num_bins = 15):
    if preentrain:
        num_bins = 20

        fig, ax = plt.subplots(4,6,figsize = (14,11), subplot_kw={'projection': 'polar'})
        for b, phase in enumerate([2,4,6,8]):
            vls = []
            mmrs = []
            for s, spike_move in enumerate([0,10,20,30]):
                values = np.full(num_bins, 100)
                angles = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
                # preentrain
                bins_away = -(np.sin(angles)*prestrength).astype(int)
                spikes_moved = np.abs(np.sin(angles))*20
                for i in range(len(values)):
                    values[i] -= spikes_moved[i]
                    if bins_away[i]>0 and i<len(values)-bins_away[i]:
                        values[i+bins_away[i]] += spikes_moved[i]
                    elif bins_away[i]<0 and i>=bins_away[i]:
                        values[i+bins_away[i]] += spikes_moved[i]
                    elif bins_away[i]>0:
                        values[(i+bins_away[i])%len(values)] += spikes_moved[i]
                    elif bins_away[i]<0:
                        values[len(values)+i+bins_away[i]] += spikes_moved[i]
                values = np.roll(values,phase)
                if s>0:
                    bins_away = -(np.sin(angles)*4).astype(int)
                    spikes_moved = np.abs(np.sin(angles))*spike_move
                    for i in range(len(values)):
                        values[i] -= spikes_moved[i]
                        if bins_away[i]>0 and i<len(values)-bins_away[i]:
                            values[i+bins_away[i]] += spikes_moved[i]
                        elif bins_away[i]<0 and i>=bins_away[i]:
                            values[i+bins_away[i]] += spikes_moved[i]
                        elif bins_away[i]>0:
                            values[(i+bins_away[i])%len(values)] += spikes_moved[i]
                        elif bins_away[i]<0:
                            values[len(values)+i+bins_away[i]] += spikes_moved[i]
                Phases = []
                for i in range(len(angles)):
                    for j in range(values[i]):
                        Phases.append(angles[i]*180/np.pi)
                radians = np.asarray(Phases)*np.pi/180
                mean_phase_rad = pcs.descriptive.mean(np.array(radians))
                mean_phase_angle = mean_phase_rad*(180 / np.pi)
                mean_pvalue_z=pcs.rayleigh(np.array(radians))
                mean_vector_length = pcs.descriptive.vector_strength(np.array(radians))
                cd, mmr = MMR(Phases)
                ax[b][s].bar(angles, values, width=2 * np.pi / num_bins, bottom=0.0, color='b', edgecolor='black')
                ax[b][s].grid(False)
                ax[b][s].set_yticklabels([])
                if b==0:
                    ax[b][s].set_title('Case: '+str(s%4)+'\nRoll: '+str(phase)+'\nStrength: '+str(spike_move))
                ax[b][s].set_xlabel('VL: ' + str(round(mean_vector_length,3))+'\nMMR: ' + str(round(mmr,3)))
                vls.append(round(mean_vector_length,3))
                mmrs.append(round(mmr,3))
            ax[b][4].axis('off')
            ax[b][5].axis('off')
            ax1 = fig.add_subplot(4,3,3*(b+1))
            ax1.plot(vls, label = 'VL')
            ax1.plot(mmrs, label = 'MMR')
            ax1.set_xlabel('Case')
            ax1.set_ylabel('Length')
            ax1.legend()
        plt.tight_layout()
        plt.show()
    elif entrainment:
        num_bins = 20

        fig, ax = plt.subplots(2,6,figsize = (12,6), subplot_kw={'projection': 'polar'})
        for b, bin_move in enumerate([2,4]):
            vls = []
            mmrs = []
            for s, spike_move in enumerate([0,10,20,30]):
                values = np.full(num_bins, 100)
                angles = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
                bins_away = -(np.sin(angles)*bin_move).astype(int)
                spikes_moved = np.abs(np.sin(angles))*spike_move
                for i in range(len(values)):
                    values[i] -= spikes_moved[i]
                    if bins_away[i]>0 and i<len(values)-bins_away[i]:
                        values[i+bins_away[i]] += spikes_moved[i]
                    elif bins_away[i]<0 and i>=bins_away[i]:
                        values[i+bins_away[i]] += spikes_moved[i]
                    elif bins_away[i]>0:
                        values[(i+bins_away[i])%len(values)] += spikes_moved[i]
                    elif bins_away[i]<0:
                        values[len(values)+i+bins_away[i]] += spikes_moved[i]
                Phases = []
                for i in range(len(angles)):
                    for j in range(values[i]):
                        Phases.append(angles[i]*180/np.pi)
                radians = np.asarray(Phases)*np.pi/180
                mean_phase_rad = pcs.descriptive.mean(np.array(radians))
                mean_phase_angle = mean_phase_rad*(180 / np.pi)
                mean_pvalue_z=pcs.rayleigh(np.array(radians))
                mean_vector_length = pcs.descriptive.vector_strength(np.array(radians))
                cd, mmr = MMR(Phases)
                ax[b][s].bar(angles, values, width=2 * np.pi / num_bins, bottom=0.0, color='b', edgecolor='black')
                ax[b][s].grid(False)
                ax[b][s].set_yticklabels([])
                ax[b][s].set_xlabel('VL: ' + str(round(mean_vector_length,3))+'\nMMR: ' + str(round(mmr,3)))
                ax[b][s].set_title('Case: '+str(s%4)+'\nMove '+str(spike_move)+', '+str(bin_move)+' bins')
                vls.append(round(mean_vector_length,3))
                mmrs.append(round(mmr,3))
            ax[b][4].axis('off')
            ax[b][5].axis('off')
            ax1 = fig.add_subplot(2,3,3*(b+1))
            ax1.plot(vls, label = 'VL')
            ax1.plot(mmrs, label = 'MMR')
            ax1.set_xlabel('Case')
            ax1.set_ylabel('Length')
            ax1.legend()
        plt.tight_layout()
        plt.show()
    else:
        values = np.full(num_bins, 100)
        angles = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)

        fig, ax = plt.subplots(1,2,subplot_kw={'projection': 'polar'})
        ax[0].bar(angles, values, width=2 * np.pi / num_bins, bottom=0.0, color='b', edgecolor='black')
        ax[0].grid(False)
        ax[0].set_yticklabels([])
        ax[0].set_title('Control')

        if time_displace:
            bins_away = -(np.sin(angles)*3).astype(int)
            spikes_moved = np.abs(np.sin(angles))*30
            for i in range(len(values)):
                values[i] -= spikes_moved[i]
                if bins_away[i]>0 and i<len(values)-bins_away[i]:
                    values[i+bins_away[i]] += spikes_moved[i]
                elif bins_away[i]<0 and i>=bins_away[i]:
                    values[i+bins_away[i]] += spikes_moved[i]
                elif bins_away[i]>0:
                    values[(i+bins_away[i])%len(values)] += spikes_moved[i]
                elif bins_away[i]<0:
                    values[len(values)+i+bins_away[i]] += spikes_moved[i]
        else:
            values = np.full(num_bins, 100)+np.cos(angles)*40
        ax[1].bar(angles, values, width=2 * np.pi / num_bins, bottom=0.0, color='b', edgecolor='black')
        ax[1].grid(False)
        ax[1].set_yticklabels([])
        ax[1].set_title('With ES')
        plt.tight_layout()
        plt.show()
        
def Plot_MMR_vs_FR (grouped = False, bytype = False, annotate = False, point = False):
    if point:
        df = pd.read_pickle('../Results/Result_Tables/G.pkl')
        control = pd.read_pickle('../Results/Result_Tables/G.pkl').groupby('ES_current(uA)').get_group(0)
        max_amp = 0.2
    else:
        df = pd.read_pickle('../Results/Result_Tables/F.pkl')
        control = pd.read_pickle('../Results/Result_Tables/F.pkl').groupby('ES_current(uA)').get_group(0)
        max_amp = 500

    df['PM'] = 0
    targets = pd.read_csv('../Required_files/Calibrated.csv')
    tar = targets.set_index('cell_id')['PM']
    if bytype:
        for i in range(len(df)):
            df.iloc[i,-1] = tar.loc[int(df.iloc[i,1])]
        if grouped:
            fig, ax = plt.subplots(1,1, figsize = (14,4))
            df.loc[(df['Cell_Type'] == 'ITL46') & (df['ES_frequency(hz)']==8), 'PM'] = 61
            df.loc[(df['Cell_Type'] == 'ITL46') & (df['ES_frequency(hz)']==28), 'PM'] = 62
            df.loc[(df['Cell_Type'] == 'ITL46') & (df['ES_frequency(hz)']==140), 'PM'] = 63
            df.loc[(df['Cell_Type'] == 'ITL35') & (df['ES_frequency(hz)']==8), 'PM'] = 65
            df.loc[(df['Cell_Type'] == 'ITL35') & (df['ES_frequency(hz)']==28), 'PM'] = 66
            df.loc[(df['Cell_Type'] == 'ITL35') & (df['ES_frequency(hz)']==140), 'PM'] = 67
            df.loc[(df['Cell_Type'] == 'ITL23') & (df['ES_frequency(hz)']==8), 'PM'] = 69
            df.loc[(df['Cell_Type'] == 'ITL23') & (df['ES_frequency(hz)']==28), 'PM'] = 70
            df.loc[(df['Cell_Type'] == 'ITL23') & (df['ES_frequency(hz)']==140), 'PM'] = 71
            df.loc[(df['Cell_Type'] == 'PVALB') & (df['ES_frequency(hz)']==8), 'PM'] = 73
            df.loc[(df['Cell_Type'] == 'PVALB') & (df['ES_frequency(hz)']==28), 'PM'] = 74
            df.loc[(df['Cell_Type'] == 'PVALB') & (df['ES_frequency(hz)']==140), 'PM'] = 75
            df.loc[(df['Cell_Type'] == 'SST') & (df['ES_frequency(hz)']==8), 'PM'] = 77
            df.loc[(df['Cell_Type'] == 'SST') & (df['ES_frequency(hz)']==28), 'PM'] = 78
            df.loc[(df['Cell_Type'] == 'SST') & (df['ES_frequency(hz)']==140), 'PM'] = 79
            df.loc[(df['Cell_Type'] == 'VIP') & (df['ES_frequency(hz)']==8), 'PM'] = 81
            df.loc[(df['Cell_Type'] == 'VIP') & (df['ES_frequency(hz)']==28), 'PM'] = 82
            df.loc[(df['Cell_Type'] == 'VIP') & (df['ES_frequency(hz)']==140), 'PM'] = 83
            df.loc[:,'PM'] -= 59
            ax = [ax]
            all_results = []
        else:
            fig, ax = plt.subplots(4,1, figsize = (8,10))
        for i,freq in enumerate([0,8,28,140]):
            if grouped:
                i=0
            if freq==0:
                dfg = df.groupby('ES_current(uA)').get_group(0)
                dfg.loc[:,'PM'] -= 1
                dfg = dfg.groupby('ES_frequency(hz)').get_group(8).groupby('Cell_Type')
            else:
                dfg = df.groupby('ES_current(uA)').get_group(max_amp)
                dfg = dfg.groupby('ES_frequency(hz)').get_group(freq).groupby('Cell_Type')
            for name, dft in dfg:
                pf = dft.loc[dft.groupby('Cell_ID')['MMR'].idxmax()]
                if freq==8 or not grouped:
                    ax[i].plot(pf.loc[:,'PM']-1, pf.loc[:,'MMR'], marker = '.', linestyle = 'none',color=type_colors_A[name], label = name)
                else:
                    ax[i].plot(pf.loc[:,'PM']-1, pf.loc[:,'MMR'], marker = '.', linestyle = 'none',color=type_colors_A[name], label = None)
                if grouped:
                    for k in range(len(pf)):
                        all_results.append([name, pf['PM'].iloc[k], pf['MMR'].iloc[k]])
            # Add labels and title
            ax[i].set_xlabel('Firing rate (Hz)', fontsize = 16)
            ax[i].set_ylabel('MMR', fontsize = 16)
            ax[i].set_ylim(0,1)
            ax[i].set_title('ES Frequency ' + str(freq) + ' Hz', fontsize = 16)
            if grouped: 
                ax[i].set_xticks(np.arange(24))
                ax[i].set_xticklabels(['Control','8 Hz','28 Hz\n-- ITL46 --','140 Hz',
                                       'Control','8 Hz','28 Hz\n-- ITL35 --','140 Hz',
                                       'Control','8 Hz','28 Hz\n-- ITL23 --','140 Hz',
                                       'Control','8 Hz','28 Hz\n-- PVALB --','140 Hz',
                                       'Control','8 Hz','28 Hz\n-- SST --','140 Hz',
                                       'Control','8 Hz','28 Hz\n-- VIP --','140 Hz']) 
                ax[i].set_xlabel('State', fontsize = 16)
                ax[i].set_title(' ')
        if grouped:
            anot_res = []
            rf = pd.DataFrame(all_results, columns = ['Type','Group','MMR']).groupby('Group')
            for grp in np.arange(24)+1:
                A = np.array(rf.get_group(grp)['MMR'])
                q1 = np.percentile(A, 25)
                median = np.percentile(A, 50)
                q3 = np.percentile(A, 75)
                iqr = q3 - q1
                whisker_low = max(min(A), q1 - 1.5 * iqr)
                whisker_high = min(max(A), q3 + 1.5 * iqr)
                outliers = [d for d in A if d < whisker_low or d > whisker_high]
                box = plt.Rectangle((grp - 0.5 / 2-1, q1), 0.5, q3 - q1, color='k', lw=2, zorder=3)
                ax[0].add_patch(box)
                ax[0].plot([grp - 0.5 / 2-1, grp + 0.5 / 2-1], [median, median], color='k', lw=2, zorder=4)
                ax[0].plot([grp-1, grp-1], [whisker_low, q1], color='k', lw=2, zorder=2)
                ax[0].plot([grp-1, grp-1], [q3, whisker_high], color='k', lw=2, zorder=2)
                anot_res.append([grp, whisker_low, q1, median, q3, whisker_high])
            if annotate:
                test_results = add_stat_annotation(ax[0], data=pd.DataFrame(all_results, columns = ['Type','Group','MMR']), x='Group', y='MMR',
                                               box_pairs=[(1,2), (1,3), (1,4), (5,6), (5,7), (5,8), (9,10), (9,11), (9,12),
                                                          (13,14), (13,15), (13,16), (17,18), (17,19), (17,20), (21,22), (21,23), (21,24)],
                                               test='Mann-Whitney', text_format='star',
                                               loc='outside')

        else:
            ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=6)
            
        plt.tight_layout()
        plt.show()
    else:
        for i in range(len(df)):
            df.iloc[i,-1] = tar.loc[int(df.iloc[i,1])]
        if grouped:
            fig, ax = plt.subplots(1,1, figsize = (10,4))
            df.loc[(df['PM'] > 60) & (df['ES_frequency(hz)']==8), 'PM'] = 69
            df.loc[(df['PM'] > 60) & (df['ES_frequency(hz)']==28), 'PM'] = 70
            df.loc[(df['PM'] > 60) & (df['ES_frequency(hz)']==140), 'PM'] = 71
            df.loc[(df['PM'] <= 20) & (df['ES_frequency(hz)']==8), 'PM'] = 61
            df.loc[(df['PM'] <= 20) & (df['ES_frequency(hz)']==28), 'PM'] = 62
            df.loc[(df['PM'] <= 20) & (df['ES_frequency(hz)']==140), 'PM'] = 63
            df.loc[(df['PM'] <= 60) & (df['ES_frequency(hz)']==8), 'PM'] = 65
            df.loc[(df['PM'] <= 60) & (df['ES_frequency(hz)']==28), 'PM'] = 66
            df.loc[(df['PM'] <= 60) & (df['ES_frequency(hz)']==140), 'PM'] = 67
            df.loc[:,'PM'] -= 59
            ax = [ax]
            all_results = []
        else:
            fig, ax = plt.subplots(4,1, figsize = (8,10))
        for i,freq in enumerate([0,8,28,140]):
            if grouped:
                i=0
            if freq==0:
                dfg = df.groupby('ES_current(uA)').get_group(0)
                dfg.loc[:,'PM'] -= 1
                dfg = dfg.groupby('ES_frequency(hz)').get_group(8).groupby('Cell_Type')
            else:
                dfg = df.groupby('ES_current(uA)').get_group(max_amp)
                dfg = dfg.groupby('ES_frequency(hz)').get_group(freq).groupby('Cell_Type')
            for name, dft in dfg:
                pf = dft.loc[dft.groupby('Cell_ID')['MMR'].idxmax()]
                if freq==8 or not grouped:
                    ax[i].plot(pf.loc[:,'PM']-1, pf.loc[:,'MMR'], marker = '.', linestyle = 'none',color=type_colors_A[name], label = name)
                else:
                    ax[i].plot(pf.loc[:,'PM']-1, pf.loc[:,'MMR'], marker = '.', linestyle = 'none',color=type_colors_A[name], label = None)
                if grouped:
                    for k in range(len(pf)):
                        all_results.append([name, pf['PM'].iloc[k], pf['MMR'].iloc[k]])
            # Add labels and title
            ax[i].set_xlabel('Firing rate (Hz)', fontsize = 16)
            ax[i].set_ylabel('MMR', fontsize = 16)
            ax[i].set_ylim(0,1)
            ax[i].set_title('ES Frequency ' + str(freq) + ' Hz', fontsize = 16)
            if grouped: 
                ax[i].set_xticks(np.arange(12))
                ax[i].set_xticklabels(['Control','8 Hz','28 Hz\n-- FR 0-20 Hz --','140 Hz',
                                       'Control','8 Hz','28 Hz\n-- FR 20-60 Hz --','140 Hz',
                                       'Control','8 Hz','28 Hz\n-- FR 60+ Hz --','140 Hz']) 
                ax[i].set_xlabel('State', fontsize = 16)
                ax[i].set_title(' ')
        if grouped and annotate:
            anot_res = []
            rf = pd.DataFrame(all_results, columns = ['Type','Group','MMR']).groupby('Group')
            for grp in np.arange(12)+1:
                A = np.array(rf.get_group(grp)['MMR'])
                q1 = np.percentile(A, 25)
                median = np.percentile(A, 50)
                q3 = np.percentile(A, 75)
                iqr = q3 - q1
                whisker_low = max(min(A), q1 - 1.5 * iqr)
                whisker_high = min(max(A), q3 + 1.5 * iqr)
                outliers = [d for d in A if d < whisker_low or d > whisker_high]
                box = plt.Rectangle((grp - 0.5 / 2-1, q1), 0.5, q3 - q1, color='k', lw=2, zorder=3)
                ax[0].add_patch(box)
                ax[0].plot([grp - 0.5 / 2-1, grp + 0.5 / 2-1], [median, median], color='k', lw=2, zorder=4)
                ax[0].plot([grp-1, grp-1], [whisker_low, q1], color='k', lw=2, zorder=2)
                ax[0].plot([grp-1, grp-1], [q3, whisker_high], color='k', lw=2, zorder=2)
                anot_res.append([grp, whisker_low, q1, median, q3, whisker_high])
            test_results = add_stat_annotation(ax[0], data=pd.DataFrame(all_results, columns = ['Type','Group','MMR']), x='Group', y='MMR',
                                               box_pairs=[(1,2), (1,3), (1,4), (5,6), (5,7), (5,8), (9,10), (9,11), (9,12)],
                                               test='Mann-Whitney', text_format='star',
                                               loc='outside')
        else:
            ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=6)

        plt.tight_layout()
        plt.show()
        
def find_first_peak(trace, after = 0):
    peaks, _ = find_peaks(trace)
    if len(peaks) > 0:
        for i in peaks:
            if (i>after) and (i<len(trace)):
                return i
    return 0
def shift_fill(arr, shift):
    if shift > 0:
        return np.concatenate(([arr[0]] * shift, arr[:-shift]))
    elif shift < 0:
        return np.concatenate((arr[-shift:], [arr[0]] * (-shift)))
    else:
        return arr
def align_to_reference(trace, trace_peak, ref_peak):
    shift = ref_peak - trace_peak
    return shift_fill(trace, shift)[475:575]     
        
def Plot_custom_axon_spikes(Manual_picked_model_list, plot_axon = False, plot_spikes = True, hof = False):
    fig, ax = plt.subplots(3,6,figsize=(14, 10))
    ax = ax.flatten()
    all_axon_result = []
    t = pd.read_pickle('../Results/Result_Tables/H.pkl')['Cell_Type']
    c = pd.read_pickle('../Results/Result_Tables/H.pkl')['Cell_ID']
    if hof:
        h = pd.read_pickle('../Results/Result_Tables/H.pkl')['Trace(mV)']
        i = pd.read_pickle('../Results/Result_Tables/N.pkl').groupby('HOF').get_group(1)['Trace(mV)']
        j = pd.read_pickle('../Results/Result_Tables/N.pkl').groupby('HOF').get_group(2)['Trace(mV)']
        k = pd.read_pickle('../Results/Result_Tables/N.pkl').groupby('HOF').get_group(3)['Trace(mV)']
        l = pd.read_pickle('../Results/Result_Tables/N.pkl').groupby('HOF').get_group(4)['Trace(mV)']
        m = pd.read_pickle('../Results/Result_Tables/N.pkl').groupby('HOF').get_group(5)['Trace(mV)']
        ia = pd.read_pickle('../Results/Result_Tables/N.pkl').groupby('HOF').get_group(1)['Traces(mV)']
        ja = pd.read_pickle('../Results/Result_Tables/N.pkl').groupby('HOF').get_group(2)['Traces(mV)']
        ka = pd.read_pickle('../Results/Result_Tables/N.pkl').groupby('HOF').get_group(3)['Traces(mV)']
        la = pd.read_pickle('../Results/Result_Tables/N.pkl').groupby('HOF').get_group(4)['Traces(mV)']
        ma = pd.read_pickle('../Results/Result_Tables/N.pkl').groupby('HOF').get_group(5)['Traces(mV)']
    else:
        h = pd.read_pickle('../Results/Result_Tables/H.pkl')['Trace(mV)']
        i = pd.read_pickle('../Results/Result_Tables/I.pkl')['Trace(mV)']
        j = pd.read_pickle('../Results/Result_Tables/J.pkl')['Trace(mV)']
        k = pd.read_pickle('../Results/Result_Tables/K.pkl')['Trace(mV)']
        l = pd.read_pickle('../Results/Result_Tables/L.pkl')['Trace(mV)']
        m = pd.read_pickle('../Results/Result_Tables/M.pkl')['Trace(mV)']
        ia = pd.read_pickle('../Results/Result_Tables/I.pkl')['Traces(mV)']
        ja = pd.read_pickle('../Results/Result_Tables/J.pkl')['Traces(mV)']
        ka = pd.read_pickle('../Results/Result_Tables/K.pkl')['Traces(mV)']
        la = pd.read_pickle('../Results/Result_Tables/L.pkl')['Traces(mV)']
        ma = pd.read_pickle('../Results/Result_Tables/M.pkl')['Traces(mV)']
    cnt = 0
    all_files = zip(t,c,h,i,j,k,l,m,ia,ja,ka,la,ma)
    for t,c,h,i,j,k,l,m,ia,ja,ka,la,ma in all_files:
        if int(c) in Manual_picked_model_list:
            ref_peak = find_first_peak(h.flatten())
            h = align_to_reference(h, ref_peak, 500)
            to_zip = [i.flatten(),j.flatten(),k.flatten(),l.flatten(),m.flatten()]
            peaks = [find_first_peak(trace) for trace in to_zip]
            aligned_traces = [align_to_reference(trace, peak, 500) for trace, peak in zip(to_zip, peaks)]
            if plot_spikes:
                if hof:
                    titles = ['HoF 1','HoF 2','HoF 3','HoF 4','HoF 5']
                else:
                    titles = ['Passive axon\ncm = 4','Passive axon\ncm = 1','Passive axon\ncm = 0.5',
                                        'Active axon dendrite\nparameters','Active axon soma\nparameters']
                for title, data in zip(titles, aligned_traces):
                    ax[cnt].plot(data,  color = custom_colors[title], linewidth = 2, alpha = 0.8, label=title)
            aligned_traces = []
            for axon_segment in range(len(ia.T)):
                to_zip = [ia.T[axon_segment], ja.T[axon_segment], ka.T[axon_segment], 
                          la.T[axon_segment], ma.T[axon_segment]]
                aligned_traces = [align_to_reference(trace, peak, 500) for trace, peak in zip(to_zip, peaks)]
                if hof:
                    titles = ['HoF 1','HoF 2','HoF 3','HoF 4','HoF 5']
                else:
                    titles = ['Passive axon\ncm = 4','Passive axon\ncm = 1','Passive axon\ncm = 0.5',
                                        'Active axon dendrite\nparameters','Active axon soma\nparameters']

                for title, data in zip(titles, aligned_traces):
                    instantaneus_peak = find_first_peak(data, after = 25)
                    if instantaneus_peak!=0 and plot_axon:
                        ax[cnt].plot(instantaneus_peak, data[instantaneus_peak], 
                                 color = custom_colors[title], marker = '.', 
                                 linestyle = 'none', alpha = 0.6) 
                    all_axon_result.append([t,c,title,axon_segment,instantaneus_peak-25,data[instantaneus_peak]])
            if hof:
                ax[cnt].plot(h, color = 'k', linewidth = 2, alpha = 0.8, label = 'HoF[0]')
            else:
                ax[cnt].plot(h, color = 'k', linewidth = 2, alpha = 0.8, label = 'Active optimized\nwithout')
            ax[cnt].set_title(str(t)+'\n' + str(c))
            ax[cnt].spines[['right', 'left', 'bottom', 'top']].set_visible(False)
            ax[cnt].tick_params(right=False, left=False, bottom=False, top=False,
                                labelleft=False, labelbottom=False)
            if cnt==16:
                ax[cnt].legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
                cnt+=1
                fig.delaxes(ax[cnt])
            cnt+=1
    plt.show()
    return all_axon_result
        
def Plot_morphologies (files = 'Custom', calibrated = True):
    swc_path = '../Required_Files/Models/SWCs connected XXX/'.replace('XXX', files)
    morphfiles = os.listdir(swc_path)
    current_vals = pd.read_csv('../Required_files/Calibrated.csv')
    current_vals = current_vals.sort_values('Target', ascending=True).reset_index(drop=True).reset_index()
    current_vals = current_vals.set_index('cell_id')
    figcnt = 0
    fig, ax = plt.subplots(1,5, figsize = (12,3))
    toplot = []
    for i in morphfiles:
        if int(i.split('_')[2]) not in current_vals.index and calibrated:
            continue
        if 'rmed_0'not in i:
            continue
        toplot.append(i)
    for i in toplot:
        morph = neurom.load_morphology(swc_path+i)
        viewer.plot_morph(morph, ax = ax[figcnt])
        ax[figcnt].set_aspect('equal')
        ax[figcnt].set_xlim(-500,500)
        ax[figcnt].set_ylim(-1500,1500)
        ax[figcnt].set_axis_off()
        ax[figcnt].set_title(ax[figcnt].get_title().replace('_transformed_0.swc','').replace('__','\n'))
        figcnt += 1
        if figcnt>4 and i!=toplot[-1]:
            plt.show()
            figcnt = 0
            fig, ax = plt.subplots(1,5, figsize = (10,3))
    plt.show()

def Plot_axon_propagation (all_axon_result):
    fig, ax = plt.subplots(6,6,figsize=(14, 14))
    ax = ax.flatten()  
    df = pd.DataFrame(all_axon_result, columns = ['Type','Cell','Axon type','Axon segment',
                                                  'Delay (*0.05ms)','Amplitude (mV)'])
    df.loc[df['Delay (*0.05ms)']==-25,'Delay (*0.05ms)'] = np.nan
    df.dropna(inplace = True)
    df = df.groupby('Type')
    cnt = 0
    for t in type_colors_A:
        dfg = df.get_group(t).groupby('Cell')
        for cell, cgroup in dfg:
            gfg = cgroup.groupby('Axon type')
            for axon, agroup in gfg:
                ax[cnt].plot(agroup['Axon segment']*20, agroup['Amplitude (mV)'], 
                             color = custom_colors[axon], marker = '.', linestyle = 'none')
                ax[cnt].set_title(agroup.iloc[0,0])
                ax[cnt].set_xlabel('Path distance (um)')
                ax[cnt].set_ylabel('Amplitude (mV)')
                ax[cnt].set_xlim(20,2000)
                ax[cnt].set_xscale('log')
            cnt+=1
            ax[cnt].axhline(y=1, color='k', linestyle='--')
            ax[cnt].axvline(x=100, color='k', linestyle='--')
            for axon, agroup in gfg:
                ax[cnt].plot(agroup['Axon segment']*20, agroup['Delay (*0.05ms)']*0.05, 
                             color = custom_colors[axon], label = axon, marker = '.', linestyle = 'none')
                ax[cnt].set_title(str(cell))
                ax[cnt].set_xlabel('Path distance (um)')
                ax[cnt].set_ylabel('Delay (ms)')
                ax[cnt].set_xlim(20,2000)
                ax[cnt].set_xscale('log')
            if cnt==33:
                ax[cnt].legend(loc='center left',ncols = 2, bbox_to_anchor=(1.15, 0.5))
                cnt+=1
                fig.delaxes(ax[cnt])
                cnt+=1
                fig.delaxes(ax[cnt])
            cnt+=1

    plt.subplots_adjust(wspace=0.5, hspace=0.75)
    plt.show()    
  

def find_local_maxima(arr):
    local_maxima_indices = []
    if len(arr) < 3:
        return local_maxima_indices
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            local_maxima_indices.append(i)
    return local_maxima_indices
  
def plot_HOF_spikes(file = 'O', es_amp = -100, clean = True):
    all_results = []
    nscnt = 0
    s1cnt = 0
    mscnt = 0
    bscnt = 0
    df = pd.concat([pd.read_pickle('../Results/Result_Tables/' + file + '_A.pkl'), 
                    pd.read_pickle('../Results/Result_Tables/' + file + '_B.pkl'),
                    pd.read_pickle('../Results/Result_Tables/' + file + '_C.pkl')], axis=0).reset_index(drop = True)

    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure(figsize = (14,3))
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1.*i/20) for i in range(20)])
    for i in range(20):
        ax.plot(np.arange(10)*(i+1), label = 'HOF: '+str(i))
    ax.set_axis_off()
    ax.legend(ncols = 4)
    plt.show()

    dfgrp = df.groupby('Cell_ID')
    for name, dfg in dfgrp:
        temp_results = [dfg.iloc[0,0],name, es_amp]
        dfg = dfg.groupby('ES_current(uA)').get_group(es_amp)
        fig, ax = plt.subplots(1,1, figsize = (14,4))
        ax.set_prop_cycle(color=[cm(1.*i/20) for i in range(20)])
        dfgh = dfg.groupby('HOF')
        for hof, dfh in dfgh:
            data = np.array(dfh['Trace(mV)'])[0]
            
            good_model = False
            hof_peaks = find_local_maxima(data[90:490])

            if len(hof_peaks)==1 and hof_peaks[0] == 10:
                ax.plot(data)
                data = data[90:490]
                temp_results.append([hof,'NS',data[10]-data[0]])
                nscnt += 1
            elif len(hof_peaks)==2 and hof_peaks[0] == 10:
                if data[90+hof_peaks[1]]>0:
                    ax.plot(data)
                    data = data[90:490]
                    temp_results.append([hof,'1S',data[10]-data[0],hof_peaks[1],data[hof_peaks[1]]-data[0]])
                    s1cnt += 1
                else:
                    ax.plot(data)
                    data = data[90:490]
                    temp_results.append([hof,'NS',data[10]-data[0]])
                    nscnt += 1
            elif len(hof_peaks)>0 and hof_peaks[0] == 10:
                temp_array = [hof,'MS',data[101]-data[90]]
                for i in range(len(hof_peaks)-1):
                    temp_array.append(hof_peaks[i+1])
                    temp_array.append(data[90+hof_peaks[i+1]]-data[90])
                temp_results.append(temp_array)
                if clean == False:
                    ax.plot(data)
                    data = data[90:490]
                mscnt += 1
            elif len(hof_peaks)>0:
                temp_array = [hof,'BS']
                for i in range(len(hof_peaks)):
                    temp_array.append(hof_peaks[i])
                    temp_array.append(data[90+hof_peaks[i]]-data[90])
                temp_results.append(temp_array)
                if clean == False:
                    ax.plot(data)
                    data = data[90:490]
                bscnt += 1
        all_results.append(temp_results)
        ax.set_title(str(dfg.iloc[0,0])+ '_'+str(name))
        plt.show()
    if clean:
        data = pd.DataFrame(all_results)
        Passive_spikes = []
        Active_spikes = []
        for i in range(len(data)):
            for j in range(20):
                spikes = data.iloc[i,j+3]
                if spikes is not None:
                    if spikes[1] == 'NS':
                        Passive_spikes.append([data.iloc[i,0], data.iloc[i,1], j, spikes[2]])
                    elif spikes[1] == '1S':
                        Passive_spikes.append([data.iloc[i,0], data.iloc[i,1], j, spikes[2]])
                        Active_spikes.append([data.iloc[i,0], data.iloc[i,1], j, (spikes[3]-10)*0.05, spikes[4]]) #convert to ms
        return Passive_spikes, Active_spikes, nscnt, s1cnt, mscnt, bscnt

def Plot_Soma_stats (Passive_spikes, Active_spikes):
    df = pd.DataFrame(Passive_spikes, columns=['Type', 'Cell', 'HoF', 'Passive peak amplitude (mV)'])
    df = df[~df['Type'].isin(['L56NP','ITL6'])]
    df['Passive peak amplitude (mV)'] = [i[0] for i in df['Passive peak amplitude (mV)']]
    ax = sns.swarmplot(data=df, x='Type', y='Passive peak amplitude (mV)', hue='Cell', size = 1)
    ax.legend().set_visible(False)
    ax.set_title('Passive peak amplitude (mV)', fontsize = 16)
    Fix_Lamp_Label(ax)
    plt.show()
    df = pd.DataFrame(Active_spikes, columns=['Type', 'Cell', 'HoF', 'Propagation delay (ms)', 'Spike amplitude (mV)'])
    df['Spike amplitude (mV)'] = [i[0] for i in df['Spike amplitude (mV)']]
    ax = sns.swarmplot(data=df, x='Type', y='Spike amplitude (mV)', hue='Cell', size = 1)
    ax.legend().set_visible(False)
    ax.set_title('Spike amplitude (mV)', fontsize = 16)
    Fix_Lamp_Label(ax)
    plt.show()
    ax = sns.swarmplot(data=df, x='Type', y='Propagation delay (ms)', hue='Cell', size = 1)
    ax.legend().set_visible(False)
    ax.set_title('Propagation delay (ms)', fontsize = 16)
    Fix_Lamp_Label(ax)
    plt.show()
    
def Plot_consistency (Passive_spikes, Active_spikes):
    df = pd.DataFrame(Active_spikes)
    cell_list = list(set([i[0]+'_'+i[1] for i in Passive_spikes]))
    do_it = []
    for i in range(len(cell_list)):
        do_it.append(np.zeros(21))
    do_it = pd.DataFrame(do_it).astype(int)
    do_it.iloc[:,0] = sorted(cell_list)
    do_it.set_index(0, inplace = True)
    for active in range(len(df)):
        for row in do_it.index:
            if str(row).split('_')[1]==str(df.iloc[active,1]):
                do_it.loc[row,1+int(df.iloc[active,2])] = 1
    plt.figure(figsize=(8, 50))
    cmap = ListedColormap(['red', 'green'])
    ax = sns.heatmap(do_it, cmap=ListedColormap(['red', 'green']), cbar=False, linewidths=0.5, linecolor='black')
    ax.set_xlabel('HoF model', fontsize = 16)
    ax.set_ylabel('Cell ID', fontsize = 16)
    plt.show()
    
def plot_dendrite_propagation(file = 'O', es_amp = -100, clean = True):
    all_results = []
    df = pd.concat([pd.read_pickle('../Results/Result_Tables/' + file + '_A.pkl'), 
                    pd.read_pickle('../Results/Result_Tables/' + file + '_B.pkl'), 
                    pd.read_pickle('../Results/Result_Tables/' + file + '_C.pkl')], axis=0).reset_index(drop = True)
    for subfile in "DEFGH":
        try:
            df = pd.concat([df, 
                pd.read_pickle('../Results/Result_Tables/' + file + '_'+subfile+'.pkl')], axis=0).reset_index(drop = True)
        except:
            break
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure(figsize = (14,3))
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1.*i/20) for i in range(20)])
    for i in range(20):
        ax.plot(np.arange(10)*(i+1), label = 'HOF: '+str(i))
    ax.set_axis_off()
    ax.legend(ncols = 4)
    plt.show()

    dfgrp = df.groupby('Cell_ID')
    for name, dfg in dfgrp:
        temp_results = [dfg.iloc[0,0],name, es_amp]
        dfg = dfg.groupby('ES_current(uA)').get_group(es_amp)
        dfgh = dfg.groupby('HOF')
        fig = plt.figure(figsize = (14,10))
        gs = fig.add_gridspec(12,6)
        ax = []
        ax.append(fig.add_subplot(gs[:5, 3:]))
        ax.append(fig.add_subplot(gs[5:, 3:]))
        ax[0].set_prop_cycle(color=[cm(1.*i/20) for i in range(20,0,-1)])
        ax[1].set_prop_cycle(color=[cm(1.*i/20) for i in range(20,0,-1)])
        fig.suptitle(str(dfg.iloc[0,0])+'_'+name)
        fig.supylabel(str('Distance from soma (um)'), fontsize = 16, x = 0.15)
        for hof, dfh in reversed(list(dfgh)):
            data = pd.DataFrame(np.array(dfh['Traces(mV)'])[0])
            for segm in range(10):
                if hof==19:
                    if segm:
                        ax.append(fig.add_subplot(gs[segm+1, 1:3]))
                    else:
                        ax.append(fig.add_subplot(gs[segm+1, 1:3]))
                    ax[segm+2].set_prop_cycle(color=[cm(1.*i/20) for i in range(20,0,-1)])
                ax[segm+2].tick_params(left=False, right=False, labelleft=True, labelbottom=False) 
                ax[segm+2].spines['left'].set_visible(False) 
                ax[segm+2].spines['bottom'].set_visible(False)
                ax[segm+2].spines['right'].set_visible(False) 
                ax[segm+2].spines['top'].set_visible(False)
                ax[segm+2].set_xticks([])
                ax[segm+2].set_yticks([])
                ax[segm+2].set_ylabel(str(segm*200+30))
                ax[segm+2].plot(data.iloc[100:400,segm*10]+100, alpha = 0.6, linewidth = 2)
            max_values = data.max(axis=0)
            max_index = data.idxmax(axis=0).where(data.idxmax(axis=0) > 100, np.nan).replace(499, np.nan)
            ax[0].plot(np.arange(len(max_values))*20+30, max_values, alpha = 0.6, linewidth = 2)
            ax[0].set_xlabel('Distance from soma (um)', fontsize = 16)
            ax[0].set_ylabel('Peak amplitude (mV)', fontsize = 16)
            ax[1].plot(np.arange(len(max_index))*20+30, (max_index-100)*0.05, alpha = 0.6, linewidth = 2)
            ax[1].set_xlabel('Distance from soma (um)', fontsize = 16)
            ax[1].set_ylabel('Peak delay (ms)', fontsize = 16)
        ax.append(fig.add_subplot(gs[:, 0]))
        morph = neurom.load_morphology('../Required_Files/Models/Stick.swc')
        viewer.plot_morph(morph, ax = ax[-1], plane = 'xy')
        ax[-1].set_aspect('equal')
        ax[-1].set_ylim(-2001,1)
        ax[-1].set_xlim(-50,10)
        ax[-1].plot(-25,-1250, marker = 'x')
        ax[-1].set_axis_off()
        fig.tight_layout()
        plt.show()
        
def plot_dendrite_propagation_vs_dL(file = 'OTU', dL = [20,40,10], es_amp = -100, clean = True):
    all_results = []
    dframes = []
    for i in file:
        df = pd.concat([pd.read_pickle('../Results/Result_Tables/' + i + '_A.pkl'), 
                        pd.read_pickle('../Results/Result_Tables/' + i + '_B.pkl'), 
                        pd.read_pickle('../Results/Result_Tables/' + i + '_C.pkl')], axis=0).reset_index(drop = True)
        for subfile in "DEFGH":
            try:
                df = pd.concat([df, 
                    pd.read_pickle('../Results/Result_Tables/' + i + '_'+subfile+'.pkl')], axis=0).reset_index(drop = True)
            except:
                break
        dframes.append(df)
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure(figsize = (14,3))
    ax = fig.add_subplot(111)
    for i in range(len(dL)):
        ax.plot(np.arange(10)*(i+1), label = 'dL = '+str(dL[i]))
    ax.set_axis_off()
    ax.legend(ncols = 4)
    plt.show()
    dfgrp = dframes[0].groupby('Cell_ID')
    for name, dfg in dfgrp:
        fig = plt.figure(figsize = (14,10))
        gs = fig.add_gridspec(12,6)
        ax = []
        ax.append(fig.add_subplot(gs[:5, 3:]))
        ax.append(fig.add_subplot(gs[5:, 3:]))
        for sweep, df in enumerate(dframes):
            dfgrp = df.groupby('Cell_ID')
            dfg = dfgrp.get_group(name)
            temp_results = [dfg.iloc[0,0],name, es_amp]
            dfg = dfg.groupby('ES_current(uA)').get_group(es_amp)
            dfh = dfg.groupby('HOF').get_group(0)
            fig.suptitle(str(dfg.iloc[0,0])+'_'+name)
            fig.supylabel(str('Distance from soma (um)'), fontsize = 16, x = 0.15)
            hof = 0
            data = pd.DataFrame(np.array(dfh['Traces(mV)'])[0])
            for segm in range(10):
                if not sweep:
                    ax.append(fig.add_subplot(gs[segm+1, 1:3]))
                ax[segm+2].tick_params(left=False, right=False, labelleft=True, labelbottom=False) 
                ax[segm+2].spines['left'].set_visible(False) 
                ax[segm+2].spines['bottom'].set_visible(False)
                ax[segm+2].spines['right'].set_visible(False) 
                ax[segm+2].spines['top'].set_visible(False)
                ax[segm+2].set_xticks([])
                ax[segm+2].set_yticks([])
                ax[segm+2].set_ylabel(str(segm*200+30))
                ax[segm+2].plot(data.iloc[100:400,segm*int(200/dL[sweep])]+100, alpha = 0.6, linewidth = 2)
            max_values = data.max(axis=0)
            max_index = data.idxmax(axis=0).where(data.idxmax(axis=0) > 100, np.nan).replace(499, np.nan)
            ax[0].plot(np.arange(len(max_values))*dL[sweep]+30, max_values, alpha = 0.6, linewidth = 2)
            ax[0].set_xlabel('Distance from soma (um)', fontsize = 16)
            ax[0].set_ylabel('Peak amplitude (mV)', fontsize = 16)
            ax[1].plot(np.arange(len(max_index))*dL[sweep]+30, (max_index-100)*0.05, alpha = 0.6, linewidth = 2)
            ax[1].set_xlabel('Distance from soma (um)', fontsize = 16)
            ax[1].set_ylabel('Peak delay (ms)', fontsize = 16)
        ax.append(fig.add_subplot(gs[:, 0]))
        morph = neurom.load_morphology('../Required_Files/Models/Stick.swc')
        viewer.plot_morph(morph, ax = ax[-1], plane = 'xy')
        ax[-1].set_aspect('equal')
        ax[-1].set_ylim(-2001,1)
        ax[-1].set_xlim(-50,10)
        ax[-1].plot(-25,-1250, marker = 'x')
        ax[-1].set_axis_off()
        fig.tight_layout()
        plt.show()
        
def plot_custom_dendrite_propagation(file = 'P', es_amp = -100, clean = True):
    all_results = []
    df = pd.concat([pd.read_pickle('../Results/Result_Tables/' + file + '_A.pkl'), 
                    pd.read_pickle('../Results/Result_Tables/' + file + '_B.pkl'), 
                    pd.read_pickle('../Results/Result_Tables/' + file + '_C.pkl'), 
                    pd.read_pickle('../Results/Result_Tables/' + file + '_D.pkl'), 
                    pd.read_pickle('../Results/Result_Tables/' + file + '_E.pkl')], axis=0).reset_index(drop = True)
    
    radii = list(pd.read_csv('../Required_Files/Models/StickFluctuating.swc', header = None, delimiter = ' ').iloc[1:,5])

    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure(figsize = (14,3))
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1.*i/20) for i in range(20)])
    for i in range(20):
        ax.plot(np.arange(10)*(i+1), label = 'HOF: '+str(i))
    ax.set_axis_off()
    ax.legend(ncols = 4)
    plt.show()

    dfgrp = df.groupby('Cell_ID')
    for name, dfg in dfgrp:
        temp_results = [dfg.iloc[0,0],name, es_amp]
        dfg = dfg.groupby('ES_current(uA)').get_group(es_amp)
        dfgh = dfg.groupby('HOF')
        fig = plt.figure(figsize = (14,11))
        gs = fig.add_gridspec(18,6)
        gs2 = fig.add_gridspec(3,5)
        ax = []
        ax.append(fig.add_subplot(gs2[0, 3:]))
        ax.append(fig.add_subplot(gs2[1, 3:]))
        ax.append(fig.add_subplot(gs2[2, 3:]))
        ax[0].set_prop_cycle(color=[cm(1.*i/20) for i in range(20,0,-1)])
        ax[1].set_prop_cycle(color=[cm(1.*i/20) for i in range(20,0,-1)])
        ax[2].set_prop_cycle(color=[cm(1.*i/20) for i in range(20,0,-1)])
        fig.suptitle(str(dfg.iloc[0,0])+'_'+name)
        fig.supylabel(str('Distance from soma (um)'), fontsize = 16, x = 0.22)
        for hof, dfh in reversed(list(dfgh)):
            data = pd.DataFrame(np.array(dfh['Traces(mV)'])[0])

            for segm in range(18):
                if hof==19:
                    if segm:
                        ax.append(fig.add_subplot(gs[segm, 1:3]))
                    else:
                        ax.append(fig.add_subplot(gs[segm, 1:3]))
                    ax[segm+3].set_prop_cycle(color=[cm(1.*i/20) for i in range(20,0,-1)])
                ax[segm+3].tick_params(left=False, right=False, labelleft=True, labelbottom=False) 
                ax[segm+3].spines['left'].set_visible(False) 
                ax[segm+3].spines['bottom'].set_visible(False)
                ax[segm+3].spines['right'].set_visible(False) 
                ax[segm+3].spines['top'].set_visible(False)
                ax[segm+3].set_xticks([])
                ax[segm+3].set_yticks([])
                ax[segm+3].set_ylabel(str(segm*200+30))
                ax[segm+3].plot(data.iloc[100:400,segm*10]+100, alpha = 0.6, linewidth = 2)
            max_values = data.max(axis=0)
            max_index = data.idxmax(axis=0).where(data.idxmax(axis=0) > 100, np.nan).replace(499, np.nan)
            ax[0].plot(np.arange(len(radii))*20+30, radii, alpha = 0.6, linewidth = 2)
            ax[0].set_xlabel('Distance from soma (um)', fontsize = 16)
            ax[0].set_ylabel('Dendrite radius (um)', fontsize = 16)
            ax[0].set_xlim(0,3450)
            ax[1].plot(np.arange(len(max_values))*20+30, max_values, alpha = 0.6, linewidth = 2)
            ax[1].set_xlabel('Distance from soma (um)', fontsize = 16)
            ax[1].set_ylabel('Peak amplitude (mV)', fontsize = 16)
            ax[1].set_xlim(0,3450)
            ax[2].plot(np.arange(len(max_index))*20+30, (max_index-100)*0.05, alpha = 0.6, linewidth = 2)
            ax[2].set_xlabel('Distance from soma (um)', fontsize = 16)
            ax[2].set_ylabel('Peak delay (ms)', fontsize = 16)
            ax[2].set_xlim(0,3450)
        ax.append(fig.add_subplot(gs[:, 0]))
        morph = neurom.load_morphology('../Required_Files/Models/StickFluctuating.swc')
        viewer.plot_morph(morph, ax = ax[-1], plane = 'xy')
        ax[-1].set_aspect('equal')
        ax[-1].set_ylim(-2280,1150)
        ax[-1].set_xlim(-250,250)
        ax[-1].plot(-25,1120-1250, marker = 'x')
        ax[-1].set_axis_off()
        ax[-1].set_title('')
        
        plt.show()