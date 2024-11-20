from scipy.optimize import curve_fit
from scipy.stats import norm
from collections import Counter
from statannot import add_stat_annotation
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
            plt.show()
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
        
 