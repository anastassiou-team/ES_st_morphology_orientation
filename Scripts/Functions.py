from scipy.optimize import curve_fit
from collections import Counter
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

def Read_Tables(files, frequency, amplitude, phase = False, maximal = False):
    dfs = []
    for i in files:
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
            cell_sets.append(set(i.loc[:,'Cell_ID']))
            if max_set < len(cell_sets[-1]):
                max_set = len(cell_sets[-1])
        common_cells = cell_sets[0]
        for s in cell_sets[1:]:
            common_cells = common_cells.intersection(s)
        for i in range(len(dfs)):
            dfs[i] = dfs[i][dfs[i]['Cell_ID'].isin(common_cells)]
        if max_set != len(common_cells):
            print ("WARNING: Reduced cells from "+str(max_set)+" to "+str(len(common_cells))+" due to mismatch.")     
    df = dfs[0].iloc[:,:3].copy().reset_index(drop = True)
    for i in range(len(dfs)):
        assert list(df.loc[:,'Cell_ID'])==list(dfs[i].loc[:,'Cell_ID'])
        if phase:
            df.loc[:,files[i]] = (dfs[i].loc[:,'Phs@40000(deg)'].reset_index(drop = True)+270)%360
            if maximal:
                df.loc[:,'Amplitude_'+files[i]] = dfs[i].loc[:,'Amp_Median(mV)'].reset_index(drop = True)
        else: 
            df.loc[:,files[i]] = dfs[i].loc[:,'Amp_Median(mV)'].reset_index(drop = True)
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