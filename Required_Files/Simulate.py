base_dir = './'
inj_curr = 0.6
inj_std = 0.0
exp_rate = 10
es_freq = 0.0
es_ampl = 10
mod_proc = 'csmc_allactive'
point = 0
run = 1
returns = 0

if run:
    from bmtk.builder.networks import NetworkBuilder
    from bmtk.utils.sim_setup import build_env_bionet
    from scipy.stats import norm
    from bmtk.simulator import bionet
    import os, re, shutil, h5py, efel
    import pandas as pd
    import numpy as np
    import cell_functions
    from scipy.signal import hilbert        
    
    sim_step =   0.05 # ms
    stim_start =  100 # ms
    if returns >= 5:
        inj_curr = 0.0
        inj_std = 0.0
        sim_stop = 2000
    elif returns > 2:
        sim_stop = 500
    else:
        sim_stop =   1000+int(800000/exp_rate) # ms   
    stim_dura =  sim_stop - stim_start # ms
        

    # Set intracellular params
    in_amp =      inj_curr # nA
    
    if returns == 8:
        probeX =       25 # um
        probeY =      450 # um
    elif returns == 7:
        probeX =       25 # um
        probeY =     -450 # um
    elif returns == 6:
        probeX =       25 # um
        probeY =     1250 # um
    elif returns == 5:
        probeX =       25 # um
        probeY =    -1250 # um
    elif point:
        probeX =       50 # um
        probeY =        0 # um
    else:
        import bmtk.simulator.bionet.modules.config as cnf
        cnf.x =         1 # plane field at X
        probeX =     2500 # um
        probeY =        0 # um

    # Read frequency and current
    fr = es_freq  # Hz
    cr = es_ampl  # uA

    #Preset simulation (only to create files)
    net = NetworkBuilder('single_neuron')
    net.add_nodes(cell_name='cell',
        potental='exc',
        model_type='biophysical',
        model_template='ctdb:Biophys1.hoc',
        model_processing=mod_proc,
        dynamics_params='model.json',
        morphology='morph.swc')
    net.build()
    net.save_nodes(output_dir=base_dir + '/Model')

    build_env_bionet(
        base_dir=base_dir + '/Simulation',
        config_file='config.json',
        network_dir=base_dir + '/Model',
        tstop=sim_stop, dt=sim_step,
        report_vars=['v', 'cai'],
        current_clamp={
            'amp': in_amp,
            'delay': stim_start,
            'duration': stim_dura},
        include_examples=False,
        compile_mechanisms=False)

    #Set simulation (poke files and set values)
    shutil.move(base_dir + '/morph.swc', base_dir + '/Simulation/components/morphologies/morph.swc')
    shutil.move(base_dir + '/model.json', base_dir + '/Simulation/components/biophysical_neuron_models/model.json')
    shutil.move(base_dir + '/nrnmech.dll', base_dir + '/Simulation/components/mechanisms/nrnmech.dll')
    if es_freq>0:
        sim_file=base_dir + '/Simulation/simulation_config.json'
        f = open(sim_file,'r')
        filedata = f.read()
        f.close()
        if returns >= 5:
            filedata=re.sub('"inputs": {.*?},','"inputs": {\n    \
            "current_clamp": {\n      \
            "input_type": "current_clamp",\n      \
            "module": "IClamp",\n      \
            "node_set": "all",\n      \
            "gids": "all",\n      \
            "amp": '+str(in_amp)+',\n      \
            "delay": '+str(stim_start)+',\n      \
            "duration": '+str(stim_dura)+'\n    \
            },\n    \
            "extra_stim": {\n      \
            "input_type": "lfp",\n      \
            "module": "xstim",\n      \
            "node_set": "all",\n      \
            "positions_file": "$BASE_DIR/inputs/xstim_electrode.csv",\n      \
            "waveform": {\n        \
            "shape": "dc",\n        \
            "del": '+str(1000)+',\n        \
            "amp": '+str(cr/1000)+',\n        \
            "dur": '+str(fr)+'\n        \
            }\n    \
            }\n  \
            },',filedata, flags=re.DOTALL)
        else:
            filedata=re.sub('"inputs": {.*?},','"inputs": {\n    \
            "current_clamp": {\n      \
            "input_type": "current_clamp",\n      \
            "module": "IClamp",\n      \
            "node_set": "all",\n      \
            "gids": "all",\n      \
            "amp": '+str(in_amp)+',\n      \
            "delay": '+str(stim_start)+',\n      \
            "duration": '+str(stim_dura)+'\n    \
            },\n    \
            "extra_stim": {\n      \
            "input_type": "lfp",\n      \
            "module": "xstim",\n      \
            "node_set": "all",\n      \
            "positions_file": "$BASE_DIR/inputs/xstim_electrode.csv",\n      \
            "waveform": {\n        \
            "shape": "sin",\n        \
            "del": '+str(stim_start)+',\n        \
            "amp": '+str(cr/1000)+',\n        \
            "freq": '+str(fr)+',\n        \
            "offset": 0.0,\n        \
            "dur": '+str(stim_dura)+'\n      \
            }\n    \
            }\n  \
            },',filedata, flags=re.DOTALL)
 
        if returns==3 or returns>=5:
            filedata=re.sub('"cai_report": {.*?}','"v_axon": {\n      \
            "variable_name": "v",\n      \
            "cells": "all",\n      \
            "module": "membrane_report",\n      \
            "sections": "axon"\n    \
            }',filedata, flags=re.DOTALL)
        
        f = open(sim_file,'w')
        f.write(filedata)
        f.close()
        print ("Simulation set.")
    
        #Copy component files
        os.mkdir(base_dir + '/Simulation/inputs/')
        f = open(base_dir + '/Simulation/inputs/xstim_electrode.csv', 'w')
    
        #Extracellular stimuli probe
        Epositions=[]
        Epositions.append('ip pos_x pos_y pos_z rotation_x rotation_y rotation_z\n')
        Epositions.append('0 '+str(probeX)+' '+str(probeY)+' 0.0 0.0 0.0 0.0')
        f.writelines(Epositions)
        f.close()

    #Run simulation
    conf = bionet.Config.from_json(base_dir + '/Simulation/config.json')
    conf.build_env()
    net = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=net)
    if inj_std>0:
        cell = sim.net.get_cell_gid(0)
        ig = sim.h.InGauss(cell._secs[0](0.5))
        ig.delay=stim_start
        ig.dur=stim_dura
        ig.stdev=inj_std
        ig.mean=0.0
        sim._iclamps.append(ig)

    sim.run()
    
    if returns in [1,2]:
        soma_data_output_file = base_dir +'/Simulation/output/v_report.h5' #Soma spike train file
        with h5py.File(soma_data_output_file, "r") as f: #Read soma spike train file
            report = f['report']
            single_neuron = report['single_neuron'] #Depends on network name (NetworkBuilder BMTK)
            data2=single_neuron['data'][()]
        f.close()
        co = pd.DataFrame(np.array(data2))
        thrs = co.iloc[10000:-10000]
        thrs = (thrs.max()-thrs.min())/2+thrs.min()
        print("Threshold: "+str(thrs))
        efel.setThreshold(thrs)
        trace1 = {}
        trace1['T'] = co.index*sim_step
        trace1['V'] = co.iloc[:,0]
        trace1['stim_start'] = [stim_start]
        trace1['stim_end'] = [sim_stop]
        traces = [trace1]
        Peak_indices = efel.getFeatureValues(traces, ['peak_indices'])[0]['peak_indices']
        start = len(Peak_indices)
        stop = 0
        for j in range(len(Peak_indices)):
            if Peak_indices[j]<10000:
                start = j
            if Peak_indices[j]<=(len(co)+10000):
                stop = j
        Peak_indices = Peak_indices[start+1:stop]
    
        ISIs = Peak_indices[1:]-Peak_indices[:-1]
        ISIs = 20000 / ISIs
        if returns==2:
            mean,std=norm.fit(ISIs)
            pd.DataFrame([mean,std]).to_csv(base_dir + '/Data.csv',index=False)
        elif returns==1:
            pd.DataFrame(Peak_indices).to_csv(base_dir + '/Data.csv',index=False)
    elif returns == 0:
        neuron_data_output_file = base_dir + '/Simulation/output/v_report.h5'
        with h5py.File(neuron_data_output_file, "r") as f: #Read soma spike train file
            report = f['report']
            single_neuron = report['single_neuron'] #Depends on network name (NetworkBuilder BMTK)
            data=single_neuron['data'][()]
        f.close()
        
        df = pd.DataFrame(np.array(data)).loc[2000/sim_step:].reset_index(drop = True)
        analytic_signal = hilbert((df-df.mean()).values.flatten())
        amplitude_envelope = pd.DataFrame(np.abs(analytic_signal))
        instantaneous_phase = pd.DataFrame(np.unwrap(np.angle(analytic_signal))*180/np.pi%360)
        df = df.iloc[int(500/sim_step):-int(500/sim_step)]
        amplitude_envelope = amplitude_envelope.iloc[int(500/sim_step):-int(500/sim_step)]
        instantaneous_phase = instantaneous_phase.iloc[int(500/sim_step):-int(500/sim_step)]
        rf = pd.DataFrame([
            df.min(),
            df.max(),
            df.mean(),
            df.median(),
            amplitude_envelope.min(),
            amplitude_envelope.max(),
            amplitude_envelope.mean(),
            amplitude_envelope.median(),
            amplitude_envelope.loc[40000],
            instantaneous_phase.min(),
            instantaneous_phase.max(),
            instantaneous_phase.mean(),
            instantaneous_phase.median(),
            instantaneous_phase.loc[30000],  
            instantaneous_phase.loc[40000]])
        rf.to_csv(base_dir + '/Data.csv', index = False)
    try:
        shutil.rmtree(base_dir + '/Simulation/components')
        shutil.rmtree(base_dir + '/Simulation/inputs')
        shutil.rmtree(base_dir + '/Model')
    except:
        pass