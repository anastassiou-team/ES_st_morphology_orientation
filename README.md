## **Electrical Stimulation field properties** versus **Cell morphology, orientation and modeling**

Here we try to understand the effects of **cell morphology** and **field parameters** during **Electrical Stimulation**.
We test **200+ reconstructed morphologies** of human cells, under **different field types, locations and orientations**.

Required installations/environment:
> a) **Windows 10** Home 22H2 19045.5679 <br/>
> b) **Anaconda3** 2023.03-0 Windows x86_64 <br/>
> c) **Neuron Yale** 8.2.6 <br/>
> d) **Python modules**: **bmtk** 1.1.2, **efel** 5.7.16, **h5py** 3.10.0, **neurom** 3.2.1, **numpy** 1.23.5, **pandas** 1.5.3, 
**pycircstat** 0.0.2, **nose** 1.3.7, **scipy** 1.15.2, **statannot** 0.2.3 <br/>

Additional BMTK functionality for plane field simulations:
> a) Copy the files included in the directory **.\Required_Files\PlaneField** <br/>
> b) Paste them and replace the original files in **.\site-packages\bmtk\simulator\bionet\modules** <br/>