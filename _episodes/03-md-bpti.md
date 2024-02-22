---
title: "MD Simulation of a Protein"
teaching: 30
exercises: 60
questions:
- "How do I run a simulation for a protein?"
- "How do I analyze a simulation for a protein?"
objectives:
- "Understand the steps to preparing the structure from a PDB file for a simulation."
- "Set up a simulation for a protein."
- "Identify bond lengths and angles for analysis of a simulation trajectory."
- "Make plots to understand the motion of a protein from a molecular dynamics simulation."
keypoints:
- "Preparing a protein for an MD simulation has several steps.  You will often use existing tools to help with this task."
- "Identifying the relevant bond lengths and angles to analyze and graph is an important part of analyzing a molecular dynamics simulation."
- "Landmark simulations from the early days of molecular dynamics can now be completed in a few minutes on modern laptops."
---

This website is a customized version of the [MolSSI Molecular Mechancs Tools workshop](https://education.molssi.org/mm-tools/) created for the [Skidmore Computational Biophysics Lab](https://academics.skidmore.edu/blogs/computationalbiophysicslab/). All others should visit the original MolSSI site.

## Introduction

In this lesson we will carry out an MD simulation of the protein bovine pancreatic trypsin inhibitor (BPTI). This was one of the first proteins to ever be simulated in its entirety, and the results of this simulation – that at its time was a computational tour de force – can be found in a Nature article from 1977 by Andy McCammon, Bruce Gelin, and Martin Karplus (co-recipient of the 2013 Nobel Prize in Chemistry).  We will carry out a very similar simulation in just a few minutes!

## Starting structure and simulation preliminaries

Before you start, sign on to the pugetsound cluster.

The X-ray crystal structures (and NMR-derived structures) of many folded proteins can be found at the [Protein Data Bank](http://www.rcsb.org). Every structure deposited in the PDB has a four character code; for this exercise we will use the structure 4PTI. This structure was deposited in 1982 (!), but has excellent resolution (1.5 Å) and provides a good starting point for our simulation.

Taking a structure from the PDB and getting it ready for simulation is not a trivial task. For us to carry out a gas phase simulation of this protein, crystallographic waters must be removed and disulfide bonds between various cysteine residues must be specified. Then, to run a simulation of this protein in water, we must add in water molecules. Thankfully there are applications that help to automate this task. In this particular case, the application pdb4amber (part of the [AmberTools distribution](http://www.ambermd.org) was used to generate an appropriate [pdb file](../data/bpti_gas.pdb). Then another AmberTools application, tleap, was used to create parameter/topology files and starting coordinate files that can be understood by Amber or OpenMM.  If you want to learn more about how to use these tools, see the yellow box below, but you can also just copy all five of these files from the chem_shared directory. 

`cp /data/chem_shared/tutorial_files/bpti_* ./`

For this exercise, we will be using the Amber ff14SB protein force field.

> ## Getting the PDB file ready for the simulation
> To prepare the initial parameter/topology file and the input coordinate file, you will need AmberTools18.  
> - Download the file 4PTI.pdb from the PDB website into the directory of your choice.
> - In a terminal window, change to that directory and type the following commands:
>
> ~~~
> module load amber18/18
> pdb4amber -i 4PTI.pdb -o 4PTI_withCYX.pdb --dry --nohyd
> ~~~
> {: .language-bash}
>
> - Create a text file `setup_BPTI.leap` with the following content:
>
> ~~~
> source leaprc.protein.ff14SB
> ffmodw = loadamberparams frcmod.ff99SB_w_dih
> source leaprc.water.tip3p
> loadamberparams frcmod.tip3pfb
> bpti = loadpdb 4PTI_withCYX.pdb
> saveamberparm bpti bpti_gas.prmtop bpti_gas.inpcrd
> savepdb bpti bpti_gas.pdb
> solvateoct bpti TIP3PBOX 10.0
> addionsrand bpti Cl- 0
> addionsrand bpti Na+ 0
> saveamberparm bpti bpti_wat.prmtop bpti_wat.inpcrd
> quit
> ~~~
>
> - Back in the Terminal Window, copy the force field modification file to your current directory:
> 
> ~~~
> cp /data/chem_shared/tutorial_files/frcmod.ff99SB_w_dih ./
> ~~~
> {: .language-bash}
> 
> Enter the following command.
>
> ~~~
> tleap -f setup_BPTI.leap
> ~~~
> {: .language-bash}
>
> This will create the Amber-format prmtop and inpcrd files for BPTI in the gas phase, as well as an Amber-friendly pdb file.
{: .callout}

## MD simulation protocol
Before you begin, make sure the pdb file, parameter/topology file, and the starting coordinate file are in the same directory as your jupyter notebook where you plan to run the simulation.

Sign on to a compute node by typing `srun --partition=Legacy_Nodes --pty --nodes=1 --tasks-per-node=1 --gres=gpu:1 --time=24:00:00 --share --wait=0 --export=ALL /bin/bash`. **Write down the node you are signed on** to by looking at the prompt (it should be a number from 1 to 8).

Load the latest version of python by typing `module load anaconda3/python-3.7`.
Open a jupyter notebook by typing `jupyter notebook --no-browser`. **Write down the local host number.** 

In a separate terminal window on your local computer, log in to that jupyter notebook on the compute node by typing `ssh -L 8157:127.0.0.1:#### username@cnode00#.skidmore.edu` where '####' is the local host number of your jupyter notebook and where 'cnode00#' is whichever node you are running the jupyter notebook on.
Open a browser window and navigate to `http://localhost:8157/`.
Open a new jupyter notebook and name it `BPTI_gas_OpenMM`.

We will carry out a simulation protocol very similar to that of McCammon et al. (as well as our earlier exercises), with ​some​ modernized aspects to it:
1. Up to 100 steps of energy minimization using the L-BFGS algorithm. [In the original study the authors performed 100 steps of MD with initial velocities set to zero/a starting temperature of 0 K.]
2. A 1.0 ps MD simulation to bring the BPTI protein to an equilibrium temperature of 298 K (25 °C). [Original study: velocities from previous phase are multiplied by 1.5 and then 0.25 ps of equilibration in the NVE ensemble are performed, after which the system is at 285 K. This means the simulation was carried out without thermostatting and instead the total energy of the system – kinetic + potential – was conserved.]
3. A 9.0 ps MD simulation at 298 K in which structures are saved ​every step​ (1 fs) into a file called ​BPTI_gas_sim_1.dcd.​ [This is basically identical to the original study except that the original was again performed without thermostatting. Additional note: These days we usually do not save protein coordinates more frequently than once every 1 ps, but our simulations are typically 5-6 orders of magnitude longer than 40 years ago!]

### Importing required python libraries
First, import the required python as you did in the previous lesson.

~~~
from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
from sys import stdout
import time as time
~~~
{: .language-python}

### Simulation setup up
Specify the parameter/topology file and the initial coordinate file.  Again, make sure the files you downloaded are in the same directory as the jupyter notebook where you are running the simulation.  

~~~
prmtop = app.AmberPrmtopFile('bpti_gas.prmtop')
inpcrd = app.AmberInpcrdFile('bpti_gas.inpcrd')

system = prmtop.createSystem(constraints=app.HBonds)
integrator = mm.LangevinIntegrator(298.15*unit.kelvin, 1.0/unit.picoseconds,
    1.0*unit.femtoseconds)

platform = mm.Platform.getPlatformByName('CUDA')
simulation = app.Simulation(prmtop.topology, system, integrator, platform)
simulation.context.setPositions(inpcrd.positions)
if inpcrd.boxVectors is not None:
    simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
~~~
{: .language-python}

### Energy minimization
Minimize the energy of the initial structure.  You should see that the energy decreases from the minimization procedure.

~~~
st = simulation.context.getState(getPositions=True,getEnergy=True)
print("Potential energy before minimization is %s" % st.getPotentialEnergy())

print('Minimizing...')
simulation.minimizeEnergy(maxIterations=100)

st = simulation.context.getState(getPositions=True,getEnergy=True)
print("Potential energy after minimization is %s" % st.getPotentialEnergy())
~~~
{: .language-python}

~~~
Potential energy before minimization is -2108.982628950387 kJ/mol
Minimizing...
Potential energy after minimization is -5073.176745362812 kJ/mol
~~~
{: .output}

### Equilibration simulation
As in our previous activity, we will first perform a short equilibration simulation.  

~~~
simulation.context.setVelocitiesToTemperature(298.15*unit.kelvin)
print('Equilibrating...')
simulation.step(1000)
print('Done!')
~~~
{: .language-python}

~~~
Equilibrating...
Done!
~~~
{: .output}

### Production simluation

For our production simulation, while we will save the coordinates to the trajectory file every 1 fs, we will print the energy in our jupyter notebook much mess frequently, so that we don't have 9000 lines printing!  Once the simulation is finished, the total time for the simulation will be printed.  Note that your computer may be faster or slower, so your total time will probably be different than the example given here.

~~~
simulation.reporters.append(app.DCDReporter('bpti_gas_sim_1.dcd', 1))
simulation.reporters.append(app.StateDataReporter(stdout, 250, step=True, time=True,
    potentialEnergy=True, temperature=True, speed=True, separator='\t'))

tinit=time.time()
print('Running Production...')
simulation.step(9000)
tfinal=time.time()
simulation.saveState('bpti_gas_sim_1.xml')
print('Done!')
print('Time required for simulation:', tfinal-tinit, 'seconds')
~~~
{: .language-python}

~~~
Running Production...
#"Step"	"Time (ps)"	"Potential Energy (kJ/mole)"	"Temperature (K)"	"Speed (ns/day)"
1250	1.2499999999999731	-3195.9158598032036	281.21486080243955	0
1500	1.4999999999999456	-3069.091379717989	291.2978292809881	5.79
1750	1.749999999999918	-3091.0191534139713	303.29022389181273	5.89
2000	1.9999999999998905	-3020.211314437227	292.4791472408793	5.62
2250	2.249999999999863	-3069.568076612484	300.87350446797655	5.32
2500	2.4999999999998357	-3054.734889643838	293.40895254954535	5.3
2750	2.749999999999808	-3182.7126507253643	300.19316447041916	5.35
3000	2.9999999999997806	-3134.07692516566	299.00523225118224	5.35
3250	3.249999999999753	-3084.5665143787674	294.6905071769165	5.36
3500	3.4999999999997256	-3088.2552816555335	287.1261135399029	5.38
3750	3.749999999999698	-3204.359079271292	311.05528742652456	5.38
4000	3.9999999999996705	-3059.5513129293113	314.3251868807196	5.39
4250	4.249999999999754	-2940.1045801153577	316.6742135542288	5.4
4500	4.4999999999998375	-2885.054412899688	300.48851229925594	5.38
4750	4.749999999999921	-3053.810904377323	306.1508532271461	5.35
5000	5.000000000000004	-2944.8314085401407	296.0008091598833	5.36
5250	5.250000000000088	-3126.1550618978144	301.77770492343996	5.36
5500	5.500000000000171	-3169.040273360173	292.9367059671542	5.37
5750	5.750000000000255	-3174.787374145096	304.92553982461357	5.35
6000	6.000000000000338	-3018.7517664568213	286.770250900887	5.36
6250	6.250000000000422	-2903.335036848406	288.9085230499434	5.34
6500	6.500000000000505	-3000.193267342126	298.6804422396432	5.31
6750	6.750000000000589	-2966.4071969867964	301.95636855144227	5.27
7000	7.000000000000672	-2896.3409604537283	304.54002619973164	5.28
7250	7.250000000000756	-3186.709402033014	322.4658266519081	5.29
7500	7.500000000000839	-3007.1725334140556	295.2712831468122	5.3
7750	7.750000000000923	-3069.7933377429144	306.0929545959182	5.32
8000	8.000000000001005	-3177.5331702617914	306.6509042161882	5.33
8250	8.250000000000867	-3095.853001303728	309.62647528762903	5.33
8500	8.500000000000728	-3074.9351608901015	306.9449384768154	5.32
8750	8.75000000000059	-2982.541196133081	307.7459093556099	5.31
9000	9.000000000000451	-2947.1879215796907	297.596927175106	5.32
9250	9.250000000000313	-2830.0792625779504	293.7666227426069	5.33
9500	9.500000000000174	-2928.130915478312	304.4073787636173	5.34
9750	9.750000000000036	-2940.6819943612936	308.91827133280117	5.36
10000	9.999999999999897	-2938.041552736946	312.2066948779066	5.37
Done!
Time required for simulation: 144.49478197097778 seconds
~~~
{: .output}

### Sumbitting to the queue

This simulation was pretty short, but if we want to run a longer simulation (nanoseconds or microseconds) we will need to be able to submit our job and come back to it. We can do this by running our code in a python script rather than a jupyter notebook. Copy the files `simulation_template.py` and `template_gpu.scr` from `/data/chem_shared/tutorial_files/` to your current directory. You will use these files as templates for your future simulation scripts, so don't modify them. Instead, copy them to new files `BPTI_wat_new.py` and `BPTI_wat_new.scr` respectively.

Copy the python code from your jupyter notebook for running the BPTI simulation to the indicated spot in the `BPTI_wat_new.py` python script.
Edit the `BPTI_wat_new.scr` file to have the correct paths to your current directory, error file names, and py script name.

Now, you are ready to submit your job. To do so, open a new terminal window and log on to pugetsound. `cd` into the directory with your BPTI scripts and type the command:

`sbatch BPTI_wat_new.scr`

Your job has been submitted to the queue. You can see if it is running by typing

`squeue`

You will see your job running on a node. When you come back later you can check again to see if it is still running. If it is not, that means it finished or there was an error. To check, open the error file that was created and look in it. Once your job has finished successfully, you can move on to analysis.


## Trajectory analysis

Your overall goal in the exercise below is to reproduce – in a rough way – Figure 3a from McCammon et al. paper.  (Remember this figure was made in 1977!)

<img src="../fig/Figure3a.png" title="Figure 3a from McCammon, Gelin, and Karplus, ​Nature​ 1977." style="display: block; margin: auto;" />

> ## Exercise
> Create a new jupyter notebook called BPTI_gas_MDTraj.
> Begin by once again importing the required libraries.
>
> ~~~
> import mdtraj as md
> import numpy as np
> import matplotlib.pyplot as plt
> import nglview as ngl
> %matplotlib inline
> ~~~
> {: .language-python}
>
> Next use this code to identify all of the atoms that will be involved in the bond length, angle, and torsions we want to analyze.
>
> ~~~
> traj = md.load('bpti_gas_sim_1.dcd', top='bpti_gas.prmtop')
> atoms, bonds = traj.topology.to_dataframe()
> atoms[:10] # there are many more!
>
> # find all of the backbone atoms for residues 21-23
> print('N atoms for residues 21 to 23 are:')
> print(traj.topology.select('resid 20 to 22 and name N')) # beware the zero-indexing of Python!
> print('CA atoms for residues 21 to 23 are:')
> print(traj.topology.select('resid 20 to 22 and name CA'))
> print('C atoms for residues 21 to 23 are:')
> print(traj.topology.select('resid 20 to 22 and name C'))
> print('All atoms of residue 22 are:')
> print(atoms[349:369])
> ~~~
> {: .language-python}
>
> Next, use this code to plot the N-Cα bond length vs. time for residue Phe22 for 0.5 ps, beginning with step 5000.
> ~~~
> bond_indices = [349, 351]
> NClength = md.compute_distances(traj, [bond_indices])
>
> tstart = 5000 # start at step 5000
> dt = 500 # plot over 0.5 ps
>
> plt.plot(10*NClength[tstart:tstart+dt], color='Tomato')
> # note above that we have multiplied NClength by 10 to convert from nm to Å
> plt.title(r'N-C$\alpha$ bond length')
> plt.xlabel('Time (fs)')
> plt.ylabel(r'Bond length ($\AA$)')
> plt.show()
> ~~~
> {: .language-python}
>
> Write code to complete the analysis:
> 1. Convert the N-H bond length from nm to Å and then plot the bond length vs. time over the same time interval as the N-Cα bond. How do the dynamics of this bond differ from those of the N-Cα bond? [Just by eyeballing the range of N-H bond lengths in this plot, the distribution of the N-H bond length should look ​very​ different from the C-H bond lengths in the butane simulation. What’s different about this simulation?]
> 2. Convert the N-Cα-C bond angle from radians to degrees and then plot the bond angle vs. time over the same time interval.
> 3. Convert the φ, ψ, and ω torsions from radians to degrees and then plot > them vs. time for 2.0 ps, also beginning with step 5000. You might notice that the ω torsion plot looks suboptimal because this torsion angle oscillates around ±180°. How might we fix it to look more human-readable?
> 4. IN CLASS EXERCISE: Copy your BPTI_gas_OpenMM notebook and call the new notebook BPTI_wat_OpenMM. Modify this notebook to run the same simulation, but this time on the BPTI protein in water. You will need to add an additional minimization and equilibration of just the water with restraints on the protein (see the [OpenMM documentation](http://docs.openmm.org/7.0.0/userguide/application.html#force-fields)). Compare the speed of this simulation to the BPTI in gas. What explains this difference? Complete the same analysis of the BPTI simulation in water. Do you notice any differences?
>
>> ## Solution
>>
>> ~~~
>> import mdtraj as md
>> import numpy as np
>> import matplotlib.pyplot as plt
>> import nglview as ngl
>> %matplotlib inline
>>
>> traj = md.load('bpti_gas_sim_1.dcd', top='bpti_gas.prmtop')
>> atoms, bonds = traj.topology.to_dataframe()
>> atoms[:10] # there are many more!
>>
>> # find all of the backbone atoms for residues 21-23
>> print('N atoms for residues 21 to 23 are:')
>> print(traj.topology.select('resid 20 to 22 and name N')) # beware the zero-indexing of Python!
>> print('CA atoms for residues 21 to 23 are:')
>> print(traj.topology.select('resid 20 to 22 and name CA'))
>> print('C atoms for residues 21 to 23 are:')
>> print(traj.topology.select('resid 20 to 22 and name C'))
>> print('All atoms of residue 22 are:')
>> print(atoms[349:369])
>>
>> bond_indices = [349, 351]
>> NClength = md.compute_distances(traj, [bond_indices])
>>
>> tstart = 5000 # start at step 5000
>> dt = 500 # plot over 0.5 ps
>>
>> plt.plot(10*NClength[tstart:tstart+dt], color='Tomato')
>> # note above that we have multiplied NClength by 10 to convert from nm to Å
>> plt.title(r'N-C$\alpha$ bond length')
>> plt.xlabel('Time (fs)')
>> plt.ylabel(r'Bond length ($\AA$)')
>> plt.show()
>>
>> # Number 1
>> bond_indices = [349, 350]
>> NHlength = md.compute_distances(traj, [bond_indices])
>>
>> plt.plot(10*NHlength[tstart:tstart+dt], color='DarkViolet')
>> plt.title('N-H bond length')
>> plt.xlabel('Time (fs)')
>> plt.ylabel(r'Bond length ($\AA$)')
>> plt.show()
>>
>> # Number 2
>> angle_indices = [349, 351, 367]
>> bondangle = md.compute_angles(traj, [angle_indices])
>>
>> plt.plot((180/np.pi)*bondangle[tstart:tstart+dt], color='Goldenrod')
>> plt.title(r'N-C$\alpha$-C bond angle')
>> plt.xlabel('Time (fs)')
>> plt.ylabel('Bond angle (degrees)')
>> plt.show()
>>
>> # Number 3
>> phi_ind = [347,349,351,367] # define phi22
>> psi_ind = [349,351,367,369] # define psi22
>> omega_ind = [330,347,349,351] # define omega22
>> torsions = md.compute_dihedrals(traj, [phi_ind, psi_ind, omega_ind])
>> torsionsd = (180/np.pi)*torsions # convert torsions to degrees!
>>
>> dt = 2000 # plot over 2.0 ps
>>
>> plt.plot(torsionsd[tstart:tstart+dt,0], color='FireBrick')
>> plt.title(r'$\phi$22 torsion angle')
>> plt.xlabel('Time (fs)')
>> plt.ylabel(r'$\phi$ (degrees)')
>> plt.show()
>>
>> plt.plot(torsionsd[tstart:tstart+dt,1], color='ForestGreen')
>> plt.title(r'$\psi$22 torsion angle')
>> plt.xlabel('Time (fs)')
>> plt.ylabel(r'$\psi$ (degrees)')
>> plt.show()
>>
>> plt.plot(torsionsd[tstart:tstart+dt,2], color='DodgerBlue')
>> plt.title(r'$\omega$22 torsion angle')
>> plt.xlabel('Time (fs)')
>> plt.ylabel(r'$\omega$ (degrees)')
>> plt.show()
>>
>> # Cleaning up #3
>> # well that last graph looks kind of wacky...
>> # let's fix it up by changing the limits from (-180°,180°) to (0°,360°)
>> omegad = torsionsd[:,2]
>> omegad[omegad < 0] = 360 + omegad[omegad < 0]
>>
>> plt.plot(omegad[tstart:tstart+dt], color='DodgerBlue')
>> plt.title(r'$\omega$22 torsion angle')
>> plt.xlabel('Time (fs)')
>> plt.ylabel(r'$\omega$ (degrees)')
>> plt.show()
>> ~~~
>> {: .language-python}
>> 
>> Exercise 4
>> ~~~
>> from simtk.openmm import app
>> import simtk.openmm as mm
>> from simtk import unit
>> from sys import stdout
>> import time as time 
>>
>> # Set up the system
>> prmtop = app.AmberPrmtopFile('bpti_wat.prmtop')
>> inpcrd = app.AmberInpcrdFile('bpti_wat.inpcrd')
>> 
>> num_protein_res = 58 # change this to match the number of residues in your protein
>> # find the number of atoms in the protein
>> num_protein_atoms = prmtop.topology._chains[0]._residues[num_protein_res]._atoms[0].index
>> num_protein_atoms
>> 
>> system = prmtop.createSystem(constraints=app.HBonds, nonbondedMethod=app.PME,
>>                             nonbondedCutoff=1*unit.nanometer) # new parameters for in water
>> integrator = mm.LangevinIntegrator(298.15*unit.kelvin, 1.0/unit.picoseconds,
>>     2.0*unit.femtoseconds) # run at 298.15 K with a time step of 2 fs
>> platform = mm.Platform.getPlatformByName('CUDA')
>> 
>> # Add restraints on all protein atoms
>> force = mm.CustomExternalForce("k*periodicdistance(x,y,z,x0,y0,z0)^2")
>> force.addGlobalParameter("k", 10.0*unit.kilocalories_per_mole/(unit.angstroms**2))
>> force.addPerParticleParameter("x0")
>> force.addPerParticleParameter("y0")
>> force.addPerParticleParameter("z0")
>> 
>> for i in range(num_protein_atoms):
>>     force.addParticle(i, inpcrd.getPositions()[i])
>>     
>> force_index = system.addForce(force)
>> 
>> # Create simulation
>> simulation = app.Simulation(prmtop.topology, system, integrator, platform)
>> simulation.context.setPositions(inpcrd.positions)
>> if inpcrd.boxVectors is not None:
>>     simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
>>     
>> # Minimize system
>> st = simulation.context.getState(getPositions=True,getEnergy=True)
>> print("Potential energy before minimization is %s" % st.getPotentialEnergy())
>> 
>> print('Minimizing...')
>> simulation.minimizeEnergy(maxIterations=1000) # minimize for 1000 steps
>> 
>> st = simulation.context.getState(getPositions=True,getEnergy=True)
>> print("Potential energy after minimization is %s" % st.getPotentialEnergy())
>> 
>> # Equilibrate system with restraints on the protein
>> simulation.context.setVelocitiesToTemperature(100*unit.kelvin) # start the simulation at 100 K
>> st = simulation.context.getState(getPositions=True,getEnergy=True)
>> print("Kinetic energy before equilibration is %s" % st.getKineticEnergy())
>> 
>> print('Equilibrating...')
>> simulation.step(50000) # equilibrate for 100 ps
>> print('Done!')
>> 
>> st = simulation.context.getState(getPositions=True,getEnergy=True)
>> print("Kinetic energy after equilibration is %s" % st.getKineticEnergy())
>> 
>> # Remove the restraints on the protein
>> system.removeForce(force_index)
>> # Hold pressure constant
>> system.addForce(mm.MonteCarloBarostat(1.013*unit.bar, 298.15*unit.kelvin))
>> 
>> # Recreate simulation with system without restraints
>> integrator = mm.LangevinIntegrator(298.15*unit.kelvin, 1.0/unit.picoseconds,
>>     2.0*unit.femtoseconds) # run at 298.15 K with a time step of 2 fs
>> simulation = app.Simulation(prmtop.topology, system, integrator, platform)
>> simulation.context.setState(st)
>> 
>> # Minimize system without restraints
>> st = simulation.context.getState(getPositions=True,getEnergy=True)
>> print("Potential energy before minimization is %s" % st.getPotentialEnergy())
>> 
>> print('Minimizing...')
>> simulation.minimizeEnergy(maxIterations=1000) # minimize for 1000 steps
>> 
>> st = simulation.context.getState(getPositions=True,getEnergy=True)
>> print("Potential energy after minimization is %s" % st.getPotentialEnergy())
>> 
>> # Equilibrate system without restraints on the protein
>> simulation.context.setVelocitiesToTemperature(298.15*unit.kelvin)
>> st = simulation.context.getState(getPositions=True,getEnergy=True)
>> print("Box volume before equilibration is %s" % st.getPeriodicBoxVolume())
>> 
>> print('Equilibrating...')
>> simulation.step(50000) # equilibrate for 100 ps
>> print('Done!')
>> 
>> st = simulation.context.getState(getPositions=True,getEnergy=True)
>> print("Box volume after equilibration is %s" % st.getPeriodicBoxVolume())
>> 
>> # Run production simulation
>> 
>> #only save structure every 10 ps
>> simulation.reporters.append(app.DCDReporter('bpti_wat_sim_1.dcd', 5000))
>> # Print every 10 ps (you can change this frequency)
>> simulation.reporters.append(app.StateDataReporter('bpti_wat_sim_1.out', 5000, step=True, time=True,
>>     potentialEnergy=True, temperature=True, density=True, speed=True, separator='\t'))
>> 
>> tinit=time.time()
>> print('Running Production...')
>> simulation.step(500000) # run simulation for 1 ns
>> tfinal=time.time()
>> simulation.saveState('bpti_wat_sim_1.xml')
>> print('Done!')
>> print('Time required for simulation:', tfinal-tinit, 'seconds')
>> 
>> # Load in trajectory and remove solvent
>> # Before running this, create a directory called 'stripped' inside the directory with this notebook
>> 
>> import mdtraj as md
>> traj = md.load('bpti_wat_sim_1.dcd', top='bpti_wat.prmtop')
>> stripped_traj = traj.atom_slice(traj.top.select('resSeq 0 to 57')) # change this to be one less than the number of residues in your protein
>> stripped_traj.save_dcd('stripped/bpti_wat_sim_1_stripped.dcd')
>> 
>> ~~~
>> {: .language-python}
>>
>> To continue this simulation in a new notebook
>> ~~~
>> from simtk.openmm import app
>> import simtk.openmm as mm
>> from simtk import unit
>> from sys import stdout
>> import time as time 
>> 
>> num_protein_res = 58 # change this to match the number of residues in your protein
>> 
>> # Create system using the original prmtop file
>> prmtop = app.AmberPrmtopFile('bpti_wat.prmtop')
>> system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1*unit.nanometer, 
>>                              constraints=app.HBonds) # new parameters for in water
>> system.addForce(mm.MonteCarloBarostat(1.013*unit.bar, 298.15*unit.kelvin)) # hold pressure constant
>> integrator = mm.LangevinIntegrator(298.15*unit.kelvin, 1.0/unit.picoseconds,
>>     2.0*unit.femtoseconds) # run at 298.15 K with a time step of 2 fs
>> 
>> platform = mm.Platform.getPlatformByName('CUDA')
>> simulation = app.Simulation(prmtop.topology, system, integrator, platform)
>> 
>> # Load in the saved state from the end of the previous simulation
>> simulation.loadState('bpti_wat_sim_1.xml')
>> 
>> # Run production simulation
>> simulation.reporters.append(app.DCDReporter('bpti_wat_sim_2.dcd', 5000)) #only save structure every 10 ps
>> # You can change how often the output is printed
>> simulation.reporters.append(app.StateDataReporter('bpti_wat_sim_2.out', 5000, step=True, time=True,
>>     potentialEnergy=True, temperature=True, density=True, speed=True, separator='\t'))
>> 
>> tinit=time.time()
>> print('Running Production...')
>> simulation.step(500000) # run simulation for 1 ns
>> tfinal=time.time()
>> simulation.saveState('bpti_wat_sim_2.xml')
>> print('Done!')
>> print('Time required for simulation:', tfinal-tinit, 'seconds')
>> 
>> # Load in trajectory and remove solvent
>> import mdtraj as md
>> traj = md.load('bpti_wat_sim_2.dcd', top='bpti_wat.prmtop')
>> stripped_traj = traj.atom_slice(traj.top.select('resSeq 0 to 57')) # change this to be one less than the number of residues in your protein
>> stripped_traj.save_dcd('stripped/bpti_wat_sim_2_stripped.dcd')
>> 
>> ~~~
>> {: .language-python}
> {: .solution}
{: .challenge}

{% include links.md %}
