import os
import sys
import time

import os
import sys
import json
import hoomd
import codecs

import gsd

import polychrom_hoomd.log as log
import polychrom_hoomd.build as build
import polychrom_hoomd.forces as forces

from polykit.generators.initial_conformations import grow_cubic

import numpy as np

import click
@click.command()
@click.option('--gpu-id', 'GPU', required=True, type=str, help='GPU id you want to use')
@click.option('--n-monomers', 'N', required=True, type=int, help='Number of monomers, aka system size')
@click.option('--output-folder', 'folder', 
              required=False, help='Output folder name, or None if skip writing output', default=None)
@click.option('--md-steps', 'n_steps', required=False, type=int, help='Number of MD simulation rounds.', default=int(2e5))
@click.option('--step-size', required=False, type=int, help='Number of simulation rounds per one data dump.', default=2000)
@click.option('--box-size', required=False, type=int, help='Box size', default=None)
@click.option('--density', required=False, type=float, 
        help='Density of the polymer. If neither density and box size are provided, the function falls back to density 0.2.', 
        default=None)
@click.option('--integrator', required=False, type=str, help="Type of integrator: DPD vs Langevin", default="Langevin")

def test_hoomd(GPU, N, folder, n_steps, step_size, box_size, density, integrator):
    start = time.time()
    
    unique_id = np.random.randint(1000000)  # random number to avoid overwriting

    if folder is not None: 
        folder = f"{folder}_id{unique_id}"
    
    print(f"Starting simulations in ... {folder}")

    # Initialise HooMD on the GPU
    hoomd_device = hoomd.device.GPU(gpu_ids=[int(GPU)])
    
    # Generate RNG seed
    rng_seed = os.urandom(2) # Random see to ensure we run different sims each iteration of benchmark.
    rng_seed = int(codecs.encode(rng_seed, 'hex'), 16)
    
    print("Using entropy-harvested random seed: %d" % rng_seed)

    chromosome_sizes = [N]

    # Initialize simulation with the appropriate box size
    number_of_monomers = sum(chromosome_sizes)

    # Retrieve box size:
    if box_size is None: 
        if density is None:
            density = 0.2

        box_size = int( (N / density) ** (1/3.) )

    elif density is None:
        density = N/box_size**3
    else:
        print('Both box_size and density are provided. Ignoring density.')
        density = N/box_size**3
    print('System size: ', N, 'Density: ', density, 'Box size: ', box_size)

    # HooMD to openMM time conversion factor
    t_conv = (1.67377*10**-27/(1.380649*10**-23*300))**0.5

# Initialize empty simulation object
    system = hoomd.Simulation(device=hoomd_device, seed=rng_seed)
    
    snapshot = build.get_simulation_box(box_length=box_size)
    
    # Build random, dense initial conformations
    monomer_positions = grow_cubic(N=number_of_monomers, boxSize=int(box_size-2))
    
    # Populate snapshot with the generated chains
    build.set_chains(snapshot, monomer_positions, chromosome_sizes, monomer_type_list=['A'])
    
    # Setup HooMD simulation object
    system.create_state_from_snapshot(snapshot)
    
    # Setup neighbor list
    nl = hoomd.md.nlist.Cell(buffer=0.4)

    # Read input force parameters
    with open("force_dict_homopolymer.json", 'r') as dict_file:
        force_dict = json.load(dict_file)
    
    # Set chromosome excluded volume
    repulsion_forces = forces.get_repulsion_forces(nl, **force_dict)
    
    # Set bonded/angular potentials
    bonded_forces = forces.get_bonded_forces(**force_dict)
    angular_forces = forces.get_angular_forces(**force_dict)
    
    if integrator.lower()=='dpd':
        # Set up DPD forces:
        dpd_forces = forces.get_dpd_forces(nl, **force_dict)

        # Set up the attraction forces:
        neighbors_list = hoomd.md.nlist.Cell(buffer=0.4)
        attraction_forces = forces.get_attraction_forces(
                neighbors_list, **force_dict)

        # Define full force_field
        force_field = repulsion_forces + bonded_forces + angular_forces + dpd_forces + attraction_forces

        # Setup new DPD integrator
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        dpd_integrator = hoomd.md.Integrator(dt=5e-3, methods=[nve], forces=force_field)

        # Setup simulation engine
        system.operations.integrator = dpd_integrator

    else:
        # Define full force_field
        force_field = repulsion_forces + bonded_forces + angular_forces
    
        # Initialize integrators and Langevin thermostat
        langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
        integrator = hoomd.md.Integrator(dt=70*t_conv, methods=[langevin], forces=force_field)
    
        # Setup simulation engine
        system.operations.integrator = integrator

    # Define how to write the results:
    if folder is not None:
        logger = log.get_logger(system)
        # system.operations.integrator = integrator # Do we need this line? 
        system.operations.writers.append(log.table_formatter(logger, period=step_size))
    
        gsd_writer = hoomd.write.GSD(filename=folder,
                                     trigger=hoomd.trigger.Periodic(step_size),
                                     dynamic=['topology'],
                                     mode='wb')
        system.operations.writers.append(gsd_writer)

    # Run the simulations:
    system.run(n_steps)
    
    end = time.time()
    print("Simulation time:", end - start, "sec.")

if __name__ == '__main__':
        test_hoomd()

