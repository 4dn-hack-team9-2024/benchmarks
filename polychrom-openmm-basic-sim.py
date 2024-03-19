import os
import sys
import time

import openmm

import polychrom
from polychrom import forcekits, forces, simulation, starting_conformations
from polychrom.hdf5_format import HDF5Reporter

import numpy as np

import click
@click.command()
@click.option('--gpu-id', 'GPU', required=True, type=str, help='GPU number you want to use')
@click.option('--n-monomers', 'N', required=True, type=int, help='Number of monomers')
@click.option('--output-folder', 'folder', required=True, type=str, help='Output folder name')
@click.option('--md-steps', 'n_steps', required=False, type=int, help='Number of simulation rounds', default=int(2e5))
@click.option('--step-size', required=False, type=int, help='Number of simulation rounds per one data dump', default=2000)
@click.option('--box-size', required=False, type=int, help='Box size', default=None)
@click.option('--density', required=False, type=float, 
        help='Density of the polymer. If neither density and box size are provided, the function falls back to density 0.2.', 
        default=None)


def test_polychrom(GPU, N, folder, n_steps, step_size, box_size, density):
    start = time.time()
    
    unique_id = np.random.randint(1000000)  # random number to avoid overwriting
    
    folder = f"{folder}_id{unique_id}"
    
    print(f"Starting simulations in ... {folder}")
    
    reporter = HDF5Reporter(folder=folder, max_data_length=5, overwrite=True)
        
    sim = simulation.Simulation(
        platform="CUDA",
        integrator="variableLangevin",
        error_tol=0.003,
        GPU=GPU,
        collision_rate=0.03,
        N=N,
        save_decimals=2,
        PBCbox=False,
        reporters=[reporter] if folder is not None else [],
    )
    
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
    
    polymer = starting_conformations.grow_cubic(N, box_size)
    
    sim.set_data(polymer, center=True)  # loads a polymer, puts a center of mass at zero
    
    sim.add_force(forces.spherical_confinement(sim, density=0.8, k=1))
    
    sim.add_force(
        forcekits.polymer_chains(
            sim,
            chains=[(0, None, False)],
            # By default the library assumes you have one polymer chain
            # If you want to make it a ring, or more than one chain, use self.setChains
            # self.setChains([(0,50,True),(50,None,False)]) will set a 50-monomer ring and a chain from 50 to the end
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                "bondLength": 1.0,
                "bondWiggleDistance": 0.05,  # Bond distance will fluctuate +- 0.05 on average
            },
            angle_force_func=forces.angle_force,
            angle_force_kwargs={
                "k": 1.5,
                # K is more or less arbitrary, k=4 corresponds to presistence length of 4,
                # k=1.5 is recommended to make polymer realistically flexible; k=8 is very stiff
            },
            nonbonded_force_func=forces.polynomial_repulsive,
            nonbonded_force_kwargs={
                "trunc": 3.0,  # this will let chains cross sometimes
                # 'trunc':10.0, # this will resolve chain crossings and will not let chain cross anymore
            },
            except_bonds=True,
        )
    )
    
    n_rounds = int(n_steps/step_size)
    for _ in range(n_rounds):  # Run blocks
        sim.do_block(step_size)  # of `step_size` timesteps each. Data is saved automatically.
    sim.print_stats()  # In the end, print very simple statistics
    
    reporter.dump_data()  # always need to run in the end to dump the block cache to the disk
    
    end = time.time()
    print("Simulation time:", end - start, "sec.")

if __name__ == '__main__':
        test_polychrom()

