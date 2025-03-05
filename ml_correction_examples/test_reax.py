import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax_md import space, simulate, quantity
from jax_md.reaxff.reaxff_interactions import reaxff_inter_list, calculate_all_angles_and_distances
from jax_md.reaxff.reaxff_helper import read_force_field
from jax_md.reaxff.reaxff_forcefield import ForceField
from ase.io import read
import json
import csv

dtype = jnp.float64

"""
Some notes:

- data.py generates data.json with the parameter values in a format that is
  easy to read. this can be a permenant solution or a placeholder until the
  fields are added to the official ffield file format

- this file can either compare energy values to the .csv from the original code
  for validation purposes, or run a short MD simulation as an example, see the
  functions at the bottom for examples

The user facing changes from the original code are the following:

- must load json and pass to read_force_field as corr_params,
  if corr_params is not passed or == None, the params will not
  be loaded

- use_ML_correction=True must be set in interaction list function

- ret_ML_correction=True can be set to return the total energy (including
  correction energy) as well as the correction energy by itself for debugging
"""

base_dir = "/mnt/home/betanc18/reax_correction/jax-md/ml_correction_examples/"
geo_path = base_dir + "SiC_ReaxFF_files/SiC_ReaxFF_files/xmolout_reaxff/raw/"
ffield_path = base_dir + "SiC_ReaxFF_files/SiC_ReaxFF_files/ffield"
param_file = base_dir + "data.json"
csv_path = base_dir + "results.csv"

with open(param_file, "r") as f:
    corr_params = json.load(f)

def load_csv_with_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Skip the header row
        for row in csv_reader:
            data.append(row)
    return header, data

header, data = load_csv_with_csv(csv_path)

ffield = read_force_field(ffield_path, cutoff2 = 0.001, corr_params=corr_params, dtype=dtype)
ffield = ForceField.fill_off_diag(ffield)
ffield = ForceField.fill_symm(ffield)

def check_csv():
    print("Number of systems in .csv", len(data))
    
    for i, row in enumerate(data):
        csv_corr_nrg = jnp.float64(row[-5])
        geo_file = row[-1] + '.xyz'
        print("Evaluating system", i, "named", geo_file)

        geo = read(geo_path + geo_file)

        positions = (geo.positions - np.min(geo.positions,axis=0))
        R = jnp.array(positions, dtype=dtype)

        types = geo.get_chemical_symbols()
        types_int = [ffield.name_to_index[t] for t in types]
        species = jnp.array(types_int)
        atomic_nums = jnp.array(geo.get_atomic_numbers())

        if (np.all(geo.pbc)):
            box_size = jnp.array(geo.cell.lengths(),dtype=dtype)
        else:
            diff = jnp.max(R,axis=0) - jnp.min(R,axis=0)
            box_size = jnp.max(diff) + 20

        displacement, shift = space.periodic(box_size)

        reaxff_inter_fn, energy_fn = reaxff_inter_list(displacement,
                                                            box_size,
                                                            species,
                                                            atomic_nums,
                                                            ffield,
                                                            use_ML_correction=True,
                                                            ret_ML_correction=True,
                                                            tol=1e-14)

        metric = space.metric(displacement)
        map_metric = space.map_neighbor(metric)
        map_disp = space.map_neighbor(displacement)

        nbr_lists = reaxff_inter_fn.allocate(R)

        total_nrg, corr_nrg = energy_fn(R, nbr_lists)

        if(jnp.abs(corr_nrg - csv_corr_nrg) > 1e-4):
            print("There is disagreement > 1e-4 in the correction terms for", geo_file)
            print("Calculated correction energy:", corr_nrg)
            print("CSV extracted correction energy:", csv_corr_nrg)
            print("Difference:", jnp.abs(corr_nrg - csv_corr_nrg))
    
    return

def check_single(geo_file):
    print("Evaluating system named", geo_file)

    geo = read(geo_path + geo_file)

    positions = (geo.positions - np.min(geo.positions,axis=0))
    R = jnp.array(positions, dtype=dtype)

    types = geo.get_chemical_symbols()
    types_int = [ffield.name_to_index[t] for t in types]
    species = jnp.array(types_int)
    atomic_nums = jnp.array(geo.get_atomic_numbers())

    if (np.all(geo.pbc)):
        box_size = jnp.array(geo.cell.lengths(),dtype=dtype)
    else:
        diff = jnp.max(R,axis=0) - jnp.min(R,axis=0)
        box_size = jnp.max(diff) + 20

    displacement, shift = space.periodic(box_size)

    reaxff_inter_fn, energy_fn = reaxff_inter_list(displacement,
                                                        box_size,
                                                        species,
                                                        atomic_nums,
                                                        ffield,
                                                        use_ML_correction=True,
                                                        ret_ML_correction=True,
                                                        tol=1e-14)

    metric = space.metric(displacement)
    map_metric = space.map_neighbor(metric)
    map_disp = space.map_neighbor(displacement)

    nbr_lists = reaxff_inter_fn.allocate(R)

    total_nrg, corr_nrg = energy_fn(R, nbr_lists)

    print("Calculated correction energy:", corr_nrg)

    return

def run_md(geo_file, MD_count, reax_time):
    print("Evaluating system named", geo_file)

    geo = read(geo_path + geo_file)

    positions = (geo.positions - np.min(geo.positions,axis=0))
    R = jnp.array(positions, dtype=dtype)

    types = geo.get_chemical_symbols()
    types_int = [ffield.name_to_index[t] for t in types]
    species = jnp.array(types_int)
    atomic_nums = jnp.array(geo.get_atomic_numbers())

    if (np.all(geo.pbc)):
        box_size = jnp.array(geo.cell.lengths(),dtype=dtype)
    else:
        diff = jnp.max(R,axis=0) - jnp.min(R,axis=0)
        box_size = jnp.max(diff) + 20

    displacement, shift = space.periodic(box_size)

    reaxff_inter_fn, energy_fn = reaxff_inter_list(displacement,
                                                        box_size,
                                                        species,
                                                        atomic_nums,
                                                        ffield,
                                                        use_ML_correction=True,
                                                        ret_ML_correction=False,
                                                        tol=1e-14)
    
    reaxff_inter_lists = reaxff_inter_fn.allocate(R)
    res = energy_fn(R, reaxff_inter_lists)
    forces = jax.grad(energy_fn)(R, reaxff_inter_lists)

    if jnp.count_nonzero(jnp.isnan(forces)) > 0:
        print("Nan forces")
        print(forces)
        sys.exit()

    nbrs = reaxff_inter_lists
    mass = ffield.amas[species]
    mass = jnp.array(mass, dtype=dtype).reshape(-1,1)
    base_time = 48.8882129 
    multip = reax_time/base_time
    init_fn, apply_fn = simulate.nve(energy_fn, shift, multip) #4.184e-3
    state = init_fn(jax.random.PRNGKey(0), R, kT=1e-8, mass=mass, nbr_lists=nbrs)

    @jax.jit
    def body_fn(i, state):
        state, nbrs = state
        nbrs = reaxff_inter_fn.update(state.position, nbrs)
        state = apply_fn(state, nbr_lists=nbrs)
        return state, nbrs
 
    #PE = []
    #KE = []
    step = 0
    data_collect_freq = 20
    #pbar = tqdm(total = MD_count)
    print("step, PE, KE")
    while step < MD_count:
        new_state, nbrs  = body_fn(step, (state, nbrs))
        if nbrs.did_buffer_overflow == True:
            print("overflow: ", step)
            print('Neighbor list overflowed, reallocating.')
            nbrs = reaxff_inter_fn.allocate(state.position)
        else:
            if step % data_collect_freq == 0:
                p_energy = jax.jit(energy_fn)(state.position, nbrs)
                k_energy = jax.jit(quantity.kinetic_energy)(velocity=state.velocity, mass=state.mass)
                print(step, p_energy, k_energy)
                #PE += [p_energy]
                #KE += [k_energy]
            state = new_state
            #pbar.update(1)
            step += 1
    return

# Check either a single file, every file in the csv,
# or run a short NVE simulation

run_md('4_1_xmolout_frame_34.xyz', 1000, 0.2)
# check_csv()
# check_single('4_1_xmolout_frame_34.xyz')