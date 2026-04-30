import os
import json
from datetime import datetime

import math

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
import numpy as np

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")


def plot_FER(*args, title='', sigma=[0]):
    '''
    Plots FER over physical error probability.

    Parameters:
    ------------------------------------------------------------
    args    (dict)      - FER mapping
            (tuple)     - Tuple containing FER mapping and label
    title   (str)       - Plot title
    '''

    ax = plt.subplot()
    markers = ['x', '+', '*', 's', '^', 'D', 'v', 'p', 'd', '|', '_']

    for ind, arg in enumerate(args):
        if isinstance(arg, dict):
            ref_curve = arg
            label = ''
        elif isinstance(arg, tuple) and len(arg) == 2:
            ref_curve, label = arg
        else:
            print(f"Invalid argument {arg} - must be dict or (dict, label)")
            continue

        ref_params = sorted(ref_curve.keys())
        ref_values = np.array([ref_curve[p] for p in ref_params])

        marker = markers[ind % len(markers)]
        # ax.semilogy(ref_params, ref_values, marker=marker, label=label)
        ax.errorbar(ref_params, ref_values, yerr=3*sigma, marker=marker, label=label, capsize=4)
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlim(left=min(ref_params))
    ax.set_ylim(top=1)
    ax.set_xlabel(r'Physical error rate $\varepsilon$')
    ax.set_ylabel('FER')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    if matplotlib.get_backend() == "Agg":
        plt.savefig(f'{datetime.now()}.png')
    else:
        plt.show()



def save(sim, location, name, notes=''):
    '''
    Saves simulation results into specified directory.

    Parameters:
    ---------------------------------------------------------------
    sim         (QEC.Simulation_Env)    - C++ Simulation instance
    location    (string)                - Path to parent directory
    name        (string)                - Name of results directory
    notes       (string)                - Annotations for meta.txt
    '''

    # Create directory at specified location
    timestamp = datetime.now()
    dir = os.path.join(location, name)
    os.makedirs(dir, exist_ok=False)

    # Save metadata in .txt file
    meta_file = os.path.join(dir, 'meta.txt')
    with open(meta_file, 'w') as file:
        file.write(f'QEC Simulation {timestamp.strftime("%d.%m.%Y %H:%M")}\n')
        file.write('===============================\n\n')

        file.write('Code\n')
        file.write('-------------------------------\n')
        file.write(f'n: {sim.n}\n')
        file.write(f'k: {sim.k}\n')
        file.write(f'm: {sim.m}\n')
        if (sim.decode_on_alternative_PCM):
            file.write('\n')
            file.write(f'n_dec: {sim.n_dec}\n')
            file.write(f'm_dec: {sim.m_dec}\n')
        file.write('\n\n')

        file.write('Decoder\n')
        file.write('-------------------------------\n')
        file.write(f'decoder_type: {sim.decoder_type}\n')
        if (sim.ensemble_type != 'none'):
            file.write(f'ensemble_type: {sim.ensemble_type}\n')
            file.write(f'ensemble_size: {sim.ensemble_size}\n')
            if (sim.ensemble_type == 'SCED'):
                file.write(f'(SCED) d_c: {int(sim.decoder_params["d_c"])}\n')
                file.write(f'(SCED) m_s: {int(sim.decoder_params["m_s"])}\n')
                if (not sim.decoder_params['separate_checks']):
                    file.write(f'(SCED) Y-checks appended\n')
                if (sim.decoder_params['avoid_mixed_cycles']): 
                    file.write(f'(SCED) avoid_mixed_cycles\n')
                if (sim.decoder_params['overcomplete_subcode']):
                    file.write(f'(SCED) overcomplete_subcode\n')
                    file.write(f'(SCED) m_oc: {int(sim.decoder_params["m_oc"])}\n')
        file.write(f'max_iterations: {sim.max_iterations}\n')
        epsilon_0 = 'epsilon' if math.isnan(sim.epsilon_0) else sim.epsilon_0
        file.write(f'epsilon_0: {epsilon_0}\n')
        file.write('\n\n')

        if not sim.use_depolarizing_channel or sim.use_phenomenological_noise:
            file.write('Channel\n')
            file.write('-------------------------------\n')
            if (not sim.use_depolarizing_channel):
                file.write(f'pX: {sim.pX}\n')
                file.write(f'pZ: {sim.pZ}\n')
                file.write(f'pY: {sim.pY}\n')
            if (sim.use_phenomenological_noise):
                file.write(f'use_phenomenological_noise\n')
                file.write(f'pS: {sim.pS}\n')
            file.write('\n\n')

        file.write('Simulation\n')
        file.write('-------------------------------\n')
        file.write(f'target_errors: {sim.target_errors}\n')
        file.write(f'max_transmissions: {sim.max_transmissions}\n')
        file.write('\n\n')

        file.write(f'NOTES: {notes}')


    # FER.json
    #------------------------------------------------------
    # Round FER dict to avoid precision errors
    FER = {}
    for key, value in sim.FER.items():
        FER[round(key, 6)] = round(value, 6)
    # Save FER dict in .json file
    FER_file = os.path.join(dir, 'FER.json')
    with open(FER_file, 'w') as file:
        json.dump(FER, file, indent=4)


    # stats.json
    #-------------------------------------------------------
    # Round stats dict to avoid precision errors    
    stats = {}
    for stat_key, stat_dict in sim.stats.items():
        stats[stat_key] = {}
        for key, value in stat_dict.items():
            stats[stat_key][round(key, 6)] = round(value, 6)
    # Save stats dict in .json file
    stats_file = os.path.join(dir, 'stats.json')
    with open(stats_file, 'w') as file:
        json.dump(stats, file, indent=4)


    # ensemble_stats.json
    #--------------------------------------------------------
    if (sim.ensemble_type != 'none'):
        # Round ensemble_stats dict to avoid precision errors
        ensemble_stats = {}
        for key, stats in sim.ensemble_stats.items():
            ensemble_stats[round(key, 6)] = stats
        # Save ensemble_stats dict in .json file
        ensemble_stats_file = os.path.join(dir, 'ensemble_stats.json')
        with open(ensemble_stats_file, 'w') as file:
            json.dump(ensemble_stats, file, indent=4)


    # SCED specific files
    #--------------------------------------------------------
    if (sim.ensemble_type == 'SCED'):
        if (sim.decoder_params['overcomplete_subcode']):
            # Save search space statistics of subcodes:
            #----------------------------------------------
            # Convert nested list into dict for readability
            search_space_stats_list = sim.get_SCED_stats()
            search_space_stats = {}
            for subcode_index, submatrix_list in enumerate(search_space_stats_list):
                submatrix_dict = {}
                for submatrix_index, stats_list in enumerate(submatrix_list):
                    stats_dict = {}
                    submatrix_key = 'H_x' if submatrix_index == 0 else 'H_z'
                    for stats in stats_list:
                        stats_dict[stats[1]] = stats[0]
                    submatrix_dict[submatrix_key] = stats_dict
                search_space_stats['s' + str(subcode_index)] = submatrix_dict
            # Save search_space_stats dict in .json file
            search_space_stats_file = os.path.join(dir, 'search_space_stats.json')
            with open(search_space_stats_file, 'w') as file:
                json.dump(search_space_stats, file, indent=4)
                



def load_FER(dir):
    '''
    Loads FER dict from .json file.

    Parameters:
    -----------------------------------------------
    dir     (string)    - Path to results directory
    '''

    try:
        # Load dict from .json
        FER_file = os.path.join(dir, 'FER.json')
        with open(FER_file, 'r') as file:
            dict = json.load(file)
        # Convert keys (epsilon) back to floats
        FER = {float(key): value for key, value in dict.items()}
        return FER
    
    except: return {}



def load_stats(dir):
    '''
    Loads stats dict from .json file.

    Parameters:
    -----------------------------------------------
    dir     (string)    - Path to results directory
    '''

    try:
        # Load dict from .json
        stats_file = os.path.join(dir, 'stats.json')
        with open(stats_file, 'r') as file:
            dict = json.load(file)
        # Convert keys (epsilon) back to floats
        stats = {}
        for stat_key, stat_dict in dict.items():
            stats[stat_key] = {float(key): value for key, value in stat_dict.items()}
        return stats
    
    except: return {}



def load_ensemble_stats(dir):
    '''
    Loads ensemble_stats dict from .json file

    Parameters:
    -----------------------------------------------
    dir     (string)    - Path to results directory
    '''

    try:
        # Load dict from .json
        ensemble_stats = os.path.join(dir, 'ensemble_stats.json')
        with open(ensemble_stats, 'r') as file:
            dict = json.load(file)
        # Convert keys (epsilon) bakc to floats
        ensemble_stats = {}
        for key, stats in dict.items():
            ensemble_stats[float(key)] = stats
        return ensemble_stats
    
    except: return {}