# Imports

import numpy as np

from flood_adapt.objects import (
    EventSet
)

def create_event_sequence(
        fn_event_set,   
        years=30, 
        n_seq=20, 
        dt=1,
        seed=42):
    
    """
    Create event sequences for Monte Carlo simulation.
    
    Parameters:
    DATA_DIR: Path to the database directory
    site: Site name
    name_event_set: Name of the event set
    years: Number of years to simulate
    n_seq: Number of sequences to simulate
    dt: Time step in years
    seed: Random seed for reproducibility
    
    Returns:
    sequences: List of event sequences for each simulation
    """
    
    event_set = EventSet.load_file(fn_event_set)
    probs = []
    event_ids = []
    for event in event_set.sub_events:
        if event.frequency <= 1./dt:
            probs.append(event.frequency*dt)
            event_ids.append(event.name)
            
    occ = generate_event_sequences(probs, years=years, n_seq=n_seq, seed=seed)
    sequences = occurrences_to_sequences(occ, event_ids=event_ids)

    return occ, sequences, event_ids, probs


def generate_event_sequences(event_probs, years=30, n_seq=1000, seed=None):
    """
    event_probs: sequence of annual occurrence probabilities (len = n_events)
    returns: boolean array shape (n_sims, years, n_events) where True=event occurs that year
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(event_probs, dtype=float)
    draws = rng.random((n_seq, years, p.size))
    return draws < p[np.newaxis, np.newaxis, :]

def occurrences_to_sequences(occ, event_ids=None):
    """
    occ: boolean array from generate_event_sequences
    event_ids: optional list of identifiers for events (len = n_events)
    returns: list of length n_sims; each element is list of length years with lists of event ids occurring that year
    """
    n_sims, years, n_events = occ.shape
    if event_ids is None:
        event_ids = [f"event_{i}" for i in range(n_events)]
    sequences = []
    for s in range(n_sims):
        sim_seq = []
        for y in range(years):
            evs = [event_ids[i] for i in range(n_events) if occ[s, y, i]]
            sim_seq.append(evs)
        sequences.append(sim_seq)
    return sequences

