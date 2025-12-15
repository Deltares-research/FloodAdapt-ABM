import numpy as np


class ABMSimulator:
    def __init__(self, ds_impacts, fa, years, dt, slr_scenario_name, n_seq, damage_threshold=0.3, seed=42, start_year=2020):
        """
        ds_impacts: xarray dataset with impact values
        fa: FloodAdapt instance
        years: number of years to simulate
        dt: timestep in years
        slr_scenario_name: name of SLR scenario
        n_seq: number of event sequences to generate
        damage_threshold: fraction of max potential damage to trigger floodproofing
        seed: random seed for reproducibility
        """
        self.ds_impacts = ds_impacts
        self.fa = fa
        self.years = years
        self.dt = dt
        self.slr_scenario_name = slr_scenario_name
        self.n_seq = n_seq
        self.damage_threshold = damage_threshold
        self.seed = seed
        self.n_households = len(ds_impacts.object_id)
        self.strategies = ds_impacts.strategy.values
        self.event_names = ds_impacts.event.values
        self.max_pot_dmg = ds_impacts.object_id.attrs['max_pot_dmg']
        self.start_year = start_year
        # Generate event sequences
        self.occ, self.sequences, self.event_names, self.probs = self.create_event_sequence()

    def create_event_sequence(self):
        """
        Create event sequences for Monte Carlo simulation.
        Returns:
            occ: boolean array [n_seq, years, n_events]
            sequences: list of n_seq elements, each is list of years with event names
            event_names: list of event names
            probs: list of event probabilities
        """
        # Get event set from FloodAdapt database
        # Assume the first event set in the database is used
        probs = []
        event_ids = []
        for i, event in enumerate(self.ds_impacts.event.values):
            freq = self.ds_impacts.event.attrs["freq"][i]
            if freq <= 1.0 / self.dt:
                probs.append(freq * self.dt)
                event_ids.append(event)
        occ = self.generate_event_sequences(probs, years=self.years, n_seq=self.n_seq, seed=self.seed)
        sequences = self.occurrences_to_sequences(occ, event_ids=event_ids)
        return occ, sequences, event_ids, probs

    @staticmethod
    def generate_event_sequences(event_probs, years=30, n_seq=1000, seed=None):
        rng = np.random.default_rng(seed)
        p = np.asarray(event_probs, dtype=float)
        draws = rng.random((n_seq, years, p.size))
        return draws < p[np.newaxis, np.newaxis, :]

    @staticmethod
    def occurrences_to_sequences(occ, event_ids=None):
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

    def get_slr_for_year(self, year):
        """Interpolate SLR value for a given year using FloodAdapt."""
        slr_value = self.fa.interp_slr(slr_scenario=self.slr_scenario_name, year=self.start_year + year)
        return slr_value


    def slr_damage_lookup(self, slr_value, event, strategies, method='linear'):
        """
        Vectorized lookup/interpolation of damage for a given SLR value, event, and a list of strategies (one per object_id).
        Returns an array of damages for each object_id.
        Args:
            slr_value: float, the SLR value to interpolate to
            event: str, event name
            strategies: list/array of str, strategy for each object_id (length = n_households)
            method: interpolation method ('linear', 'nearest', etc.)
        Returns:
            damages: np.ndarray of shape (n_households,)
        """
        import xarray as xr
        slr_sim = self.ds_impacts['slr'].values
        object_ids = self.ds_impacts['object_id'].values
        strat_da = xr.DataArray(strategies, dims=['object_id'], coords={'object_id': object_ids})
        damages_da = self.ds_impacts.sel(event=event).sel(strategy=strat_da)["total_damage"]
        damages_matrix = damages_da.values  # shape (n_slr, n_obj)
        damages = self._interpolate_damages(slr_sim, damages_matrix, slr_value, method)
        return damages

    # _extract_damage is now obsolete for this use case and can be removed or left for backward compatibility.


    @staticmethod
    def _interpolate_damages(slr_sim, damages_matrix, slr_value, method):
        """
        Interpolate damages for all objects at once.
        slr_sim: 1D array of simulated SLR values
        damages_matrix: 2D array (n_obj, n_slr)
        slr_value: float
        method: interpolation method
        Returns: 1D array (n_obj,)
        """
        import numpy as np
        from scipy.interpolate import interp1d
        # damages_matrix shape: (n_obj, n_slr)
        n_obj, n_slr = damages_matrix.shape
        if method == 'linear':
            # np.interp does not support 2D arrays, so loop over objects
            damages = np.empty(damages_matrix.shape[0])
            for i in range(damages_matrix.shape[0]):
                damages[i] = np.interp(slr_value, slr_sim, damages_matrix[i, :])
            return damages
        elif method == 'nearest':
            idx = (np.abs(slr_sim - slr_value)).argmin()
            return damages_matrix[:, idx]
        elif method == 'cubic':
            if len(slr_sim) < 4:
                raise ValueError('Cubic interpolation requires at least 4 SLR points.')
            damages = np.empty(n_obj)
            for i in range(n_obj):
                f = interp1d(slr_sim, damages_matrix[i, :], kind='cubic', fill_value='extrapolate')
                damages[i] = float(f(slr_value))
            return damages
        elif method == 'floor':
            slr_sim_sorted = np.sort(slr_sim)
            sort_idx = np.argsort(slr_sim)
            idxs = np.where(slr_sim_sorted <= slr_value)[0]
            if len(idxs) == 0:
                idx = 0
            else:
                idx = idxs[-1]
            return damages_matrix[:, sort_idx[idx]]
        elif method == 'ceil':
            slr_sim_sorted = np.sort(slr_sim)
            sort_idx = np.argsort(slr_sim)
            idxs = np.where(slr_sim_sorted >= slr_value)[0]
            if len(idxs) == 0:
                idx = -1
            else:
                idx = idxs[0]
            return damages_matrix[:, sort_idx[idx]]
        else:
            raise ValueError(f'Unknown interpolation method: {method}')

    def run(self):
        """
        Run the ABM simulation for all sequences and households using vectorized calculations.
        Returns:
            damage_history: [sequence, household, year] array of damages
            floodproofed: [sequence, household, year] boolean array of floodproofing state
        """
        damage_history = np.zeros((self.n_seq, self.n_households, self.years))
        floodproofed = np.zeros((self.n_seq, self.n_households, self.years), dtype=bool)

        for seq_idx in range(self.n_seq):
            is_floodproofed = np.zeros(self.n_households, dtype=bool)
            for year_idx in range(self.years):
                slr_val = self.get_slr_for_year(year_idx)
                year_events = self.sequences[seq_idx][year_idx]
                # Vectorized damage calculation for all households
                # For each event, get damages for all households, then sum over events
                total_damage = np.zeros(self.n_households)
                for event in year_events:
                    # For all households, select strategy based on floodproofed state
                    strats = np.where(is_floodproofed, 'floodproof_all_0', 'no_measures')
                    # Vectorized lookup for all households
                    # Use list comprehension to call slr_damage_lookup for all households
                    damages = self.slr_damage_lookup(
                            slr_val,
                            event,
                            strats,
                            method='linear'
                            )

                    total_damage += damages
                damage_history[seq_idx, :, year_idx] = total_damage
                floodproofed[seq_idx, :, year_idx] = is_floodproofed
                # Vectorized floodproofing decision
                not_floodproofed = ~is_floodproofed
                with_pot_dmg = self.max_pot_dmg > 0
                threshold_exceeded = np.zeros(self.n_households, dtype=bool)
                valid = not_floodproofed & with_pot_dmg
                threshold_exceeded[valid] = (total_damage[valid] / self.max_pot_dmg[valid]) > self.damage_threshold
                is_floodproofed = is_floodproofed | threshold_exceeded
        
        self.damage_history = damage_history
        self.floodproofed = floodproofed
        
        return damage_history, floodproofed
    
    def plot_event_sequences(self, seq_max=20):
        """
        Plot the first seq_max event sequences as a raster plot.
        """
        import matplotlib.pyplot as plt
        event_names = self.event_names
        sequences = self.sequences
        years = self.years
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        colors = {event_id: plt.cm.tab10(i) for i, event_id in enumerate(event_names)}
        yticklabels = []
        for seq_idx in range(min(seq_max, len(sequences))):
            ax = axes
            for year_idx, year_events in enumerate(sequences[seq_idx]):
                for marker_offset, event_id in enumerate(sorted(year_events)):
                    ax.scatter(year_idx, marker_offset*0.15+seq_idx,
                              color=colors[event_id],
                              s=50,
                              marker='o',
                              label=event_id if year_idx == 0 else "")
            yticklabels.append(f"Seq {seq_idx}")
        ax.set_xlabel('Year')
        ax.set_yticks([i for i in range(min(seq_max, len(sequences)))])
        ax.set_yticklabels(yticklabels)
        ax.set_xlim(-0.5, years - 0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, years, 5))
        handles = [plt.scatter([], [], color=colors[e], s=200, marker='o') for e in event_names]
        fig.legend(handles, event_names, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(event_names))
        plt.tight_layout()
        plt.show()