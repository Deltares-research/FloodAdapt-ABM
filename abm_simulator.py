import numpy as np


class ABMSimulator:

    def __init__(self, ds_impacts, times, slr_times, no_seq, damage_threshold=0.3, seed=42):
        self.ds_impacts = ds_impacts
        self.times = times
        self.dt = self.times[1] - self.times[0]
        self.time_steps = len(self.times)
        self.slr_times = slr_times
        self.no_seq = no_seq
        self.damage_threshold = damage_threshold
        self.seed = seed
        self.n_households = len(ds_impacts.object_id)
        self.strategies = ds_impacts.strategy.values
        self.event_names = ds_impacts.event.values
        self.max_pot_dmg = ds_impacts.object_id.attrs['max_pot_dmg']
        # Generate event sequences
        self.sequences= self.create_event_sequences()

    def create_event_sequences(self):
        """
        Combines event probability calculation, event occurrence simulation, and sequence construction.
        Returns:
            sequences: list of n_seq elements, each is list of years with event names
        """
        probs = []
        event_ids = []
        for i, event in enumerate(self.ds_impacts.event.values):
            freq = self.ds_impacts.event.attrs["freq"][i]
            if freq <= 1.0 / self.dt:
                probs.append(freq * self.dt)
                event_ids.append(event)
        # Simulate event occurrences
        rng = np.random.default_rng(self.seed)
        p = np.asarray(probs, dtype=float)
        draws = rng.random((self.no_seq, len(self.times), p.size))
        occ = draws < p[np.newaxis, np.newaxis, :]
        # Convert occurrences to sequences
        n_sims, years, n_events = occ.shape
        sequences = []
        for s in range(n_sims):
            sim_seq = []
            for y in range(years):
                evs = [event_ids[i] for i in range(n_events) if occ[s, y, i]]
                sim_seq.append(evs)
            sequences.append(sim_seq)
        return sequences


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
            f = interp1d(slr_sim, damages_matrix, kind='linear', axis=1, bounds_error=False, fill_value='extrapolate')
            damages = f(slr_value)
            return damages
        elif method == 'nearest':
            idx = (np.abs(slr_sim - slr_value)).argmin()
            return damages_matrix[:, idx]
        elif method == 'cubic':
            if len(slr_sim) < 4:
                raise ValueError('Cubic interpolation requires at least 4 SLR points.')
            f = interp1d(slr_sim, damages_matrix, kind='cubic', axis=1, bounds_error=False, fill_value='extrapolate')
            damages = f(slr_value)
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

    def run(self, method='linear'):
        """
        Run the ABM simulation for all sequences and households using vectorized calculations.
        Returns:
            damage_history: [sequence, household, year] array of damages
            floodproofed: [sequence, household, year] boolean array of floodproofing state
        """
        self._compute_baseline_no_floodproofing()
        damage_history, damage_history_per_event, floodproofed = self._calculate_damage_history(floodproofing=True, method=method)
        self.damage_history = damage_history
        self.damage_history_per_event = damage_history_per_event
        self.floodproofed = floodproofed
        self.has_run = True

    def _compute_baseline_no_floodproofing(self):
        """
        Compute and store baseline damages (per event and total) for all sequences,
        assuming 'no_measures' strategy for all households and all years (no floodproofing).
        Stores:
            self.baseline_damage_history: [sequence, household, year] array of damages
            self.baseline_damage_history_per_event: [sequence, household, year, event] array of per-event damages
        """
        baseline_damage_history, baseline_damage_history_per_event, _ = self._calculate_damage_history(floodproofing=False, method='linear')
        self.baseline_damage_history = baseline_damage_history
        self.baseline_damage_history_per_event = baseline_damage_history_per_event
           
    def plot_event_damage_timeseries(self, seq_id, figsize=(12, 6)):
        """
        Plots a time series for a given sequence id, showing:
        - For each time step (year), a stacked column of dots for each event that occurred (stacked from bottom)
        - A bar plot of the total damage for that time step
        Args:
            seq_id (int): The sequence index to plot
            figsize (tuple): Figure size for the plot
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np

        # Use self.times for the time axis
        times = np.array(self.times)
        # Get the event sequence for the given seq_id
        seq = self.sequences[seq_id]
        # seq is a list of event names (or ids) for each time step
        # If multiple events per year, seq should be a list of lists
        # If not, convert to list of lists
        if not isinstance(seq[0], (list, np.ndarray)):
            seq = [[e] if e is not None else [] for e in seq]

        # Get damages for each time step (sum of all events in that year)
        # Assume self.occ is shape (n_seq, years, n_events), 1 if event occurred
        # and self.ds_impacts has damages for each event
        damages = []
        for t, events in enumerate(seq):
            total_damage = 0
            for event in events:
                # If event is index, get name
                if isinstance(event, (int, np.integer)):
                    event_name = self.event_names[event]
                else:
                    event_name = event
                # Get damage for this event (assume damages are in ds_impacts)
                # This may need to be adapted to your data structure
                try:
                    dmg = float(self.ds_impacts.sel(event=event_name)["total_damage"].values.sum())
                except Exception:
                    dmg = 0
                total_damage += dmg
            damages.append(total_damage)

        # Prepare event color mapping
        unique_events = list({e for events in seq for e in events})
        cmap = cm.get_cmap('tab20', len(unique_events))
        event2color = {e: cmap(i) for i, e in enumerate(unique_events)}

        fig, ax1 = plt.subplots(figsize=figsize)

        # Plot stacked dots for events
        for t, events in enumerate(seq):
            for i, event in enumerate(events):
                color = event2color[event]
                ax1.scatter(t, i, color=color, s=60, marker='o', edgecolor='k', zorder=3)

        # Set y-limits for event stack
        max_stack = max(len(events) for events in seq)
        ax1.set_ylim(-0.5, max_stack + 0.5)
        ax1.set_ylabel('Events (stacked dots)')
        ax1.set_xticks(np.arange(len(times)))
        ax1.set_xticklabels(times, rotation=45)

        # Twin axis for damage bar plot
        ax2 = ax1.twinx()
        ax2.bar(np.arange(len(times)), damages, alpha=0.3, color='red', width=0.7, zorder=2)
        ax2.set_ylabel('Total Damage')

        # Legend for events
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=event2color[e], markeredgecolor='k', label=str(e), markersize=8) for e in unique_events]
        ax1.legend(handles=handles, title='Event', bbox_to_anchor=(1.05, 1), loc='upper left')

        ax1.set_title(f'Time Series of Events and Damages (Sequence {seq_id})')
        fig.tight_layout()
        plt.show()
        
    def plot_total_damage_statistics(self, figsize=(12, 6)):
        """
        Plot total damages statistics (mean and 5-95 percentile) over all sequences for:
        - Actual simulation (with floodproofing)
        - Baseline (no floodproofing)
        Shows average line and hatched area for 5-95 percentile for both scenarios.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        times = np.array(self.times)
        # Aggregate over households (sum damages per year per sequence)
        sim_total = self.damage_history.sum(axis=1)  # shape: (n_seq, years)
        base_total = self.baseline_damage_history.sum(axis=1)  # shape: (n_seq, years)

        def stats(arr):
            mean = np.mean(arr, axis=0)
            p5 = np.percentile(arr, 5, axis=0)
            p95 = np.percentile(arr, 95, axis=0)
            return mean, p5, p95

        sim_mean, sim_p5, sim_p95 = stats(sim_total)
        base_mean, base_p5, base_p95 = stats(base_total)

        fig, ax = plt.subplots(figsize=figsize)
        # Baseline (no floodproofing)
        ax.plot(times, base_mean, label='Baseline (no floodproofing)', color='tab:blue')
        ax.fill_between(times, base_p5, base_p95, color='tab:blue', alpha=0.2, hatch='//', edgecolor='tab:blue', linewidth=0.0)
        # Actual simulation
        ax.plot(times, sim_mean, label='Simulation (with floodproofing)', color='tab:orange')
        ax.fill_between(times, sim_p5, sim_p95, color='tab:orange', alpha=0.2, hatch='\\', edgecolor='tab:orange', linewidth=0.0)

        ax.set_xlabel('Time')
        ax.set_ylabel('Total Damage')
        ax.set_title('Total Damages: Simulation vs Baseline')
        ax.legend()
        fig.tight_layout()
        plt.show()
        
    def _calculate_damage_history(self, floodproofing: bool, method: str = 'linear'):
        """
        Shared logic for calculating damage history and per-event damage.
        If floodproofing is True, applies floodproofing logic; otherwise, always uses 'no_measures'.
        Returns:
            damage_history: [sequence, household, time] array
            damage_history_per_event: [sequence, household, time, event] array
            floodproofed: [sequence, household, time] boolean array (None if floodproofing is False)
        """
        n_events = len(self.event_names)
        event_names_list = list(self.event_names)
        damage_history = np.zeros((self.no_seq, self.n_households, self.time_steps))
        damage_history_per_event = np.zeros((self.no_seq, self.n_households, self.time_steps, n_events))
        floodproofed = np.zeros((self.no_seq, self.n_households, self.time_steps), dtype=bool) if floodproofing else None

        for seq_idx in range(self.no_seq):
            if floodproofing:
                print(f"Evaluating sequence {seq_idx+1}/{self.no_seq}...")
            else:
                print(f"[BASELINE] Evaluating sequence {seq_idx+1}/{self.no_seq}...")
            is_floodproofed = np.zeros(self.n_households, dtype=bool)
            for ti in range(self.time_steps):
                slr_val = self.slr_times[ti]
                year_events = self.sequences[seq_idx][ti]
                total_damage = np.zeros(self.n_households)
                year_event_damage = np.zeros((self.n_households, n_events))
                for event in year_events:
                    if floodproofing:
                        strats = np.where(is_floodproofed, 'floodproof_all_0', 'no_measures')
                    else:
                        strats = np.full(self.n_households, 'no_measures', dtype=object)
                    damages = self.slr_damage_lookup(
                        slr_val,
                        event,
                        strats,
                        method=method
                    )
                    total_damage += damages
                    if event in event_names_list:
                        event_idx = event_names_list.index(event)
                        year_event_damage[:, event_idx] = damages
                damage_history[seq_idx, :, ti] = total_damage
                damage_history_per_event[seq_idx, :, ti, :] = year_event_damage
                if floodproofing:
                    floodproofed[seq_idx, :, ti] = is_floodproofed
                    # Vectorized floodproofing decision
                    not_floodproofed = ~is_floodproofed
                    with_pot_dmg = self.max_pot_dmg > 0
                    threshold_exceeded = np.zeros(self.n_households, dtype=bool)
                    valid = not_floodproofed & with_pot_dmg
                    threshold_exceeded[valid] = (total_damage[valid] / self.max_pot_dmg[valid]) > self.damage_threshold
                    is_floodproofed = is_floodproofed | threshold_exceeded
        return damage_history, damage_history_per_event, floodproofed