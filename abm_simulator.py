import numpy as np
import xarray as xr

class ABMSimulator:

    def __init__(self, ds_impacts, times, slr_values, no_seq, damage_threshold=0.3, seed=42):
        self.ds_impacts = ds_impacts
        self.times = times
        self.dt = self.times[1] - self.times[0]
        self.time_steps = len(self.times)
        self.slr_values = slr_values
        self.no_seq = no_seq
        self.damage_threshold = damage_threshold
        self.seed = seed
        self.n_households = len(ds_impacts.object_id)
        self.strategies = ds_impacts.strategy.values
        self.event_names = ds_impacts.event.values
        self.max_pot_dmg = ds_impacts.object_id.attrs['max_pot_dmg']
        # Generate event sequences
        self.sequences = self.create_event_sequences()

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
            # if freq <= 1.0 / self.dt:
            probs.append(freq * self.dt)
            event_ids.append(event)
        # Simulate event occurrences
        rng = np.random.default_rng(self.seed)
        p = np.asarray(probs, dtype=float)
        draws = rng.random((self.no_seq, len(self.times), p.size))
        occurrences = draws < p[np.newaxis, np.newaxis, :]
        # Convert occurrences to sequences
        n_sims, years, n_events = occurrences.shape
        sequences = []
        for s in range(n_sims):
            sim_seq = []
            for y in range(years):
                evs = [event_ids[i] for i in range(n_events) if occurrences[s, y, i]]
                sim_seq.append(evs)
            sequences.append(sim_seq)
        return sequences
    
    def slr_damage_lookup(self, slr_values, event_names_list, strategy, method='linear'):
        """
        Vectorized lookup/interpolation of damage for a given SLR value, event, and a list of strategies (one per object_id).
        Returns an array of damages for each object_id.
        Args:
            slr_value: float, the SLR value to interpolate to
            event: str, event name
            strategy: str, strategy applied to all objects
            method: interpolation method ('linear', 'nearest', etc.)
        Returns:
            damages: np.ndarray of shape (n_households, n_events, n_slr_values)
        """
    
        slr_sim = self.ds_impacts['slr'].values
        object_ids = self.ds_impacts['object_id'].values
        damage_matrix = np.empty((len(object_ids), len(event_names_list), len(slr_values)))
        for ievent, event in enumerate(event_names_list):
            damages_da = self.ds_impacts.sel(event=event).sel(strategy=strategy)["total_damage"]
            damages_values = damages_da.values  # shape (n_slr, n_obj)
            damage_matrix[:,ievent,:] = self._interpolate_damages(slr_sim, damages_values, slr_values, method)
        return damage_matrix


    @staticmethod
    def _interpolate_damages(slr_sim, damages_values, slr_values, method):
        """
        Interpolate damages for all objects at once.
        slr_sim: 1D array of simulated SLR values from the impact matrix created in step 1 
        damages_values: 2D array (n_obj, n_slr), using slr from impact matrix
        slr_values: 1D array (n_slr_va;lues,) of SLR values to interpolate to
        method: str, interpolation method
        Returns: 1D array (n_obj,)
        """
        import numpy as np
        from scipy.interpolate import interp1d
        # damages_matrix shape: (n_obj, n_slr)
        if method == 'linear':
            f = interp1d(slr_sim, damages_values, kind='linear', axis=1, bounds_error=False, fill_value='extrapolate')
            damages = f(slr_values)
            return damages
        elif method == 'nearest':
            idx = (np.abs(slr_sim - slr_values)).argmin()
            return damages_values[:, idx]
        elif method == 'cubic':
            if len(slr_sim) < 4:
                raise ValueError('Cubic interpolation requires at least 4 SLR points.')
            f = interp1d(slr_sim, damages_values, kind='cubic', axis=1, bounds_error=False, fill_value='extrapolate')
            damages = f(slr_values)
            return damages
        elif method == 'floor':
            slr_sim_sorted = np.sort(slr_sim)
            sort_idx = np.argsort(slr_sim)
            idxs = np.where(slr_sim_sorted <= slr_values)[0]
            if len(idxs) == 0:
                idx = 0
            else:
                idx = idxs[-1]
            return damages_values[:, sort_idx[idx]]
        elif method == 'ceil':
            slr_sim_sorted = np.sort(slr_sim)
            sort_idx = np.argsort(slr_sim)
            idxs = np.where(slr_sim_sorted >= slr_values)[0]
            if len(idxs) == 0:
                idx = -1
            else:
                idx = idxs[0]
            return damages_values[:, sort_idx[idx]]
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
           
    def plot_event_damage_timeseries(self, seq_id, figsize=(12, 10)):
        """
        Plots a time series for a given sequence id, showing:
        - For each time step (year), a stacked column of dots for each event that occurred (stacked from bottom)
        - A bar plot of the total damage for that time step (from simulation results)
        Args:
            seq_id (int): The sequence index to plot
            figsize (tuple): Figure size for the plot
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np
        import matplotlib.colors as mcolors

        # Check if simulation has been run
        if not hasattr(self, 'has_run') or not getattr(self, 'has_run', False):
            raise RuntimeError("Simulation has not been run. Please call the 'run' method before plotting.")

        # Use self.times for the time axis
        times = np.array(self.times)
        # Get the event sequence for the given seq_id
        seq = self.sequences[seq_id]
        # seq is a list of event names (or ids) for each time step
        # If multiple events per year, seq should be a list of lists
        if not isinstance(seq[0], (list, np.ndarray)):
            seq = [[e] if e is not None else [] for e in seq]

        # Use calculated damages from simulation (sum over households for each year)
        damages = self.damage_history[seq_id].sum(axis=0)
        # Baseline damages for this sequence (sum over households for each year)
        if hasattr(self, 'baseline_damage_history'):
            baseline_damages = self.baseline_damage_history[seq_id].sum(axis=0)
        else:
            baseline_damages = None

        # Prepare event frequency mapping (use log scale for color)
        # Get all unique events in this sequence
        unique_events = list({e for events in seq for e in events})
        # Get event frequencies from ds_impacts.event.attrs['freq']
        event_freq_dict = {}
        if hasattr(self.ds_impacts, 'event') and hasattr(self.ds_impacts.event, 'values') and hasattr(self.ds_impacts.event, 'attrs'):
            all_events = self.ds_impacts.event.values
            all_freqs = self.ds_impacts.event.attrs.get('freq', None)
            if all_freqs is not None:
                for e, f in zip(all_events, all_freqs):
                    event_freq_dict[e] = f
        # For events not in ds_impacts, assign a small frequency
        min_freq = min(event_freq_dict.values()) if event_freq_dict else 1e-6
        event_freqs = [event_freq_dict.get(e, min_freq) for e in unique_events]

        # Log scale for color mapping, but colorbar ticks show actual frequencies
        cmap = cm.get_cmap('plasma_r')
        log_freqs = np.log10(event_freqs)
        # Use a consistent normalization for the color mapping
        min_logf = np.min(log_freqs)
        max_logf = np.max(log_freqs)
        norm = mcolors.Normalize(vmin=min_logf, vmax=max_logf)
        event2color = {e: cmap(norm(np.log10(event_freq_dict.get(e, min_freq)))) for e in unique_events}

        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=figsize)
        # 4 rows: colorbar, events, damages, floodproofed; 1 column
        gs = gridspec.GridSpec(4, 1, height_ratios=[0.5, 1, 4, 1.5], hspace=0.15)
        cax = fig.add_subplot(gs[0])
        ax_events = fig.add_subplot(gs[1], sharex=None)
        ax = fig.add_subplot(gs[2], sharex=ax_events)
        ax_floodproof = fig.add_subplot(gs[3], sharex=ax_events)

        # Plot event dots in the top axis (stacked vertically for same time step, no spacing)
        for t, events in enumerate(seq):
            n_events = len(events)
            if n_events == 0:
                continue
            for i, event in enumerate(events):
                color = event2color[event]
                ax_events.scatter(t, i, color=color, s=60, marker='o', edgecolor='k', zorder=3)

        # Set y-ticks for event axis to show up to max number of events
        max_stack = max(len(events) for events in seq)
        ax_events.set_ylim(-0.5, max_stack - 0.5 if max_stack > 0 else 0.5)
        ax_events.set_ylabel('Events')
        ax_events.set_xlim(-0.5, len(times) - 0.5)
        # Set x-ticks (but not labels) on all axes except bottom
        x = np.arange(len(times))
        ax_events.set_xticks(x)
        ax_events.set_xticklabels([])
        ax_events.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
        # Hide y-ticks on event axis
        ax_events.set_yticks([])
        ax_events.tick_params(axis='y', left=False, right=False, labelleft=False)
        # Draw a box around the plot
        for spine in ax_events.spines.values():
            spine.set_visible(True)

        # Plot stacked bar for damages in the bottom axis
        x = np.arange(len(times))
        width = 0.7
        if baseline_damages is not None:
            avoided = baseline_damages - damages
            avoided = np.clip(avoided, 0, None)
            ax.bar(x, damages, width=width, color='tab:orange', label='Actual Damage', zorder=2)
            ax.bar(x, avoided, width=width, bottom=damages, color='tab:blue', label='Avoided Damage (Baseline - Actual)', zorder=1)
        else:
            ax.bar(x, damages, width=width, color='tab:orange', label='Actual Damage', zorder=2)

        ax.set_ylabel('Total Damage (USD)')
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
        ax.legend()

        # Add horizontal colorbar for event frequency above the event plot
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        freq_ticks = np.unique(event_freqs)
        min_tick = np.min(freq_ticks)
        max_tick = np.max(freq_ticks)
        n_ticks = 6 if len(freq_ticks) < 6 else len(freq_ticks)
        all_ticks = np.logspace(np.log10(min_tick), np.log10(max_tick), n_ticks)
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_ticks(np.log10(all_ticks))
        cbar.set_ticklabels([f"{f:.2e}" for f in all_ticks])
        cbar.set_label('Event Frequency', labelpad=8)
        cbar.ax.set_title('Event Frequency', fontsize=10, pad=10)
        # Make colorbar about 1/3 width of the figure
        cax.set_position([0.33, cax.get_position().y0, 0.33, cax.get_position().height])

        # Plot cumulative number of floodproofed buildings in the bottom axis
        if hasattr(self, 'floodproofed') and self.floodproofed is not None:
            # self.floodproofed shape: [sequence, household, time]
            floodproofed_seq = self.floodproofed[seq_id]  # shape: [household, time]
            # Cumulative number of unique buildings floodproofed up to each time
            # A building is floodproofed if it is True at any time up to t
            cumulative_floodproofed = np.cumsum(floodproofed_seq, axis=1) > 0
            n_floodproofed = cumulative_floodproofed.sum(axis=0)
            ax_floodproof.bar(x, n_floodproofed, width=0.7, color='tab:green', label='Cumulative Floodproofed')
            ax_floodproof.set_ylabel('Floodproofed\nBuildings')
            ax_floodproof.set_xticks(x)
            ax_floodproof.set_xticklabels(times, rotation=45)
            ax_floodproof.legend()
        else:
            ax_floodproof.text(0.5, 0.5, 'No floodproofing data', ha='center', va='center')
            ax_floodproof.set_xticks(x)
            ax_floodproof.set_xticklabels(times, rotation=45)
            ax_floodproof.set_ylabel('Floodproofed\nBuildings\n(cumulative)')

        # Make colorbar height smaller
        # Set colorbar axis height to 0.15 of figure height (smaller than default)
        pos = cax.get_position()
        cax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.5])

        fig.tight_layout()
        plt.show()
        
    def plot_total_damage_statistics(self, figsize=(12, 6)):
        """
        Plot total damages statistics as bar plots (stacked actual and avoided) over all sequences for:
        - Actual simulation (with floodproofing)
        - Baseline (no floodproofing)
        Optionally, if percentiles=(min, max) is given, plot bars for those percentiles instead of mean.
        Plots a single figure with two subplots: (1) damages, (2) average number of floodproofed households per year.
        Args:
            figsize: tuple, figure size for the plots
            percentiles: tuple (min, max) or None, percentiles to plot (e.g., (5, 95)). If None, plot mean.
        """
        import matplotlib.pyplot as plt

        times = np.array(self.times)
        n_years = len(times)
        width = 0.7

        # Aggregate over households (sum damages per year per sequence)
        sim_total = self.damage_history.sum(axis=1)  # shape: (n_seq, years)
        base_total = self.baseline_damage_history.sum(axis=1)  # shape: (n_seq, years)

        # Compute mean or percentiles for damages
        def get_bar_data(arr, percentiles=None):
            if percentiles is None:
                mean = np.mean(arr, axis=0)
                return mean
            else:
                pmin = np.percentile(arr, percentiles[0], axis=0)
                pmax = np.percentile(arr, percentiles[1], axis=0)
                return pmin, pmax

        # Prepare percentiles argument
        percentiles = getattr(self, 'percentiles', None) if not hasattr(self, 'percentiles') else None
        import inspect
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        percentiles = values.get('percentiles', None)

        # Compute avoided damage
        if hasattr(self, 'baseline_damage_history') and self.baseline_damage_history is not None:
            if hasattr(self, 'floodproofed') and self.floodproofed is not None:
                avoided = base_total - sim_total
                avoided = np.clip(avoided, 0, None)
            else:
                avoided = np.zeros_like(sim_total)
        else:
            avoided = np.zeros_like(sim_total)

        # Create a single figure with two subplots (side by side)
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # --- Left plot: Damages ---
        if percentiles is None:
            sim_bar = get_bar_data(sim_total, None)
            base_bar = get_bar_data(base_total, None)
            avoided_bar = get_bar_data(avoided, None)
            x = np.arange(n_years)
            # Actual damage
            ax.bar(x, sim_bar, width=width, color='tab:orange', label='Actual Damage', zorder=2)
            # Avoided damage (stacked)
            ax.bar(x, avoided_bar, width=width, bottom=sim_bar, color='tab:blue', label='Avoided Damage (Baseline - Actual)', zorder=1)
        else:
            sim_pmin, sim_pmax = get_bar_data(sim_total, percentiles)
            base_pmin, base_pmax = get_bar_data(base_total, percentiles)
            avoided_pmin = base_pmin - sim_pmin
            avoided_pmin = np.clip(avoided_pmin, 0, None)
            x = np.arange(n_years)
            # Actual damage (lower percentile)
            ax.bar(x, sim_pmin, width=width, color='tab:orange', alpha=0.7, label=f'Actual Damage (P{percentiles[0]})', zorder=2)
            # Avoided damage (lower percentile, stacked)
            ax.bar(x, avoided_pmin, width=width, bottom=sim_pmin, color='tab:blue', alpha=0.7, label=f'Avoided Damage (P{percentiles[0]})', zorder=1)
            # Actual damage (upper percentile, hatched)
            ax.bar(x, sim_pmax, width=width, color='tab:orange', alpha=0.3, label=f'Actual Damage (P{percentiles[1]})', zorder=2, hatch='//', edgecolor='tab:orange')
            # Avoided damage (upper percentile, hatched, stacked)
            avoided_pmax = base_pmax - sim_pmax
            avoided_pmax = np.clip(avoided_pmax, 0, None)
            ax.bar(x, avoided_pmax, width=width, bottom=sim_pmax, color='tab:blue', alpha=0.3, label=f'Avoided Damage (P{percentiles[1]})', zorder=1, hatch='\\', edgecolor='tab:blue')

        ax.set_xlabel('Time')
        ax.set_ylabel('Total Damage ($)')
        ax.set_title('Total Damages: Simulation vs Baseline')
        ax.set_xticks(np.arange(n_years))
        ax.set_xticklabels(times, rotation=45)
        ax.legend()

        # --- Right plot: Average number of floodproofed households per year ---
        if hasattr(self, 'floodproofed') and self.floodproofed is not None:
            avg_floodproofed = np.mean(self.floodproofed, axis=0)  # shape: [household, time]
            avg_floodproofed_per_year = avg_floodproofed.sum(axis=0)  # shape: [time]
            ax2.bar(np.arange(n_years), avg_floodproofed_per_year, width=width, color='tab:green', label='Avg. Floodproofed Households')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Avg. Number of Floodproofed Households')
            ax2.set_title('Average Number of Floodproofed Households per Year')
            ax2.set_xticks(np.arange(n_years))
            ax2.set_xticklabels(times, rotation=45)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No floodproofing data', ha='center', va='center')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Avg. Number of Floodproofed Households')
            ax2.set_title('Average Number of Floodproofed Households per Year')
            ax2.set_xticks(np.arange(n_years))
            ax2.set_xticklabels(times, rotation=45)

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

        # full matrix lookups for no measures and floodproofing all (n_objects, n_events, n_slr_values)
        damage_matrix_no_measures = self.slr_damage_lookup(self.slr_values, event_names_list, 'no_measures', method="linear")
        if floodproofing:
            damage_matrix_floodproofing_all = self.slr_damage_lookup(self.slr_values, event_names_list, 'floodproof_all_0', method="linear")

        for seq_idx in range(self.no_seq):
            is_last = (seq_idx == self.no_seq - 1)
            if floodproofing:
                if is_last:
                    print(f"Evaluating sequence {seq_idx+1}/{self.no_seq}...")
                else:
                    print(f"Evaluating sequence {seq_idx+1}/{self.no_seq}...", end='\r', flush=True)
            else:
                if is_last:
                    print(f"[BASELINE] Evaluating sequence {seq_idx+1}/{self.no_seq}...")
                else:
                    print(f"[BASELINE] Evaluating sequence {seq_idx+1}/{self.no_seq}...", end='\r', flush=True)
            is_floodproofed = np.zeros(self.n_households, dtype=bool)
            for ti in range(self.time_steps):
                year_events = self.sequences[seq_idx][ti]
                total_damage = np.zeros(self.n_households)
                year_event_damage = np.zeros((self.n_households, n_events))
                for event in year_events:
                    if event in event_names_list:
                        event_idx = event_names_list.index(event)
                        damages = damage_matrix_no_measures[:, event_idx, ti]
                        if floodproofing: # apply floodproofing if applicable
                            damages_floodproofing_all = damage_matrix_floodproofing_all[:, event_idx, ti]
                            damages = np.where(is_floodproofed, damages_floodproofing_all, damages)
                        year_event_damage[:, event_idx] = damages
                        total_damage += damages
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