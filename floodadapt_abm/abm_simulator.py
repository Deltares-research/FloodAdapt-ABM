"""
abm_simulator.py
================
.. deprecated:: Phase 3
    ``ABMSimulator`` is the **legacy** pre-refactor threshold-rule simulator.
    It is superseded by :class:`floodadapt_abm.simulation_engine.SimulationEngine`
    with a pluggable :class:`~floodadapt_abm.decision_rule.DecisionRule`
    (use ``ThresholdRule`` to reproduce this class's 0.3-threshold behaviour, or
    ``SEURule`` for the DYNAMO-M science).

    It is intentionally **kept importable** (not moved into ``_core/``) because it
    is still a public/legacy API actively referenced by:

    * ``notebooks/archive/2_simulate_adaptation.ipynb`` (the legacy stage-2 notebook), and
    * the Gate-1 regression test, which asserts that ``ThresholdRule`` reproduces
      this class's damage/adaptation formula bit-for-bit.

    New code should use ``SimulationEngine``.  See the numbered examples in
    ``examples_engine/`` (start with ``01_quickstart.py``).
"""
import numpy as np
import warnings

from floodadapt_abm._core.lookup_utils import (
    interpolate_damage_matrix as _lu_interpolate_damage_matrix,
)


class ABMSimulator:
    """Legacy threshold-rule ABM simulator (deprecated; see module docstring)."""

    def __init__(self, ds_impacts, times, slr_values, no_seq, damage_threshold=0.3, seed=42, dmg_unit="$", slr_unit="feet", damage_dtype=np.int32):
        warnings.warn(
            "ABMSimulator is deprecated and retained only for backward "
            "compatibility (legacy notebook + Gate-1 regression test). Use "
            "floodadapt_abm.SimulationEngine with a DecisionRule "
            "(ThresholdRule reproduces the 0.3-threshold behaviour; SEURule for "
            "the DYNAMO-M SEU science).",
            DeprecationWarning,
            stacklevel=2,
        )
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
        self.dmg_unit = dmg_unit
        self.slr_unit = slr_unit
        # Controls the dtype used for storing damage history (e.g., np.float32 or an integer dtype)
        self.damage_dtype = damage_dtype
        # Generate event sequences
        self.sequences = self.generate_event_sequences()

    # =====================
    # Public API
    # =====================
    def run_simulation(self, method='linear'):
        """
        Run the ABM simulation for all sequences and households using vectorized calculations.
        Returns:
            Sets attributes:
            - self.damage_history: [sequence, household, year] array of damages (float)
            - self.floodproofed: [sequence, household, year] boolean array of floodproofing state
        """
        self._compute_baseline_no_measures()
        damage_history, floodproofed = self._simulate_damage_history(floodproofing=True, method=method)
        self.damage_history = damage_history
        self.floodproofed = floodproofed
        self.has_run = True
        print("Evaluation completed.")

    # =====================
    # Event sequences & interpolation utilities
    # =====================
    def generate_event_sequences(self):
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
    
    def interpolate_damage_matrix(self, slr_values, event_names_list, strategy, method='linear'):
        """
        Vectorized lookup/interpolation of damages across SLR values and events for a given strategy.

        Parameters
        ----------
        slr_values : array-like
            1-D sequence of SLR targets to interpolate to.
        event_names_list : list of str
            Ordered list of event names to include in the output.
        strategy : str
            Strategy name applied to all objects.
        method : str
            Interpolation method: 'linear', 'nearest', 'cubic', 'floor', 'ceil'.

        Returns
        -------
        np.ndarray
            Shape (n_households, n_events, n_slr_values), dtype float32.
        """
        print(f"[LOOKUP] Interpolating damage matrix for strategy '{strategy}' using method '{method}'...")
        return _lu_interpolate_damage_matrix(
            ds=self.ds_impacts,
            strategy=strategy,
            slr_values=slr_values,
            event_names_list=event_names_list,
            method=method,
        )

    # Backwards-compatible alias for older notebooks/code
    def slr_damage_lookup(self, slr_values, event_names_list, strategy, method='linear'):
        warnings.warn(
            "ABMSimulator.slr_damage_lookup() is deprecated. Use interpolate_damage_matrix() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.interpolate_damage_matrix(slr_values, event_names_list, strategy, method)


    @staticmethod
    def _interpolate_damage_grid(slr_sim, damages_values, slr_values, method):
        """
        .. deprecated::
            Internal implementation detail delegated to
            :func:`floodadapt_abm.lookup_utils.interpolate_damage_at_slr`.
            Kept for backwards compatibility only; do not add new call sites.
        """
        import xarray as xr
        warnings.warn(
            "ABMSimulator._interpolate_damage_grid is deprecated. "
            "Use lookup_utils.interpolate_damage_matrix instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Build a minimal Dataset wrapping the raw array so we can reuse lookup_utils
        from scipy.interpolate import interp1d
        slr_values_arr = np.asarray(slr_values)
        if method == 'linear':
            f = interp1d(slr_sim, damages_values, kind='linear', axis=1,
                         bounds_error=False, fill_value='extrapolate')
            return f(slr_values_arr)
        elif method == 'nearest':
            indices = [int(np.abs(slr_sim - v).argmin()) for v in slr_values_arr]
            return np.stack([damages_values[:, idx] for idx in indices], axis=1)
        elif method == 'cubic':
            if len(slr_sim) < 4:
                raise ValueError('Cubic interpolation requires at least 4 SLR points.')
            f = interp1d(slr_sim, damages_values, kind='cubic', axis=1,
                         bounds_error=False, fill_value='extrapolate')
            return f(slr_values_arr)
        elif method == 'floor':
            sort_idx = np.argsort(slr_sim)
            slr_sorted = slr_sim[sort_idx]
            idxs = [np.searchsorted(slr_sorted, v, side='right') - 1 for v in slr_values_arr]
            idxs = [0 if i < 0 else i for i in idxs]
            return np.stack([damages_values[:, sort_idx[i]] for i in idxs], axis=1)
        elif method == 'ceil':
            sort_idx = np.argsort(slr_sim)
            slr_sorted = slr_sim[sort_idx]
            idxs = [np.searchsorted(slr_sorted, v, side='left') for v in slr_values_arr]
            idxs = [min(i, len(slr_sorted) - 1) for i in idxs]
            return np.stack([damages_values[:, sort_idx[i]] for i in idxs], axis=1)
        else:
            raise ValueError(f'Unknown interpolation method: {method}')


    # =====================
    # Plotting helpers
    # =====================

    def _compute_baseline_no_measures(self):
        """
        Compute and store baseline damages for all sequences,
        assuming 'no_measures' strategy for all households and all years (no floodproofing).
        Stores:
            self.baseline_damage_history: [sequence, household, year] array of damages (float)
        """
        baseline_damage_history, _ = self._simulate_damage_history(floodproofing=False, method='linear')
        self.baseline_damage_history = baseline_damage_history
           
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
        # 5 rows: colorbar, events, SLR, damages, floodproofed; 1 column
        gs = gridspec.GridSpec(5, 1, height_ratios=[0.5, 1, 1, 4, 1.5], hspace=0.15)
        cax = fig.add_subplot(gs[0])
        ax_events = fig.add_subplot(gs[1], sharex=None)
        ax_slr = fig.add_subplot(gs[2], sharex=ax_events)
        ax = fig.add_subplot(gs[3], sharex=ax_events)
        ax_floodproof = fig.add_subplot(gs[4], sharex=ax_events)

        # Plot event dots in the top axis (stacked vertically for same time step, no spacing)
        for t, events in enumerate(seq):
            n_events = len(events)
            if n_events == 0:
                continue
            for i, event in enumerate(events):
                color = event2color[event]
                # Stack events vertically at the same timestep with unit spacing to avoid overlap
                ax_events.scatter(t, i, color=color, s=60, marker='o', edgecolor='k', zorder=3)

        # Set y-ticks for event axis to show up to max number of events
        max_stack = max(len(events) for events in seq)
        ax_events.set_ylim(-0.5, max_stack - 0.5 if max_stack > 0 else 0.5)
        ax_events.set_ylabel('Events')
        # Shared x across subplots; use indices for time steps
        x = np.arange(len(times))
        ax_events.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
        # New subplot: Sea Level Rise over time (line with markers)
        # Ensure slr values align with time steps
        slr = np.asarray(self.slr_values)
        if slr.shape[0] != x.shape[0]:
            n = min(slr.shape[0], x.shape[0])
            slr = slr[:n]
            x_slr = x[:n]
        else:
            x_slr = x
        ax_slr.plot(x_slr, slr, '-o', color='tab:purple', markersize=4, linewidth=1.5, label='SLR')
        ax_slr.set_ylabel(f'Sea Level\nRise ({self.slr_unit})')
        ax_slr.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
        ax_slr.legend(loc='upper left')

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

        ax.set_ylabel(f'Total Damage ({self.dmg_unit})')
        # Share x with bottom axis; hide labels here
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
        ax.legend()

        # Add horizontal colorbar for event frequency above the event plot
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # Show a small, fixed number of ticks evenly spaced between min/max (log scale)
        min_tick = float(np.min(event_freqs))
        max_tick = float(np.max(event_freqs))
        # Ensure positive range for log-space
        min_tick = max(min_tick, 1e-12)
        max_tick = max(max_tick, min_tick * 1.000000001)
        n_ticks = 4  # desired number of ticks (3-4 as requested)
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
            ax_floodproof.set_xlim(-0.5, len(times) - 0.5)
            ax_floodproof.set_xticks(x)
            ax_floodproof.set_xticklabels(times, rotation=45)
            ax_floodproof.legend()
        else:
            ax_floodproof.text(0.5, 0.5, 'No floodproofing data', ha='center', va='center')
            ax_floodproof.set_xlim(-0.5, len(times) - 0.5)
            ax_floodproof.set_xticks(x)
            ax_floodproof.set_xticklabels(times, rotation=45)
            ax_floodproof.set_ylabel('Floodproofed\nBuildings\n(cumulative)')

        # Make colorbar height smaller
        # Set colorbar axis height to 0.15 of figure height (smaller than default)
        pos = cax.get_position()
        cax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.5])

        fig.tight_layout()
        plt.show()
        
    def plot_total_damage_statistics(self, percentiles=None, figsize=(12, 8)):
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
        
        # Check if simulation has been run
        if not hasattr(self, 'has_run') or not getattr(self, 'has_run', False):
            raise RuntimeError("Simulation has not been run. Please call the 'run' method before plotting.")
        
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

        # Compute avoided damage
        if hasattr(self, 'baseline_damage_history') and self.baseline_damage_history is not None:
            if hasattr(self, 'floodproofed') and self.floodproofed is not None:
                avoided = base_total - sim_total
                avoided = np.clip(avoided, 0, None)
            else:
                avoided = np.zeros_like(sim_total)
        else:
            avoided = np.zeros_like(sim_total)

        # Create a single figure with three subplots: SLR, damages, floodproofed
        fig, (ax_slr, ax, ax2) = plt.subplots(
            3, 1, figsize=figsize, sharex=True,
            gridspec_kw={'height_ratios': [1, 4, 1.5]}
        )

        # --- Top plot: Sea Level Rise over time ---
        x = np.arange(n_years)
        slr = np.asarray(self.slr_values)
        if slr.shape[0] != x.shape[0]:
            n = min(slr.shape[0], x.shape[0])
            slr = slr[:n]
            x_slr = x[:n]
        else:
            x_slr = x
        ax_slr.plot(x_slr, slr, '-o', color='tab:purple', markersize=4, linewidth=1.5, label='SLR')
        ax_slr.set_ylabel(f'Sea Level\nRise ({self.slr_unit})')
        ax_slr.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
        ax_slr.legend(loc='upper left')

        # --- Left plot: Damages ---
        if percentiles is None:
            sim_bar = get_bar_data(sim_total, None)
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
        
        ax.set_ylabel(f'Total Damage ({self.dmg_unit})')
        ax.set_title('Total Damages: Simulation vs Baseline')
        # Hide x labels here; bottom axis owns them
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
        ax.legend()

        # --- Right plot: Average number of floodproofed households per year ---
        if hasattr(self, 'floodproofed') and self.floodproofed is not None:
            avg_floodproofed = np.mean(self.floodproofed, axis=0)  # shape: [household, time]
            avg_floodproofed_per_year = avg_floodproofed.sum(axis=0)  # shape: [time]
            ax2.bar(np.arange(n_years), avg_floodproofed_per_year, width=width, color='tab:green', label='Avg. Floodproofed Households')
            ax2.set_ylabel('Avg. Number of Floodproofed Households')
            ax2.set_title('Average Number of Floodproofed Households per Year')
            ax2.set_xticks(np.arange(n_years))
            ax2.set_xticklabels(times, rotation=45)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No floodproofing data', ha='center', va='center')
            ax2.set_ylabel('Avg. Number of Floodproofed Households')
            ax2.set_title('Average Number of Floodproofed Households per Year')
            ax2.set_xticks(np.arange(n_years))
            ax2.set_xticklabels(times, rotation=45)

        # Bottom axis owns the x-axis label
        ax2.set_xlabel('Time')

        fig.tight_layout()
        plt.show()
    
    # =====================
    # Core simulation logic
    # =====================
    def _simulate_damage_history(self, floodproofing: bool, method: str = 'linear'):
        """
        Shared logic for calculating damage history.
        If floodproofing is True, applies floodproofing logic; otherwise, always uses 'no_measures'.
        Returns:
            damage_history: [sequence, household, time] array (float)
            floodproofed: [sequence, household, time] boolean array (None if floodproofing is False)
        """
        event_names_list = list(self.event_names)
        damage_history = np.zeros((self.no_seq, self.n_households, self.time_steps), dtype=self.damage_dtype)
        floodproofed = np.zeros((self.no_seq, self.n_households, self.time_steps), dtype=bool) if floodproofing else None

        # full matrix lookups for no measures and floodproofing all (n_objects, n_events, n_slr_values)
        damage_matrix_no_measures = self.interpolate_damage_matrix(self.slr_values, event_names_list, 'no_measures', method=method)
        if floodproofing:
            damage_matrix_floodproofing_all = self.interpolate_damage_matrix(self.slr_values, event_names_list, 'floodproof_all_0', method=method)

        for seq_idx in range(self.no_seq):
            is_last = (seq_idx == self.no_seq - 1)
            if floodproofing:
                if is_last:
                    print(f"[ADAPTATION] Evaluating sequence {seq_idx+1}/{self.no_seq}...")
                else:
                    print(f"[ADAPTATION] Evaluating sequence {seq_idx+1}/{self.no_seq}...", end='\r', flush=True)
            else:
                if is_last:
                    print(f"[BASELINE] Evaluating sequence {seq_idx+1}/{self.no_seq}...")
                else:
                    print(f"[BASELINE] Evaluating sequence {seq_idx+1}/{self.no_seq}...", end='\r', flush=True)
            is_floodproofed = np.zeros(self.n_households, dtype=bool)
            for ti in range(self.time_steps):
                year_events = self.sequences[seq_idx][ti]
                total_damage = np.zeros(self.n_households)
                for event in year_events:
                    if event in event_names_list:
                        event_idx = event_names_list.index(event)
                        damages = damage_matrix_no_measures[:, event_idx, ti]
                        if floodproofing: # apply floodproofing if applicable
                            damages_floodproofing_all = damage_matrix_floodproofing_all[:, event_idx, ti]
                            damages = np.where(is_floodproofed, damages_floodproofing_all, damages)
                        total_damage += damages
                # Cast to configured dtype; round if integer dtype to avoid silent truncation
                if np.issubdtype(self.damage_dtype, np.integer):
                    damage_history[seq_idx, :, ti] = np.rint(total_damage).astype(self.damage_dtype)
                else:
                    damage_history[seq_idx, :, ti] = total_damage.astype(self.damage_dtype)
                if floodproofing:
                    floodproofed[seq_idx, :, ti] = is_floodproofed
                    # Vectorized floodproofing decision
                    not_floodproofed = ~is_floodproofed
                    with_pot_dmg = self.max_pot_dmg > 0
                    threshold_exceeded = np.zeros(self.n_households, dtype=bool)
                    valid = not_floodproofed & with_pot_dmg
                    threshold_exceeded[valid] = (total_damage[valid] / self.max_pot_dmg[valid]) > self.damage_threshold
                    is_floodproofed = is_floodproofed | threshold_exceeded           
        return damage_history, floodproofed