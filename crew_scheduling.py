"""
crew_scheduling.py
Solves a simplified Airline Crew Scheduling problem using backtracking (constraint satisfaction).
Includes:
 - Constraint checker (no overlap + minimum rest time)
 - Backtracking assignment generator (with optional cost minimization)
 - Profiling (time + tracemalloc) and recursive call counting
 - Simple Gantt chart visualization (matplotlib)
 - Small experiments to demonstrate exponential growth

Usage:
    python crew_scheduling.py
Output:
    - Prints a valid assignment (if found)
    - Saves gantt chart to /mnt/data/crew_gantt.png
    - Prints profiling results for varying flight counts
"""

import time
import tracemalloc
import itertools
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import sys

# -----------------------------
# Helper: Flight and Crew Data
# -----------------------------
# Flight tuple: (flight_id, start_time, end_time)
# Times are in integer hours (e.g., 9 means 09:00).

def sample_flights_small():
    # A small example (fits backtracking easily)
    flights = [
        ('F1', 9, 11),
        ('F2', 10, 12),
        ('F3', 13, 15),
        ('F4', 11, 13),
        ('F5', 15, 17)
    ]
    crew = ['C1', 'C2', 'C3']
    return flights, crew

def generate_random_flights(n, start_hour=6, end_hour=22):
    # Generate n flights with random durations and start times between start_hour and end_hour
    flights = []
    for i in range(n):
        dur = random.choice([1,2,3])
        s = random.randint(start_hour, end_hour - dur)
        flights.append((f'F{i+1}', s, s+dur))
    # Sort by start time for nicer output
    flights.sort(key=lambda x: x[1])
    return flights

# -----------------------------
# Constraint Checking
# -----------------------------
def flights_conflict(f1, f2, min_rest=1):
    """
    Returns True if flights f1 and f2 conflict for the same crew.
    f1, f2: tuples (id, start, end)
    min_rest: minimum rest hours required between end of one flight and start of next
    """
    _, s1, e1 = f1
    _, s2, e2 = f2
    # Overlap check
    # If one ends plus min_rest is less than or equal to the other's start, they do NOT conflict.
    if e1 + min_rest <= s2 or e2 + min_rest <= s1:
        return False
    # Otherwise they conflict (overlap or insufficient rest)
    return True

def is_valid_assignment_for_crew(assigned_flights, candidate_flight, min_rest=1):
    """
    Check if candidate_flight can be added to assigned_flights for a single crew
    without violating overlap or rest constraints.
    """
    for f in assigned_flights:
        if flights_conflict(f, candidate_flight, min_rest=min_rest):
            return False
    return True

# -----------------------------
# Backtracking Assignment
# -----------------------------
class Scheduler:
    def __init__(self, flights, crew, min_rest=1, minimize_cost=False, cost_map=None):
        self.flights = flights[:]  # list of (id,start,end)
        self.crew = crew[:]        # list of crew ids
        self.min_rest = min_rest
        self.minimize_cost = minimize_cost
        self.cost_map = cost_map or {}  # dict {(flight_id, crew_id): cost}
        self.best_assignment = None
        self.best_cost = float('inf')
        self.recursive_calls = 0

    def backtrack(self, idx, current_assign):
        """
        idx: index of flight to assign
        current_assign: dict crew_id -> list of flights assigned
        """
        self.recursive_calls += 1

        if idx >= len(self.flights):
            # all flights assigned
            if self.minimize_cost:
                total = self.compute_cost(current_assign)
                if total < self.best_cost:
                    self.best_cost = total
                    self.best_assignment = {k: v[:] for k, v in current_assign.items()}
            else:
                # first valid assignment found
                if self.best_assignment is None:
                    self.best_assignment = {k: v[:] for k, v in current_assign.items()}
            return True

        flight = self.flights[idx]
        assigned_any = False

        # Try to assign this flight to each crew member
        for c in self.crew:
            if is_valid_assignment_for_crew(current_assign[c], flight, self.min_rest):
                current_assign[c].append(flight)
                self.backtrack(idx+1, current_assign)
                current_assign[c].pop()
                assigned_any = True
        # Optionally allow leaving a flight unassigned -> but problem requires all flights assigned
        return assigned_any

    def compute_cost(self, assignment):
        total = 0
        for c, flights in assignment.items():
            for f in flights:
                key = (f[0], c)
                total += self.cost_map.get(key, 0)
        return total

    def solve(self, timeout=None):
        # initialize current_assign
        current_assign = {c: [] for c in self.crew}
        tracemalloc.start()
        t0 = time.perf_counter()
        self.backtrack(0, current_assign)
        t1 = time.perf_counter()
        mem_current, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            'assignment': self.best_assignment,
            'best_cost': self.best_cost if self.minimize_cost else None,
            'time_s': t1 - t0,
            'mem_peak_bytes': mem_peak,
            'recursive_calls': self.recursive_calls
        }

# -----------------------------
# Visualization: Gantt Chart
# -----------------------------
def plot_gantt(assignment, out_path='/mnt/data/crew_gantt.png'):
    """
    assignment: dict crew -> list of (id,start,end)
    """
    # If assignment is None or empty, create an empty plot message
    if not assignment:
        print('No assignment to plot.')
        return None
    fig, ax = plt.subplots(figsize=(10, 2 + 0.6*len(assignment)))
    yticks = []
    ylabels = []
    crew_sorted = sorted(assignment.keys())
    for i, c in enumerate(crew_sorted):
        tasks = sorted(assignment[c], key=lambda x: x[1])
        for t in tasks:
            fid, s, e = t
            ax.barh(i, e - s, left=s, height=0.4, align='center')
            ax.text(s + 0.1, i - 0.08, fid, va='center', ha='left', fontsize=8, color='white' if e-s>1 else 'black', clip_on=True)
        yticks.append(i)
        ylabels.append(c)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel('Hour (24h)')
    ax.set_title('Crew Schedule Gantt Chart')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    return out_path

# -----------------------------
# Simple Experiment: time vs number of flights
# -----------------------------
def run_scaling_experiment(crew_count=3, flight_sizes=[4,6,8], trials=3):
    results = []
    for n in flight_sizes:
        times = []
        calls = []
        for t in range(trials):
            flights = generate_random_flights(n)
            crew = [f'C{i+1}' for i in range(crew_count)]
            sched = Scheduler(flights, crew, min_rest=1)
            res = sched.solve()
            times.append(res['time_s'])
            calls.append(res['recursive_calls'])
        avg_time = sum(times)/len(times)
        avg_calls = sum(calls)/len(calls)
        results.append({'n': n, 'avg_time_s': avg_time, 'avg_recursive_calls': avg_calls})
        print(f'Flights={n}, avg_time={avg_time:.4f}s, avg_calls={avg_calls:.1f}')
    # Plot
    xs = [r['n'] for r in results]
    ys = [r['avg_time_s'] for r in results]
    plt.figure(figsize=(7,4))
    plt.plot(xs, ys, marker='o')
    plt.xlabel('Number of Flights')
    plt.ylabel('Average Time (s)')
    plt.title('Backtracking: Time vs Number of Flights (avg over trials)')
    plt.grid(True)
    out = '/mnt/data/scaling_time.png'
    plt.savefig(out)
    plt.close()
    return results, out

# -----------------------------
# Main demonstration
# -----------------------------
def main():
    print('Crew Scheduling Backtracking Demo\n')
    flights, crew = sample_flights_small()
    print('Flights:')
    for f in flights:
        print(f)
    print('Crew:', crew)
    print('\nSolving with backtracking (min_rest=1 hour)...\n')

    scheduler = Scheduler(flights, crew, min_rest=1)
    result = scheduler.solve()
    assignment = result['assignment']

    if assignment is None:
        print('No valid assignment found for the given flights and crew.\n')
    else:
        print('Found assignment:')
        for c in sorted(assignment.keys()):
            print(f'  {c}: {[f[0] for f in assignment[c]]}')
        print(f\"Time taken: {result['time_s']:.6f}s | Peak mem: {result['mem_peak_bytes']} bytes | Recursive calls: {result['recursive_calls']}\")
        # Save assignment to file
        with open('/mnt/data/crew_assignment.json', 'w') as fh:
            json.dump({c: [(f[0], f[1], f[2]) for f in assignment[c]] for c in assignment}, fh, indent=2)
        print('\\nSaved assignment to /mnt/data/crew_assignment.json')

        # Plot Gantt chart
        gantt_path = plot_gantt(assignment, out_path='/mnt/data/crew_gantt.png')
        if gantt_path:
            print(f'Gantt chart saved to {gantt_path}')

    # Run scaling experiment (small sizes to avoid long runtime)
    print('\\nRunning small scaling experiment (this may take some time)...')
    random.seed(1)
    res, plot_path = run_scaling_experiment(crew_count=3, flight_sizes=[4,6,8], trials=3)
    print(f'Scaling plot saved to {plot_path}')

if __name__ == '__main__':
    main()
