"""
=============================================================================
Assignment: Evaluating Network Switch Scheduling Algorithms
=============================================================================
Simulates a 3x3 crossbar network switch using three different scheduling
strategies:
  Part 1 - Standard FIFO Queue (demonstrates Head-of-Line blocking)
  Part 2 - Virtual Output Queue (VOQ) with Exhaustive Optimal Search
  Part 3 - VOQ with iSLIP Scheduling Algorithm
  Part 4 - Data Visualization comparing all three methods

HOW TO RUN:
  1. Install Python 3 (https://www.python.org/downloads/)
  2. Install matplotlib:
       pip install matplotlib
  3. Run:
       python switch_simulation.py
  4. Output: terminal logs + "switch_comparison.png" graph file

DEPENDENCIES:
  - Python 3.7+
  - matplotlib  (pip install matplotlib)
=============================================================================
"""

from collections import deque
import matplotlib.pyplot as plt


# =============================================================================
# THE INPUT TRACE
# =============================================================================
# Each tuple = (Packet_ID, Arrival_Time, Source_Input_Port, Dest_Output_Port)
# All packets arrive at the BEGINNING of the specified time slot.
# Switch has 3 input ports (I0, I1, I2) and 3 output ports (O0, O1, O2).
# =============================================================================
PACKETS = [
    ("p1",  0, 0, 0),
    ("p2",  0, 0, 1),
    ("p3",  0, 1, 0),
    ("p4",  0, 1, 2),
    ("p5",  0, 2, 0),
    ("p6",  1, 0, 2),
    ("p7",  1, 2, 1),
    ("p8",  2, 1, 1),
    ("p9",  2, 2, 2),
    ("p10", 3, 0, 1),
    ("p11", 3, 1, 0),
    ("p12", 3, 2, 1),
    ("p13", 4, 0, 0),
    ("p14", 4, 1, 2),
    ("p15", 4, 2, 2),
    ("p16", 5, 0, 2),
    ("p17", 5, 1, 1),
    ("p18", 5, 2, 0),
]

TOTAL_PACKETS = len(PACKETS)


def arrivals_at(t):
    """Return (pid, src, dst) for every packet arriving exactly at time t."""
    return [(pid, src, dst) for pid, arr, src, dst in PACKETS if arr == t]


# =============================================================================
# PART 1 — STANDARD FIFO QUEUE WITH HEAD-OF-LINE (HoL) BLOCKING
# =============================================================================
def run_fifo():
    print("=" * 65)
    print("PART 1: Standard FIFO Queue with Head-of-Line (HoL) Blocking")
    print("=" * 65)

    # One shared FIFO queue per input port.
    # Each entry: (packet_id, destination_output_port)
    queues = [deque() for _ in range(3)]

    sent_total = 0
    t = 0
    backlog_over_time = {}

    while sent_total < TOTAL_PACKETS:

        # -- ARRIVALS: append new packets to the back of each input queue --
        for pid, src, dst in arrivals_at(t):
            queues[src].append((pid, dst))

        in_switch = sum(len(q) for q in queues)

        if in_switch == 0:
            t += 1
            continue

        # -- SCHEDULING (FIFO Rule) ----------------------------------------
        # Only the HEAD of each queue is eligible for transmission.
        # If two inputs want the same output, the lower-numbered input wins.
        winner_inputs = {}   # output_port -> (input_port, packet_id)
        for inp in range(3):
            if not queues[inp]:
                continue
            pid, dst = queues[inp][0]   # peek at head (do not remove yet)
            if dst not in winner_inputs:
                winner_inputs[dst] = (inp, pid)
            else:
                existing_inp, _ = winner_inputs[dst]
                if inp < existing_inp:
                    winner_inputs[dst] = (inp, pid)   # lower port number wins

        # -- TRANSMIT winning packets --------------------------------------
        used_inputs  = set()
        used_outputs = set(winner_inputs.keys())
        sent_this_slot = []
        for dst, (inp, pid) in winner_inputs.items():
            queues[inp].popleft()
            sent_this_slot.append((pid, inp, dst))
            used_inputs.add(inp)
            sent_total += 1

        # -- DETECT HoL BLOCKING ------------------------------------------
        # True HoL blocking: an input's HEAD packet wants a BUSY output,
        # while a packet BEHIND the head wants an IDLE output that could
        # have been served — but can't because the head is in the way.
        hol_blocked = []
        for inp in range(3):
            if inp in used_inputs or not queues[inp]:
                continue
            head_pid, head_dst = queues[inp][0]
            # Head must be competing for a busy output (it lost contention or
            # the output was taken by another input this slot).
            if head_dst not in used_outputs:
                continue   # head's output is idle but it wasn't sent — shouldn't happen
            # Check if any packet BEHIND the head wants an idle output.
            for behind_pid, behind_dst in list(queues[inp])[1:]:
                if behind_dst not in used_outputs:
                    hol_blocked.append((behind_pid, inp, behind_dst, head_pid, head_dst))
                    break   # one report per input is enough

        # Record backlog AFTER sending (packets remaining at end of slot)
        backlog_over_time[t] = sum(len(q) for q in queues)

        sent_str = (", ".join(f"{pid}(I{i}->O{d})" for pid, i, d in sent_this_slot)
                    if sent_this_slot else "none")
        print(f"  t={t}: Sent: {sent_str}")
        if hol_blocked:
            hb_str = ", ".join(
                f"I{i}: head {hpid}->O{hdst} (busy) blocks {bpid}->O{bdst} (idle)"
                for bpid, i, bdst, hpid, hdst in hol_blocked)
            print(f"         *** HoL BLOCKING: {hb_str} ***")

        t += 1

    backlog_over_time[t] = 0
    print(f"\n  >>> Total Service Time (FIFO): {t} time slots")
    return t, backlog_over_time


# =============================================================================
# PART 2 — VIRTUAL OUTPUT QUEUING (VOQ) — EXHAUSTIVE OPTIMAL SEARCH
# =============================================================================

# Memoization cache for the DFS lookahead.
_voq_memo = {}


def _all_valid_matchings(voq_counts):
    """
    Generate ALL valid bipartite matchings given a 9-tuple of VOQ counts.
    A matching is a list of (input, output) pairs where every pair has
    packets available and no input or output appears more than once.
    Returns only non-empty matchings.
    """
    available = [(i, j) for i in range(3) for j in range(3)
                 if voq_counts[i * 3 + j] > 0]
    matchings = [[]]
    for pair in available:
        new_matchings = []
        for m in matchings:
            new_matchings.append(m)
            used_in  = {p[0] for p in m}
            used_out = {p[1] for p in m}
            if pair[0] not in used_in and pair[1] not in used_out:
                new_matchings.append(m + [pair])
        matchings = new_matchings
    return [m for m in matchings if m]   # drop the empty matching


def _min_finish_time(voq_counts, t):
    """
    TRUE exhaustive search via memoized DFS.

    Given the current VOQ state (as a 9-tuple of queue lengths) BEFORE
    arrivals at time t, return the earliest time slot at which the switch
    can be completely emptied under the best possible sequence of matchings.

    Using counts (not packet IDs) makes the state space small enough for
    memoization — two states with identical counts are equivalent regardless
    of which specific packets are waiting.
    """
    future_arrivals = sum(1 for _, arr, _, _ in PACKETS if arr >= t)
    if sum(voq_counts) == 0 and future_arrivals == 0:
        return t   # switch is empty and nothing more will arrive

    key = (t, voq_counts)
    if key in _voq_memo:
        return _voq_memo[key]

    # Apply arrivals at slot t
    vc = list(voq_counts)
    for _, arr, src, dst in PACKETS:
        if arr == t:
            vc[src * 3 + dst] += 1
    vc = tuple(vc)

    if sum(vc) == 0:
        # Nothing to send this slot; advance time
        result = _min_finish_time(vc, t + 1)
        _voq_memo[key] = result
        return result

    # Try every valid matching and recurse; keep the best (minimum) finish time
    best = float('inf')
    for m in _all_valid_matchings(vc):
        nv = list(vc)
        for (i, j) in m:
            nv[i * 3 + j] -= 1
        result = _min_finish_time(tuple(nv), t + 1)
        if result < best:
            best = result

    _voq_memo[key] = best
    return best


def run_optimal_voq():
    print("\n" + "=" * 65)
    print("PART 2: VOQ with Exhaustive Optimal Search")
    print("=" * 65)

    # voqs[i][j] = deque of actual packet IDs (for printing)
    voqs = [[deque() for _ in range(3)] for _ in range(3)]

    t = 0
    sent_total = 0
    backlog_over_time = {}

    while sent_total < TOTAL_PACKETS:

        # -- ARRIVALS: each packet goes into the correct VOQ ---------------
        for pid, arr, src, dst in PACKETS:
            if arr == t:
                voqs[src][dst].append(pid)

        in_switch = sum(len(voqs[i][j]) for i in range(3) for j in range(3))

        if in_switch == 0:
            t += 1
            continue

        # -- EXHAUSTIVE SEARCH with LOOKAHEAD ------------------------------
        # Convert current VOQ to counts for DFS state.
        counts = tuple(len(voqs[i][j]) for i in range(3) for j in range(3))

        # Evaluate every valid matching by simulating the remainder of the
        # schedule exhaustively (via memoized DFS) and pick the one that
        # leads to the globally minimum total service time.
        best_finish = float('inf')
        best_matching = None
        for m in _all_valid_matchings(counts):
            nc = list(counts)
            for (i, j) in m:
                nc[i * 3 + j] -= 1
            finish = _min_finish_time(tuple(nc), t + 1)
            if finish < best_finish:
                best_finish = finish
                best_matching = m

        # -- TRANSMIT the globally optimal matching ------------------------
        sent_this_slot = []
        for (inp, out) in best_matching:
            pid = voqs[inp][out].popleft()
            sent_this_slot.append((pid, inp, out))
            sent_total += 1

        match_str = [f"I{i}->O{o}" for i, o in best_matching]
        sent_str  = (", ".join(f"{pid}(I{i}->O{o})" for pid, i, o in sent_this_slot)
                     if sent_this_slot else "none")
        print(f"  t={t}: Matching {match_str}")
        print(f"         Sent: {sent_str}")

        # Record backlog AFTER sending
        backlog_over_time[t] = sum(len(voqs[i][j]) for i in range(3) for j in range(3))

        t += 1

    backlog_over_time[t] = 0
    print(f"\n  >>> Total Service Time (Optimal VOQ): {t} time slots")
    return t, backlog_over_time


# =============================================================================
# PART 3 — VOQ WITH iSLIP SCHEDULING ALGORITHM
# =============================================================================
def run_islip():
    print("\n" + "=" * 65)
    print("PART 3: VOQ with iSLIP Scheduling Algorithm")
    print("=" * 65)

    voqs = [[deque() for _ in range(3)] for _ in range(3)]

    # Round-robin arbiters — all start at 0 as per the assignment.
    # grant_ptr[j]  = next input output j will consider first when granting
    # accept_ptr[i] = next output input i will consider first when accepting
    grant_ptr  = [0, 0, 0]
    accept_ptr = [0, 0, 0]

    t = 0
    sent_total = 0
    backlog_over_time = {}

    while sent_total < TOTAL_PACKETS:

        # -- ARRIVALS -------------------------------------------------------
        for pid, arr, src, dst in PACKETS:
            if arr == t:
                voqs[src][dst].append(pid)

        in_switch = sum(len(voqs[i][j]) for i in range(3) for j in range(3))

        if in_switch == 0:
            t += 1
            continue

        print(f"\n  -- t={t} --")

        # iSLIP runs up to N iterations of Request->Grant->Accept per slot
        # (N=3 for a 3x3 switch). Already-matched inputs/outputs are excluded
        # from subsequent iterations within the same slot.
        # Pointers only advance for ACCEPTED matches (crucial rule).
        scheduled_inputs  = set()   # inputs  matched this slot
        scheduled_outputs = set()   # outputs matched this slot
        slot_accepted     = {}      # input -> output, final matches for this slot

        for iteration in range(3):  # up to N=3 iterations per slot

            # -- PHASE 1: REQUEST ------------------------------------------
            # Unscheduled inputs request every output they have packets for
            # (excluding already-scheduled outputs this slot).
            requests = {j: [] for j in range(3)}
            for i in range(3):
                if i in scheduled_inputs:
                    continue
                for j in range(3):
                    if voqs[i][j] and j not in scheduled_outputs:
                        requests[j].append(i)

            active_req = {f'O{j}': [f'I{i}' for i in v]
                          for j, v in requests.items() if v}
            if not active_req:
                break   # nothing left to match

            # -- PHASE 2: GRANT --------------------------------------------
            # Each unscheduled output grants ONE requesting input via RR.
            grant_given     = {}   # output -> input granted
            grants_received = {}   # input  -> list of outputs that granted it

            for j in range(3):
                if j in scheduled_outputs or not requests[j]:
                    continue
                for offset in range(3):
                    candidate = (grant_ptr[j] + offset) % 3
                    if candidate in requests[j]:
                        grant_given[j] = candidate
                        grants_received.setdefault(candidate, []).append(j)
                        break

            if not grant_given:
                break

            # -- PHASE 3: ACCEPT -------------------------------------------
            # Each granted input accepts ONE grant via its own RR pointer.
            accepted = {}   # input -> output accepted

            for i in range(3):
                if i not in grants_received:
                    continue
                for offset in range(3):
                    candidate = (accept_ptr[i] + offset) % 3
                    if candidate in grants_received[i]:
                        accepted[i] = candidate
                        break

            if not accepted:
                break

            print(f"  [Iter {iteration + 1}]")
            print(f"    REQUEST : {active_req}")
            print(f"    GRANT   : { {f'O{j}': f'I{i}' for j, i in grant_given.items()} }")
            print(f"    ACCEPT  : { {f'I{i}': f'O{j}' for i, j in accepted.items()} }")

            # Update pointers & mark scheduled for this slot
            for i, j in accepted.items():
                slot_accepted[i] = j
                scheduled_inputs.add(i)
                scheduled_outputs.add(j)
                grant_ptr[j]  = (i + 1) % 3   # advances past accepted input
                accept_ptr[i] = (j + 1) % 3   # advances past accepted output

        # -- TRANSMIT all matches found across all iterations --------------
        sent_this_slot = []
        for i, j in slot_accepted.items():
            pid = voqs[i][j].popleft()
            sent_this_slot.append((pid, i, j))
            sent_total += 1

        sent_str = (", ".join(f"{pid}(I{i}->O{j})" for pid, i, j in sent_this_slot)
                    if sent_this_slot else "none")
        print(f"  SENT    : {sent_str}")
        print(f"  Ptrs    : grant={grant_ptr[:]}  accept={accept_ptr[:]}")

        # Record backlog AFTER sending
        backlog_over_time[t] = sum(len(voqs[i][j]) for i in range(3) for j in range(3))

        t += 1

    backlog_over_time[t] = 0
    print(f"\n  >>> Total Service Time (iSLIP): {t} time slots")
    return t, backlog_over_time


# =============================================================================
# PART 4 — DATA VISUALIZATION
# =============================================================================
def plot_results(fifo_t, opt_t, islip_t, fifo_bl, opt_bl, islip_bl):
    print("\n" + "=" * 65)
    print("PART 4: Generating Comparison Graphs -> switch_comparison.png")
    print("=" * 65)

    plt.style.use('default')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        "3x3 Network Switch Scheduling Algorithm Comparison  (18 Packets)",
        fontsize=13, fontweight='bold', color='black'
    )

    FIFO_COLOR  = "#1f77b4"   # blue
    OPT_COLOR   = "#ff7f0e"   # orange
    ISLIP_COLOR = "#2ca02c"   # green

    # ── Graph 1: Bar Chart — Total Service Time ──
    ax1 = axes[0]
    ax1.set_facecolor('white')
    labels = ["FIFO", "Optimal VOQ", "iSLIP"]
    values = [fifo_t, opt_t, islip_t]
    colors = [FIFO_COLOR, OPT_COLOR, ISLIP_COLOR]
    bars = ax1.bar(labels, values, color=colors, edgecolor='black', linewidth=0.8, width=0.45)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{val} slots", ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax1.set_title("Total Service Time", fontsize=12, fontweight='bold', color='black')
    ax1.set_ylabel("Time Slots to Empty the Switch", color='black', fontsize=10)
    ax1.set_ylim(0, max(values) + 3)
    ax1.tick_params(colors='black')
    ax1.grid(axis='y', alpha=0.35, linestyle='--', color='gray')
    for spine in ax1.spines.values():
        spine.set_color('black')

    # ── Graph 2: Line Graph — Backlog Over Time ──
    ax2 = axes[1]
    ax2.set_facecolor('white')
    max_t = max(fifo_t, opt_t, islip_t)

    def series(bl, end_t):
        xs = list(range(end_t + 1))
        ys = [bl.get(tt, 0) for tt in xs]
        return xs, ys

    fx, fy = series(fifo_bl,  fifo_t)
    ox, oy = series(opt_bl,   opt_t)
    ix, iy = series(islip_bl, islip_t)

    ax2.plot(fx, fy, color=FIFO_COLOR,  label="FIFO",        linewidth=2.0)
    ax2.plot(ox, oy, color=OPT_COLOR,   label="Optimal VOQ", linewidth=2.0)
    ax2.plot(ix, iy, color=ISLIP_COLOR, label="iSLIP",       linewidth=2.0)

    ax2.set_title("Backlog Over Time", fontsize=12, fontweight='bold', color='black')
    ax2.set_xlabel("Time Slots (t)", color='black', fontsize=10)
    ax2.set_ylabel("Total Number of Packets\nRemaining in the Switch", color='black', fontsize=10)
    ax2.legend(fontsize=10, facecolor='white', edgecolor='gray', loc='upper right')
    ax2.set_xticks(range(0, max_t + 1, 2))
    ax2.set_xlim(0, max_t)
    ax2.set_ylim(0, 6)
    ax2.set_yticks(range(0, 7))
    ax2.tick_params(colors='black')
    ax2.grid(alpha=0.35, linestyle='--', color='gray')
    for spine in ax2.spines.values():
        spine.set_color('black')

    plt.tight_layout()
    plt.savefig("switch_comparison.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: switch_comparison.png")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  Network Switch Scheduling  —  3x3 Crossbar Switch")
    print("=" * 65)

    fifo_t,  fifo_bl  = run_fifo()
    opt_t,   opt_bl   = run_optimal_voq()
    islip_t, islip_bl = run_islip()

    plot_results(fifo_t, opt_t, islip_t, fifo_bl, opt_bl, islip_bl)

    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)
    print(f"  Algorithm        | Total Service Time")
    print(f"  -----------------+---------------------------")
    print(f"  FIFO             | {fifo_t:>2} slots  (HoL blocking)")
    print(f"  Optimal VOQ      | {opt_t:>2} slots  (theoretical best)")
    print(f"  iSLIP VOQ        | {islip_t:>2} slots  (practical algorithm)")
    print(f"\n  VOQ vs FIFO improvement : {fifo_t - opt_t} slots saved "
          f"({100*(fifo_t-opt_t)/fifo_t:.0f}% faster)")
    print(f"  iSLIP vs Optimal gap    : {islip_t - opt_t} extra slots")
    print("=" * 65)