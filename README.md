# Network Switch Scheduling Algorithms
### 3×3 Crossbar Switch Simulation

---

## What This Does

This project simulates a 3×3 crossbar network switch and compares three scheduling algorithms. The switch has 3 inputs and 3 outputs, and we test how many time slots each algorithm needs to send all 18 packets.

| Part | Algorithm | Time Taken |
|------|-----------|------------|
| Part 1 | FIFO with HoL Blocking | 11 slots |
| Part 2 | VOQ with Optimal Search | 7 slots (best possible) |
| Part 3 | VOQ with iSLIP | 8 slots (close to best) |
| Part 4 | Comparison Graph | switch_comparison.png |

---

## How to Run

Install the dependency first:
```bash
pip install matplotlib
```

Then run:
```bash
python3 switch_simulation_final.py
```

This will print a slot-by-slot log for all three algorithms and save `switch_comparison.png`.

---

## Files

```
switch_simulation_final.py   # main simulation code
switch_comparison.png        # output graph
README.md                    # this file
```

---

## Algorithm Summary

### Part 1 — FIFO with HoL Blocking
Each input has one shared queue. Only the front packet can be sent each slot. If it's stuck waiting for a busy output, everything behind it is also stuck — this is called Head-of-Line (HoL) blocking.

- 4 blocking events happened in this run
- Finished in **11 slots**

### Part 2 — VOQ with Optimal Search
Each input gets 3 separate queues (one per output). No HoL blocking. The scheduler tries every possible combination of matchings and picks the best one using DFS with memoization.

- Achieves the theoretical minimum
- Finished in **7 slots**
- Too slow for large switches though

### Part 3 — VOQ with iSLIP
Also uses VOQ (no HoL blocking). iSLIP runs up to 3 rounds per slot, each with three steps:
1. **Request** — inputs ask for outputs they have packets for
2. **Grant** — each output picks one input using round-robin
3. **Accept** — each input accepts one grant using round-robin

Round-robin pointers carry over between slots for fairness.

- Finished in **8 slots** (just 1 slot above optimal)
- 27% faster than FIFO
- Simple enough for real hardware

---

## Results

```
FIFO        → 11 slots  (HoL blocking hurts)
Optimal VOQ →  7 slots  (theoretical best)
iSLIP VOQ   →  8 slots  (practical and near-optimal)
```

