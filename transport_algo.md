# Flattened Event-Based GPU Monte Carlo

## Overview

Instead of assigning one thread to simulate a neutron from birth to death, process many particles one event at a time using queues.

Each particle is always waiting for one of two things:
* movement to next boundary or collision
* collision processing

This avoids long uneven histories and improves GPU utilization.

## Core Queues

* `move_queue` — particles needing transport
* `collision_queue` — particles at collision sites
* `fission_bank` — children for next generation
* dead/escaped particles are removed

## Main Algorithm

```cpp
move_queue = source_particles
collision_queue = empty
fission_bank = empty

while queues not empty:
    transport_kernel(move_queue)
    collision_kernel(collision_queue)
    compact/sort queues

source_particles = normalize(fission_bank)
```

## Transport Kernel

For each particle:
1. Compute distance to nearest boundary
2. Compute distance to next collision
3. Move to whichever is closer

If boundary first:
* update cell/material
* return to `move_queue`

If collision first:
* send to `collision_queue`

## Collision Kernel

For each particle, sample reaction:
* scatter → update direction/energy, return to `move_queue`
* capture → kill particle
* fission → add children to `fission_bank`
