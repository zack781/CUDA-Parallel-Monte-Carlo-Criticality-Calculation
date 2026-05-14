import openmc
import numpy as np
import os

def evaluate_k(boron_fraction):
    # 1. Define Materials
    uo2 = openmc.Material(name='3% Enriched UO2')
    uo2.add_element('U', 1.0, enrichment=3.0)
    uo2.add_element('O', 2.0)
    uo2.set_density('g/cm3', 10.5)

    aluminum = openmc.Material(name='Cladding')
    aluminum.add_element('Al', 1.0)
    aluminum.set_density('g/cm3', 2.7)

    water = openmc.Material(name='Moderator')
    water.add_element('H', 2.0)
    water.add_element('O', 1.0)

    # ---> INJECT VARIABLE BORON HERE <---
    if boron_fraction > 0:
        water.add_element('B', boron_fraction)

    water.set_density('g/cm3', 1.0)
    water.add_s_alpha_beta('c_H_in_H2O')

    mats = openmc.Materials([uo2, aluminum, water])
    mats.export_to_xml()

    # 2. Define Watertight Geometry
    fuel_cyl = openmc.ZCylinder(r=0.53)
    clad_cyl = openmc.ZCylinder(r=0.90)

    # Using the expanded 1.2 bounds to prevent surface coincidence leaks
    box = openmc.model.RectangularParallelepiped(-1.2, 1.2, -1.2, 1.2, -1.0, 1.0, boundary_type='reflective')

    fuel_cell = openmc.Cell(fill=uo2, region=-fuel_cyl & -box)
    clad_cell = openmc.Cell(fill=aluminum, region=+fuel_cyl & -clad_cyl & -box)
    mod_cell = openmc.Cell(fill=water, region=+clad_cyl & -box)

    geom = openmc.Geometry([fuel_cell, clad_cell, mod_cell])
    geom.export_to_xml()

    # 3. Fast Search Settings
    settings = openmc.Settings()
    settings.batches = 20     # Lower batches for faster searching
    settings.inactive = 5
    settings.particles = 1000 # Fewer particles to speed up the loop

    # Suppress output files to keep your directory clean
    settings.output = {'summary': False, 'tallies': False}

    # Spawn source well inside the fuel pin to prevent boundary leakage
    bounds = [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
    uniform_dist = openmc.stats.Box(bounds[:3], bounds[3:])
    settings.source = openmc.IndependentSource(space=uniform_dist, constraints={'fissionable': True})
    settings.export_to_xml()

    # 4. Run OpenMC silently
    openmc.run(output=False)

    # 5. Extract and return k-effective
    statepoint_file = f'statepoint.{settings.batches}.h5'
    with openmc.StatePoint(statepoint_file) as sp:
        k_eff = sp.k_combined.nominal_value

    # Clean up the statepoint file so it doesn't clutter the directory
    if os.path.exists(statepoint_file):
        os.remove(statepoint_file)

    return k_eff


# ==============================================================================
# The Bisection Search Algorithm
# ==============================================================================
print("\n" + "="*60)
print("STARTING CRITICAL BORON CONCENTRATION SEARCH")
print("="*60)

# Define our search bounds (Atomic fraction of Boron relative to Oxygen)
b_low = 0.0000   # We know this yields k ~ 1.33
b_high = 0.0200  # We know this yields k ~ 0.94 (Subcritical)

target_k = 1.000
tolerance = 0.005 # Accept anything between 0.995 and 1.005

for iteration in range(1, 15):
    # Guess the exact middle of our bounds
    b_mid = (b_low + b_high) / 2.0

    print(f"Iteration {iteration}: Testing Boron Fraction = {b_mid:.5f}")
    k_eff = evaluate_k(b_mid)
    print(f"  -> Resulting k-effective: {k_eff:.4f}")

    # Check if we hit the target
    if abs(k_eff - target_k) <= tolerance:
        print("\n" + "*"*60)
        print(f"SUCCESS! Target k=1.0 reached at Boron Fraction: {b_mid:.5f}")
        print("*"*60 + "\n")
        break

    # Adjust the bounds for the next loop
    if k_eff > target_k:
        # k is too high! The reactor is supercritical. We need MORE boron.
        b_low = b_mid
    else:
        # k is too low! The reactor is subcritical. We need LESS boron.
        b_high = b_mid
