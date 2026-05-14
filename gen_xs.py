import openmc
import openmc.mgxs as mgxs
import numpy as np

# ==============================================================================
# 1. Define Materials
# ==============================================================================
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
water.set_density('g/cm3', 1.0)
water.add_s_alpha_beta('c_H_in_H2O')
water.add_element('B', 0.0017)

mats = openmc.Materials([uo2, aluminum, water])
mats.export_to_xml()

# ==============================================================================
# 2. Define Geometry (Simple 2D Pincell)
# ==============================================================================
# ==============================================================================
# 2. Define Geometry (Simple 2D Pincell)
# ==============================================================================
fuel_cyl = openmc.ZCylinder(r=0.53)
clad_cyl = openmc.ZCylinder(r=0.90)

# Reflective boundary box to simulate an infinite lattice
# box = openmc.model.RectangularParallelepiped(-0.9, 0.9, -0.9, 0.9, -1.0, 1.0, boundary_type='reflective')
box = openmc.model.RectangularParallelepiped(-1.2, 1.2, -1.2, 1.2, -1.0, 1.0, boundary_type='reflective')

# Apply the bounded box (-box) to ALL cells to cap them in the Z direction
fuel_cell = openmc.Cell(fill=uo2, region=-fuel_cyl & -box)
clad_cell = openmc.Cell(fill=aluminum, region=+fuel_cyl & -clad_cyl & -box)
mod_cell = openmc.Cell(fill=water, region=+clad_cyl & -box)

geom = openmc.Geometry([fuel_cell, clad_cell, mod_cell])
geom.export_to_xml()

# ==============================================================================
# 3. Simulation Settings
# ==============================================================================
settings = openmc.Settings()
settings.batches = 50
settings.inactive = 10
settings.particles = 5000 # Increase this for higher accuracy later
# bounds = [-0.9, -0.9, -1.0, 0.9, 0.9, 1.0]
bounds = [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
uniform_dist = openmc.stats.Box(bounds[:3], bounds[3:])
settings.source = openmc.IndependentSource(space=uniform_dist, constraints={'fissionable': True})
settings.export_to_xml()

# ==============================================================================
# 4. Multi-Group Cross Section (MGXS) Setup
# ==============================================================================
# OpenMC requires energy groups in eV and in ASCENDING order (Thermal to Fast).
# 3e-8 MeV = 3e-2 eV, 3e1 MeV = 3e7 eV
# energy_bounds = [3.0e-2, 3.0e-1, 3.0e0, 3.0e1, 3.0e2, 3.0e3, 3.0e4, 3.0e5, 3.0e6, 3.0e7]
# energy_bounds = [3.0e-2, 3.0e-1, 3.0e0, 3.0e1, 3.0e2, 3.0e3, 3.0e4, 3.0e5, 3.0e6, 2.0e7]
# 11 boundaries to create exactly 10 energy groups (Ascending order: Thermal -> Fast)
energy_bounds = [3.0e-3, 3.0e-2, 3.0e-1, 3.0e0, 3.0e1, 3.0e2, 3.0e3, 3.0e4, 3.0e5, 3.0e6, 2.0e7]
groups = mgxs.EnergyGroups(energy_bounds)

# Create the MGXS library to calculate Fission, Capture, and Scatter
mgxs_lib = mgxs.Library(geom)
mgxs_lib.energy_groups = groups
mgxs_lib.mgxs_types = ['fission', 'capture', 'scatter']
mgxs_lib.domain_type = 'material'
mgxs_lib.domains = [uo2, aluminum, water]
mgxs_lib.build_library()

tallies = openmc.Tallies()
mgxs_lib.add_to_tallies_file(tallies, merge=True)
tallies.export_to_xml()

# ==============================================================================
# 5. Run the Simulation
# ==============================================================================
openmc.run()

# ==============================================================================
# 6. Extract Data and Format for CUDA C++
# ==============================================================================
# Load the results from the simulation
sp = openmc.StatePoint('statepoint.50.h5')
mgxs_lib.load_from_statepoint(sp)

num_groups = 10
num_regions = 3

sigma_f = np.zeros((num_groups, num_regions))
sigma_c = np.zeros((num_groups, num_regions))
sigma_s = np.zeros((num_groups, num_regions))

# OpenMC outputs arrays from Thermal -> Fast. 
# Your C++ code expects Fast -> Thermal. [::-1] reverses the array to match.
for j, mat in enumerate([uo2, aluminum, water]):
    sigma_f[:, j] = mgxs_lib.get_mgxs(mat, 'fission').get_xs()[::-1]
    sigma_c[:, j] = mgxs_lib.get_mgxs(mat, 'capture').get_xs()[::-1]
    sigma_s[:, j] = mgxs_lib.get_mgxs(mat, 'scatter').get_xs()[::-1]

# --- Print the C++ Output ---
def print_cuda_array(name, data):
    print(f"// {name.replace('d_', '').replace('_', ' ')} by energy group and region")
    print(f"__constant__ float {name}[NUM_GROUPS][NUM_REGIONS] = {{")
    for i in range(num_groups):
        row_str = ", ".join([f"{val:.3e}f" for val in data[i]])
        ending = "," if i < num_groups - 1 else ""
        print(f"    {{{row_str}}}{ending}")
    print("};\n")

print("\n" + "="*60)
print("GENERATED CUDA C++ CROSS SECTIONS")
print("="*60 + "\n")

print("// Energy group lower bounds (MeV)")
print("__constant__ float d_GROUP_ENERGY[NUM_GROUPS] = {")
print("    3.0e+1f, 3.0e+0f, 3.0e-1f, 3.0e-2f, 3.0e-3f,")
print("    3.0e-4f, 3.0e-5f, 3.0e-6f, 3.0e-7f, 3.0e-8f\n};\n")

print_cuda_array("d_sigma_f", sigma_f)
print_cuda_array("d_sigma_c", sigma_c)
print_cuda_array("d_sigma_s", sigma_s)
