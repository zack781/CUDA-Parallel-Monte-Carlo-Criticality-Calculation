###########################################################
#          2D MONTE CARLO NEUTRON TRANSPORT CODE          #
###########################################################
import random
import math
import pylab
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell
from scipy.interpolate import interp1d

np.random.seed()

def CrossSections(Energy, region):
    #=========================================================#
    #         Energy Dependent Macroscopic Cross Section Data  #
    #=========================================================#
    # Unit is 1/cm
    # sigma_x[EnergyGroup][Region]
    sigma_f = np.array([
        [1.05e-1, 0, 0],
        [5.96e-2, 0, 0],
        [6.02e-2, 0, 0],
        [1.06e-1, 0, 0],
        [2.46e-1, 0, 0],
        [2.50e-1, 0, 0],
        [1.07e-1, 0, 0],
        [1.28e+0, 0, 0],
        [9.30e+0, 0, 0],
        [2.58e+1, 0, 0]
    ])

    sigma_c = np.array([
        [1.41e-6, 1.71e-2, 3.34e-6],
        [1.34e-3, 7.83e-3, 3.34e-6],
        [1.10e-2, 2.83e-4, 2.56e-7],
        [3.29e-2, 4.52e-6, 6.63e-7],
        [8.23e-2, 1.06e-5, 2.24e-7],
        [4.28e-2, 4.39e-6, 1.27e-7],
        [9.90e-2, 1.25e-5, 2.02e-7],
        [2.51e-1, 3.98e-5, 6.02e-7],
        [2.12e+0, 1.26e-4, 1.84e-6],
        [4.30e+0, 3.95e-4, 5.76e-6]
    ])

    sigma_s = np.array([
        [2.76e-1, 1.44e-1, 1.27e-2],
        [3.88e-1, 1.76e-1, 7.36e-2],
        [4.77e-1, 3.44e-1, 2.65e-1],
        [6.88e-1, 2.66e-1, 5.72e-1],
        [9.38e-1, 2.06e-1, 6.69e-1],
        [1.52e+0, 2.14e-1, 6.81e-1],
        [2.30e+0, 2.23e-1, 6.82e-1],
        [2.45e+0, 2.31e-1, 6.83e-1],
        [9.79e+0, 2.40e-1, 6.86e-1],
        [4.36e+1, 2.41e-1, 6.91e-1]
    ])

    sigma_t = sigma_f + sigma_c + sigma_s

    #=========================================================#
    #                      Energy Groups                      #
    #=========================================================#
    # Unit is MeV
    Group_Energy = [
        3e+1,  # Group 0
        3e+0,  # Group 1
        3e-1,  # Group 2
        3e-2,  # Group 3
        3e-3,  # Group 4
        3e-4,  # Group 5
        3e-5,  # Group 6
        3e-6,  # Group 7
        3e-7,  # Group 8
        3e-8   # Group 9
    ]

    group = 9  # default to lowest group
    for g in range(len(Group_Energy)):
        if Energy >= Group_Energy[g]:
            group = g
            break

    #=========================================================#
    #                     Cross Sections                      #
    #=========================================================#
    # Unit is 1/cm
    sig_f = sigma_f[group][region]
    sig_c = sigma_c[group][region]
    sig_s = sigma_s[group][region]
    sig_t = sigma_t[group][region]
    sig   = [sig_f, sig_c, sig_s, sig_t]

    return sig


def CalcGroup(Energy):
    # Unit is MeV
    Group_Energy = [
        3e+1,  # Group 0
        3e+0,  # Group 1
        3e-1,  # Group 2
        3e-2,  # Group 3
        3e-3,  # Group 4
        3e-4,  # Group 5
        3e-5,  # Group 6
        3e-6,  # Group 7
        3e-7,  # Group 8
        3e-8   # Group 9
    ]

    group = 9  # default to lowest group
    for g in range(len(Group_Energy)):
        if Energy >= Group_Energy[g]:
            group = g
            break

    return group


def QualifyingMC(Neutrons_Number):
    #=========================================================#
    #                     Initialization                      #
    #=========================================================#
    print("Number of Neutrons.......................= ", Neutrons_Number)

    Neutrons_Produced = 0
    interaction_point_x = []
    interaction_point_y = []
    FuelSurfNeuNum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    CladSurfNeuNum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    Group_Energy = [
        3e+1, 3e+0, 3e-1, 3e-2, 3e-3,
        3e-4, 3e-5, 3e-6, 3e-7, 3e-8
    ]

    Fission    = 0
    nu         = 0
    Capture    = 0
    Absorption = 0
    Scattering = 0
    Leakage    = 0
    LostNoSurface = 0

    #=========================================================#
    #                        Geometry                         #
    #=========================================================#
    r_fuel     = 0.53   # Fuel radius (cm)
    r_clad_in  = 0.53   # Cladding inner radius (cm)
    r_clad_out = 0.90   # Cladding outer radius (cm)
    pitch      = 1.837  # Cell pitch (cm)
    t_clad     = r_clad_out - r_clad_in

    #=========================================================#
    #        Spatial and Energy Distribution of Neutrons      #
    #=========================================================#
    Neutrons_Energy = maxwell.rvs(size=Neutrons_Number)
    theta1          = 2 * np.pi * np.random.random((Neutrons_Number))
    rnd             = np.random.power(2, size=(Neutrons_Number))
    r               = rnd * r_fuel

    #=========================================================#
    #                      Calculations                       #
    #=========================================================#
    for i in range(Neutrons_Number):

        # Initialize per-neutron parameters
        alive        = 1
        interaction  = 0
        boundary     = 0
        regionchange = 0
        surface      = 0
        region       = 0   # [fuel, cladding, moderator] = [0, 1, 2]

        # Initialize neutron position and energy
        E = []
        E.append(Neutrons_Energy[i])
        Energy = E[-1]

        x = []
        y = []
        x.append(r[i] * np.cos(theta1[i]))
        y.append(r[i] * np.sin(theta1[i]))

        plt.plot(x, y, '*g')

        #======================================================#
        #              Track neutron while alive               #
        #======================================================#
        while alive == 1:

            sig = CrossSections(Energy, region)
            # sig = [sig_f, sig_c, sig_s, sig_t]

            #==================================================#
            #                   Fuel Region                    #
            #==================================================#
            if region == 0:
                A       = 238.02891  # Mass number of Uranium
                density = 10.97      # g/cc

                if regionchange == 0:
                    theta = 2 * np.pi * np.random.random()
                else:
                    theta = theta
                regionchange = 0

                # Sample free-flight distance
                d = -(1.0 / sig[3]) * np.log(np.random.random())

                # Check distance to nearest surface
                a     = 1
                b     = 2 * (x[-1] * np.cos(theta) + y[-1] * np.sin(theta))
                c     = x[-1]**2 + y[-1]**2 - r_fuel**2
                delta = b**2 - 4 * a * c

                df    = []
                d_pos = []

                if delta >= 0:
                    df1 = (-b - delta**0.5) / (2 * a)
                    df.append(df1)
                    df2 = (-b + delta**0.5) / (2 * a)
                    df.append(df2)
                else:
                    LostNoSurface += 1
                    break

                for k in range(len(df)):
                    if df[k] > 1e-9:
                        d_pos.append(df[k])
                d_pos.sort()
                if len(d_pos) == 0:
                    LostNoSurface += 1
                    break
                dmin = d_pos[0]

                if d >= dmin:
                    # Neutron leaves fuel → enters cladding
                    FuelSurfNeuNum[CalcGroup(Energy)] += 1
                    regionchange = 1
                    surface      = 1
                    region       = 1
                    x.append(x[-1] + dmin * np.cos(theta))
                    y.append(y[-1] + dmin * np.sin(theta))
                    plt.plot(x[-2:], y[-2:], '-y')
                else:
                    # Neutron interacts inside fuel
                    interaction = 1
                    x.append(x[-1] + d * np.cos(theta))
                    y.append(y[-1] + d * np.sin(theta))
                    plt.plot(x[-2:], y[-2:], '-y')

            #==================================================#
            #                 Cladding Region                  #
            #==================================================#
            if region == 1:
                A       = 26.981539  # Mass number of Aluminum
                density = 2.70       # g/cc

                if regionchange == 0:
                    theta = 2 * np.pi * np.random.random()
                else:
                    theta = theta
                regionchange = 0

                # Sample free-flight distance
                d = -(1.0 / sig[3]) * np.log(np.random.random())

                # Check distance to nearest surface
                a      = 1
                b      = 2 * (x[-1] * np.cos(theta) + y[-1] * np.sin(theta))
                c_in   = x[-1]**2 + y[-1]**2 - r_clad_in**2
                c_out  = x[-1]**2 + y[-1]**2 - r_clad_out**2
                delta_in  = b**2 - 4 * a * c_in
                delta_out = b**2 - 4 * a * c_out

                dc    = []
                d_pos = []

                if delta_in >= 0:
                    dc1 = (-b - delta_in**0.5)  / (2 * a)
                    dc.append(dc1)
                    dc2 = (-b + delta_in**0.5)  / (2 * a)
                    dc.append(dc2)
                    if delta_out >= 0:
                        dc3 = (-b - delta_out**0.5) / (2 * a)
                        dc.append(dc3)
                        dc4 = (-b + delta_out**0.5) / (2 * a)
                        dc.append(dc4)
                    else:
                        LostNoSurface += 1
                        break
                else:
                    dc1 = 1e100
                    dc2 = 1e100
                    if delta_out >= 0:
                        dc3 = (-b - delta_out**0.5) / (2 * a)
                        dc.append(dc3)
                        dc4 = (-b + delta_out**0.5) / (2 * a)
                        dc.append(dc4)
                    else:
                        LostNoSurface += 1
                        break

                for k in range(len(dc)):
                    if dc[k] > 1e-9:
                        d_pos.append(dc[k])
                d_pos.sort()
                if len(d_pos) == 0:
                    LostNoSurface += 1
                    break
                dmin = d_pos[0]

                if d >= dmin:
                    # Neutron leaves cladding
                    CladSurfNeuNum[CalcGroup(Energy)] += 1
                    regionchange = 1
                    if dmin == dc1 or dmin == dc2:
                        # cladding → fuel
                        region  = 0
                        surface = 1
                    elif dmin == dc3 or dmin == dc4:
                        # cladding → moderator
                        region  = 2
                        surface = 1
                    x.append(x[-1] + dmin * np.cos(theta))
                    y.append(y[-1] + dmin * np.sin(theta))
                    plt.plot(x[-2:], y[-2:], '-y')
                else:
                    # Neutron interacts inside cladding
                    interaction = 1
                    x.append(x[-1] + d * np.cos(theta))
                    y.append(y[-1] + d * np.sin(theta))
                    plt.plot(x[-2:], y[-2:], '-y')

            #==================================================#
            #                 Moderator Region                 #
            #==================================================#
            if region == 2:
                A       = 1.00794  # Mass number of H
                density = 1.0      # g/cc

                if regionchange == 0:
                    theta = 2 * np.pi * np.random.random()
                else:
                    theta = theta
                regionchange = 0

                # Sample free-flight distance
                d = -(1.0 / sig[3]) * np.log(np.random.random())

                # Check distance to nearest surface
                a     = 1
                b     = 2 * (x[-1] * np.cos(theta) + y[-1] * np.sin(theta))
                c_out = x[-1]**2 + y[-1]**2 - r_clad_out**2
                delta_out = b**2 - 4 * a * c_out

                dm    = []
                d_pos = []

                if delta_out >= 0:
                    dm1 = (-b - delta_out**0.5) / (2 * a)
                    dm.append(dm1)
                    dm2 = (-b + delta_out**0.5) / (2 * a)
                    dm.append(dm2)
                else:
                    dm1 = 1e100
                    dm2 = 1e100

                # Distance to cell boundaries
                dm3 = ( pitch/2 - x[-1]) / np.cos(theta)  # right
                dm4 = ( pitch/2 - y[-1]) / np.sin(theta)  # top
                dm5 = (-pitch/2 - x[-1]) / np.cos(theta)  # left
                dm6 = (-pitch/2 - y[-1]) / np.sin(theta)  # bottom
                dm.extend([dm3, dm4, dm5, dm6])

                for k in range(len(dm)):
                    if dm[k] > 1e-9:
                        d_pos.append(dm[k])
                d_pos.sort()
                dmin = d_pos[0]

                if d >= dmin:
                    # Neutron leaves moderator region
                    regionchange = 1
                    x.append(x[-1] + dmin * np.cos(theta))
                    y.append(y[-1] + dmin * np.sin(theta))
                    plt.plot(x[-2:], y[-2:], '-y')

                    if dmin == dm1 or dmin == dm2:
                        # moderator → cladding
                        region = 1
                    elif dmin == dm3:
                        # right boundary → reappear on left
                        boundary = 1
                        Leakage += 1
                        x.append(-x[-1])
                        y.append( y[-1])
                    elif dmin == dm4:
                        # top boundary → reappear on bottom
                        boundary = 1
                        Leakage += 1
                        x.append( x[-1])
                        y.append(-y[-1])
                    elif dmin == dm5:
                        # left boundary → reappear on right
                        boundary = 1
                        Leakage += 1
                        x.append(-x[-1])
                        y.append( y[-1])
                    elif dmin == dm6:
                        # bottom boundary → reappear on top
                        boundary = 1
                        Leakage += 1
                        x.append( x[-1])
                        y.append(-y[-1])
                else:
                    # Neutron interacts inside moderator
                    interaction = 1
                    x.append(x[-1] + d * np.cos(theta))
                    y.append(y[-1] + d * np.sin(theta))

            #==================================================#
            #                   Interactions                   #
            #==================================================#
            if interaction == 1:
                interaction = 0
                interaction_point_x.append(x[-1])
                interaction_point_y.append(y[-1])

                rnd = np.random.random()

                if rnd <= (sig[0] / sig[3]):
                    # Fission
                    Fission += 1
                    alive = 0
                    if np.random.random() < 0.5:
                        nf = 2
                    else:
                        nf = 3
                    Neutrons_Produced += nf
                    nu = Neutrons_Produced / Fission

                elif (sig[0] / sig[3]) < rnd <= ((sig[0] + sig[1]) / sig[3]):
                    # Capture
                    Capture += 1
                    alive = 0

                else:
                    # Scattering — neutron slows down
                    Scattering += 1
                    ksi = 1 + np.log((A - 1) / (A + 1)) * (A - 1)**2 / (2 * A)
                    E.append(E[-1] * np.exp(-ksi))
                    Energy = E[-1]

    #=========================================================#
    #                    Plotting Geometry                    #
    #=========================================================#
    plt.gcf().gca().add_artist(plt.Circle((0, 0), r_fuel,     fill=False))
    plt.gcf().gca().add_artist(plt.Circle((0, 0), r_clad_in,  fill=False))
    plt.gcf().gca().add_artist(plt.Circle((0, 0), r_clad_out, fill=False))
    plt.gcf().gca().add_artist(plt.Rectangle(
        (-pitch/2, -pitch/2), width=pitch, height=pitch, fill=False))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    #=========================================================#
    #                   Neutron Flux Spectrum                 #
    #=========================================================#
    A_fuel   = np.pi * r_fuel**2
    c_scale  = Neutrons_Number * 10
    x1       = Group_Energy
    y1       = np.array(FuelSurfNeuNum) / np.array(A_fuel * c_scale)
    x2       = np.linspace(0, 30, 500)
    y2       = 0.453 * np.sinh((2.29 * x2)**0.5) * np.exp(-x2 * 1.036)

    plt.figure()
    plt.plot(x1, y1, x2, y2)
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Flux (n/cm2/s)")
    pylab.plot(x1, y1, '-b', label='Qualifying')
    pylab.plot(x2, y2, '-r', label='Watt')
    pylab.legend(loc='upper right')

    #=========================================================#
    #                        Results                          #
    #=========================================================#
    Absorption   = Fission + Capture
    Interactions = Scattering + Absorption
    Neutrons_Lost = Leakage + Absorption + Fission
    keff = (Neutrons_Produced + Leakage) / Neutrons_Lost

    print("Number of Interactions...................= ", Interactions)
    print("Number of Scattering Events..............= ", Scattering)
    print("Number of Capture Events.................= ", Capture)
    print("Number of Fission Events.................= ", Fission)
    print("Number of Absorption Events..............= ", Absorption)
    print("Average nu...............................= ", nu)
    print("Number of Neutrons Produced by Fission...= ", Neutrons_Produced)
    print("Number of Neutrons Leaked from System....= ", Leakage)
    print("Lost With No Surface.....................= ", LostNoSurface)
    print("Fuel Surface Crossings...................= ", sum(FuelSurfNeuNum))
    print("Clad Surface Crossings...................= ", sum(CladSurfNeuNum))
    print("Square Surface Crossings.................= ", Leakage)
    print("Effective Multiplication Factor(keff)....= ", keff)

    # plt.show()
    return keff


#=========================================================#
#                        Run                             #
#=========================================================#
if __name__ == "__main__":
    keff = QualifyingMC(1000)
