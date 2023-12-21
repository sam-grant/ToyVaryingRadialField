# Samuel Grant 2023
# Toy model for a varying radial field (N = 1 term) around the ring azimuth 

# ------------------------------------------------
# External libraries 
# ------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy import stats
import math
import ROOT # We use a ROOT histogram as input

# ------------------------------------------------
# Globals
# ------------------------------------------------

OMEGA_A = 1.439359 # rad/us 1.439311; // (average for mu+ at BNL) 0.00143934*1e3;//1.439311; // rad/us 0.00143934; // kHz from gm2const, it's an angular frequency though...
G2PERIOD = (2 * np.pi / OMEGA_A) # us
M_MU = 105.6583715 #  MeV
A_MU = 11659208.9e-10
GMAGIC = np.sqrt( 1.+1./A_MU )
TAU = 2.196981 # rest frame lifetime
PMAX = 1.01 * M_MU * GMAGIC # 3127.1144
T_c = 149.2 * 1e-3 # cyclotron period [us]
OMEGA_C = 2 * np.pi / T_c # cyclotron angular frequency [rad/us]

# ------------------------------------------------
# Helper functions
# ------------------------------------------------

def Round(value, sf):

    if value == 0.00:
        return "0"
    elif math.isnan(value):
        return "NaN"
    else:
        # Determine the order of magnitude
        magnitude = math.floor(math.log10(abs(value))) + 1
        # Calculate the scale factor
        scale_factor = sf - magnitude
        # Truncate the float to the desired number of significant figures
        truncated_value = math.trunc(value * 10 ** scale_factor) / 10 ** scale_factor
        # Convert the truncated value to a string
        truncated_str = str(truncated_value).rstrip('0').rstrip('.')
        return truncated_str

# Stats for histograms tends to assume a normal distribution
# ROOT does the same thing with TH1
def GetBasicStats(data, xmin, xmax):

    filtered_data = data[(data >= xmin) & (data <= xmax)]  # Filter data within range

    N = len(filtered_data)                      
    mean = np.mean(filtered_data)  
    meanErr = stats.sem(filtered_data) # Mean error (standard error of the mean from scipy)
    stdDev = np.std(filtered_data) # Standard deviation
    stdDevErr = np.sqrt(stdDev**2 / (2*N)) # Standard deviation error assuming normal distribution
    underflows = len(data[data < xmin]) # Number of underflows
    overflows = len(data[data > xmax])

    return N, mean, meanErr, stdDev, stdDevErr, underflows, overflows

# ------------------------------------------------
# Plotting
# ------------------------------------------------

# Take a 1D array and histogram from there
def Plot1D_A(data, nBins=100, xmin=-1.0, xmax=1.0, title=None, xlabel=None, ylabel=None, fout="h1.png", stats=True, errors=False, underOver=False): # peak=False, , errors=False
    
    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot the histogram with outline
    counts, bin_edges, _ = ax.hist(data, bins=nBins, range=(xmin, xmax), histtype='step', edgecolor='black', linewidth=1.0, fill=False, density=False)

    # Set x-axis limits
    ax.set_xlim(xmin, xmax)

    # Calculate statistics
    N, mean, meanErr, stdDev, stdDevErr, underflows, overflows = GetBasicStats(data, xmin, xmax)

    # Create legend text
    legend_text = f"Entries: {N}\nMean: {Round(mean, 3)}\nStd Dev: {Round(stdDev, 3)}"
    if errors: legend_text = f"Entries: {N}\nMean: {Round(mean, 2)}$\pm${Round(meanErr, 1)}\nStd Dev: {Round(stdDev, 3)}$\pm${Round(stdDevErr, 1)}"
    # if peak and not errors: legend_text += f"\nPeak: {Round(GetMode(data, nBins / (xmax - xmin))[0], 3)}"
    # if peak and errors: legend_text += f"\nPeak: {Round(GetMode(data, nBins / (xmax - xmin))[0], 3)}$\pm${Round(GetMode(data, nBins / (xmax - xmin))[1], 1)}"
    if underOver: legend_text += f"\nUnderflows: {underflows}\nOverflows: {overflows}"

    # Add legend to the plot
    if stats: ax.legend([legend_text], loc="best", frameon=False, fontsize=13)

    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel(xlabel, fontsize=13, labelpad=10) 
    ax.set_ylabel(ylabel, fontsize=13, labelpad=10) 

    # Set font size of tick labels on x and y axes
    ax.tick_params(axis='x', labelsize=13)  # Set x-axis tick label font size
    ax.tick_params(axis='y', labelsize=13)  # Set y-axis tick label font size

    # Make the scientific notation on the axes look a bit better
    if ax.get_xlim()[1] > 9999 or ax.get_xlim()[1] < 9.999e-3:
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.xaxis.offsetText.set_fontsize(14)
    if ax.get_ylim()[1] > 9999 or ax.get_ylim()[1] < 9.999e-3:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(14)

    # Save the figure
    plt.savefig(fout, dpi=300, bbox_inches="tight")
    print("---> Written", fout)

    # Clear memory
    plt.clf()
    plt.close()

# Use bin contents and edges in this case 
def Plot1D_B(values, bin_edges, xlabel=None, ylabel=None, fout="h1.png"):

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot the histogram using plt.hist
    ax.hist(bin_edges[:-1], bins=bin_edges, weights=values, histtype='step', edgecolor='black', linewidth=1.0, fill=False, density=False, log=True)
    
    # Set x and y axis labels
    ax.set_xlabel(xlabel, fontsize=13, labelpad=10) 
    ax.set_ylabel(ylabel, fontsize=13, labelpad=10) 

    # Set font size of tick labels on x and y axes
    ax.tick_params(axis='x', labelsize=13)  # Set x-axis tick label font size
    ax.tick_params(axis='y', labelsize=13)  # Set y-axis tick label font size

    if ax.get_xlim()[1] > 9999 or ax.get_xlim()[1] < 9.999e-3:
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.xaxis.offsetText.set_fontsize(14)
    if ax.get_ylim()[1] > 9999 or ax.get_ylim()[1] < 9.999e-3:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(14)

    # Save the figure
    plt.savefig(fout, dpi=300, bbox_inches="tight")
    print("---> Written", fout)

    # Clear memory
    plt.clf()
    plt.close()

    return

# Scatter plot
def PlotGraph(x, y, title=None, xlabel=None, ylabel=None, fout="gr.png"):

    # Create a scatter plot with error bars using NumPy arrays 

    # Create figure and axes
    fig, ax = plt.subplots()

    # Fine graphs for CRV visualisation
    ax.scatter(x, y, color='black', s=0.25, edgecolor='black', marker='o', linestyle='None')

    # Set title, xlabel, and ylabel
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel(xlabel, fontsize=13, labelpad=10) 
    ax.set_ylabel(ylabel, fontsize=13, labelpad=10) 

    # Set font size of tick labels on x and y axes
    ax.tick_params(axis='x', labelsize=14)  # Set x-axis tick label font size
    ax.tick_params(axis='y', labelsize=14)  # Set y-axis tick label font size

    if ax.get_xlim()[1] > 9999 or ax.get_xlim()[1] < 9.999e-3:
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.xaxis.offsetText.set_fontsize(14)
    if ax.get_ylim()[1] > 9999 or ax.get_ylim()[1] < 9.999e-3:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(14)

    # Save the figure
    plt.savefig(fout, dpi=300, bbox_inches="tight")
    print("---> Written", fout)

    # Clear memory
    plt.clf()
    plt.close()

def PlotGraphOverlay(x, y1, y2, title=None, xlabel=None, ylabel=None, fout="gr.png", NDPI=300):

    # Create figure and axes
    fig, ax = plt.subplots()

    # Fine graphs for CRV visualisation
    ax.scatter(x, y1, color='black', s=1.0, edgecolor='black', marker='o', linestyle='None', label='Full')

    # Mask y2 where it's equal to zero
    # y2_masked = np.ma.masked_where(y2 == 0, y2)

    ax.scatter(x, y2, color="red", s=1.0, edgecolor="red", marker='o', linestyle='None', label='Accepted')

    # Draw lines at the boundaries of the masked region
    # Identify the boundaries
    boundaries_ = GetAcceptanceBoundaries(y2)
    for boundary in boundaries_:
        ax.axvline(x=x[boundary], color='red', linestyle='--', linewidth=1)

    # Set title, xlabel, and ylabel
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlabel(xlabel, fontsize=14, labelpad=10) 
    ax.set_ylabel(ylabel, fontsize=14, labelpad=10) 

    # Set font size of tick labels on x and y axes
    ax.tick_params(axis='x', labelsize=14)  # Set x-axis tick label font size
    ax.tick_params(axis='y', labelsize=14)  # Set y-axis tick label font size

    if ax.get_xlim()[1] > 9999 or ax.get_xlim()[1] < 9.999e-3:
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.xaxis.offsetText.set_fontsize(14)
    if ax.get_ylim()[1] > 9999 or ax.get_ylim()[1] < 9.999e-3:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(14)

    # Add legend to the plot
    ax.legend(loc="best", frameon=False, fontsize="14", markerscale=10)

    # Save the figure
    plt.savefig(fout, dpi=NDPI, bbox_inches="tight")
    print("---> Written", fout)

    # Clear memory
    plt.clf()
    plt.close()

# ------------------------------------------------
# Azimuthal acceptance
# ------------------------------------------------ 

def GetAzimuthalAcceptance(stn="S12S18"):

    # Open the ROOT file
    file = ROOT.TFile.Open("Data/TrackAcceptance.root")

    # Get the histograms
    h1_phi_decays = file.Get("1000_2500_MeV/AllDecays/Phi")
    h1_phi_tracks = file.Get("1000_2500_MeV/Tracks/"+stn+"_Phi")

    # Access the values
    values_decays = np.array([h1_phi_decays.GetBinContent(bin) for bin in range(1, h1_phi_decays.GetNbinsX() + 1)])
    values_tracks = np.array([h1_phi_tracks.GetBinContent(bin) for bin in range(1, h1_phi_tracks.GetNbinsX() + 1)])

    # Access bin edges
    bin_edges_decays = np.array([h1_phi_decays.GetBinLowEdge(bin) for bin in range(1, h1_phi_decays.GetNbinsX() + 2)])
    bin_edges_tracks = np.array([h1_phi_tracks.GetBinLowEdge(bin) for bin in range(1, h1_phi_tracks.GetNbinsX() + 2)])

    bin_edges = np.array([])

    if (bin_edges_decays.all() != bin_edges_tracks.all()):
        print("Error: bin edges between tracks and decays differ!")
    else: 
        bin_edges = bin_edges_decays

    # Get ratio
    values_tracks_over_decays = values_tracks / values_decays # [track / decay for track, decay in zip(values_tracks, values_decays)]

    # Normalise to maximum bin
    max_value = np.max(values_tracks_over_decays)
    values_tracks_over_decays_norm = values_tracks_over_decays / max_value

    # Plot
    Plot1D_B(values_decays, bin_edges, xlabel="Ring azimuth [rad]", ylabel="Decays / 0.01 rad", fout="Images/h1_phi_decays.png")
    Plot1D_B(values_tracks, bin_edges, xlabel="Ring azimuth [rad]", ylabel="Decays / 0.01 rad", fout="Images/h1_phi_tracks.png")
    Plot1D_B(values_tracks_over_decays, bin_edges, xlabel="Ring azimuth [rad]", ylabel="Acceptance ratio / 0.01 rad", fout="Images/h1_phi_tracks_over_decays.png")
    Plot1D_B(values_tracks_over_decays_norm, bin_edges, xlabel="Ring azimuth [rad]", ylabel="Acceptance / 0.01 rad", fout="Images/h1_phi_tracks_over_decays_norm.png")

    # Close file
    file.Close()

    return values_tracks_over_decays_norm, bin_edges

# Flat acceptance weights (I think this is the right approach)
# You could weight them according to the varying acceptance within the region the tracker sees, but I can't think of a good reason to do this. 
def GetAcceptanceWeights(acc_):
    weights_ = np.array([])
    for i in range(len(acc_)):
        weight = 1 if acc_[i] != 0 else np.nan 
        weights_ = np.append(weights_, weight)
    return weights_

# TODO: make this so it just outputs units of phi
def GetAcceptanceBoundaries(Br_acc_):

    # Identify boundaries where y transitions from np.nan to a float
    changes_to_float = np.where(~np.isnan(Br_acc_[:-1]) & np.isnan(Br_acc_[1:]))[0]
    
    # Identify boundaries where y transitions from a float to np.nan
    changes_to_nan = np.where(np.isnan(Br_acc_[:-1]) & ~np.isnan(Br_acc_[1:]))[0]
    
    # Combine and sort the boundary indices
    boundaries = np.sort(np.concatenate([changes_to_float, changes_to_nan]))

    return boundaries

# ------------------------------------------------
# Radial field 
# ------------------------------------------------

# A sine wave at the cyclotron frequency
def Br(phi):
    return 0 + 100. * np.sin(phi) # N_0 + N_1, let a_1 = 100 ppm

# ------------------------------------------------
# Vertical polarisation 
# ------------------------------------------------

def PolY(t, boundaries=np.array([]), labFrame=False):

    phi = OMEGA_C*t % 2 * np.pi

    # Tilt angle 
    tilt = Br(phi) # Muon rest frame
    if labFrame: tilt = np.arctan( np.tan(tilt) / GMAGIC ) # Lab frame
    # Polarisation
    pol_y = (0 * np.cos(OMEGA_A * t)) + (tilt * np.sin(OMEGA_A * t)) # g-2 is cosine, Br is sine 

    # Are we within the azimuthal acceptance of the trackers?
    if len(boundaries) > 0:
        acceptance_condition = (phi >= boundaries[0] and phi <= boundaries[1] ) or (phi >= boundaries[2] and phi <= boundaries[3])
        if acceptance_condition: return pol_y
        else: return np.nan
    else: 
        return pol_y


# ------------------------------------------------
# Run 
# ------------------------------------------------

def RunToyBr(n_decays=1e4):  

    # Get acceptance over a range of azimuths
    # Do this first so we have a consistent set of phi_ throughout
    # We'll come back to acc_ later...
    acc_, phi_bin_edges_ = GetAzimuthalAcceptance() # stn="S18")
    # Convert bin edges to bin centers to an array of azimuths 
    phi_ = (phi_bin_edges_[:-1] + phi_bin_edges_[1:]) / 2

    # Varying radial field for these azimuths (full ring)
    Br_ = np.array([Br(phi) for phi in phi_]) 
    # Sanity plot
    PlotGraph(x=phi_, y=Br_, xlabel="Ring azimuth [rad]", ylabel="$B_{r}$ [ppm] / 0.01 rad", fout="Images/Br_vs_phi.png")

    # Generate decay times from a exponential distribution
    # I think this is important so you're not just sampling the same g-2 phase every cyclotron period
    lifetime = TAU*GMAGIC
    t_ = np.random.exponential(scale=lifetime, size=int(n_decays)) 
    t_ = np.clip(t_, 0, 700) # Clip the samples between 0 and 700 us
    # Sanity plot
    Plot1D_A(t_, 70, 0, 700, xlabel="Time [$\mu$s]", ylabel="Decays / 10 $\mu$s", fout="Images/decay_times_"+str(int(n_decays))+".png")

    # Modulate these times over the g-2 period
    t_mod_ = t_ % G2PERIOD
    # Sanity plot
    Plot1D_A(t_mod_, 700, 0, G2PERIOD, xlabel="Time modulo $g-2$ [$\mu$s]", ylabel="Decays / $\mu$s", fout="Images/t_mod_"+str(int(n_decays))+".png")
    
    # Generate the vertical polarisation over these times with a varying radial field
    pol_y_ = np.array([PolY(t) for t in t_]) 

    # Vertical polarisation with full acceptance
    PlotGraph(t_, pol_y_, xlabel="Time [$\mu$s]", ylabel="Vertical polarisation [$\mu$rad]", fout="Images/pol_y_vs_t_"+str(int(n_decays))+".png")
    PlotGraph(t_mod_, pol_y_, xlabel="Time modulo $g-2$ [$\mu$s]", ylabel="Vertical polarisation [$\mu$rad]", fout="Images/pol_y_vs_t_mod_"+str(int(n_decays))+".png")
    Plot1D_A(pol_y_, nBins=230, xmin=-115., xmax=115., xlabel="Vertical polarisation [$\mu$rad]", ylabel="Decays / $\mu$rad", fout="Images/pol_y_"+str(int(n_decays))+".png", errors=True, underOver=True)

    # Now we include the acceptance 
    acc_weights_ = GetAcceptanceWeights(acc_)

    # Accepted Br, for illustration
    Br_acc_ = Br_ * acc_weights_ 
    PlotGraphOverlay(x=phi_, y1=Br_, y2=Br_acc_, xlabel="Ring azimuth [rad]", ylabel="$B_{r}$ [ppm] / 0.01 rad", fout="Images/Br_vs_phi_acc_"+str(int(n_decays))+".png")

    # Get acceptance boundaries
    # boundaries_ = GetAcceptanceBoundaries(Br_acc_)
    boundaries_ = GetAcceptanceBoundaries(Br_acc_) # Br_acc_)

    # This is dumb. Convert the boundaries into phi here
    boundaries_phi_ = np.array([phi_[boundary] for boundary in boundaries_])
    
    # Now generate the polarisation for the accepted Br 
    pol_y_acc_ = np.array([PolY(t, boundaries=boundaries_phi_) for t in t_]) 

    # Plot the accepted polarisation
    PlotGraph(t_, pol_y_acc_, xlabel="Time [$\mu$s]", ylabel="Vertical polarisation [$\mu$rad]", fout="Images/pol_y_vs_t_acc_"+str(int(n_decays))+".png")
    PlotGraph(t_mod_, pol_y_acc_, xlabel="Time modulo $g-2$ [$\mu$s]", ylabel="Vertical polarisation [$\mu$rad]", fout="Images/pol_y_vs_t_mod_acc_"+str(int(n_decays))+".png")
    
    Plot1D_A(pol_y_acc_, nBins=230, xmin=-115., xmax=115., xlabel="Vertical polarisation [$\mu$rad]", ylabel="Decays / $\mu$rad", fout="Images/pol_y_acc_"+str(int(n_decays))+".png", errors=True, underOver=True)

    # Get stats
    N, mean, meanErr, stdDev, stdDevErr, underflows, overflows = GetBasicStats(pol_y_, -100, 100)
    N_acc, mean_acc, meanErr_acc, stdDev_acc, stdDevErr_acc, underflows_acc, overflows_acc = GetBasicStats(pol_y_acc_, -100, 100)

    # Printout
    print("\nStatistics for the vertical polarisation WITHOUT acceptance:")
    print(f"  N          : {N}")
    print(f"  Mean       : {mean} ± {meanErr}")
    print(f"  StdDev     : {stdDev} ± {stdDevErr}")
    print(f"  Underflows : {underflows}")
    print(f"  Overflows  : {overflows}")

    print("\nStatistics for the vertical polarisation WITH acceptance:")
    print(f"  N          : {N_acc}")
    print(f"  Mean       : {mean_acc} ± {meanErr_acc}")
    print(f"  StdDev     : {stdDev_acc} ± {stdDevErr_acc}")
    print(f"  Underflows : {underflows_acc}")
    print(f"  Overflows  : {overflows_acc}")

    return



# ------------------------------------------------
# main 
# ------------------------------------------------

def main():
    
    n_samples = 1e7
    RunToyBr(n_samples) 

    return

if __name__ == "__main__":
    main()