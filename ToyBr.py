# Samuel Grant 2023
# Toy model for a varying radial field (N = 1 term) around the ring azimuth 

# ------------------------------------------------
# External libraries 
# ------------------------------------------------

import numpy as np
from scipy import stats 
from scipy.optimize import curve_fit
import math
import ROOT # We use a ROOT histogram as input
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as colors
from matplotlib.colors import LogNorm

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

def ProfileX(x, y, nBinsX=100, xmin=-1.0, xmax=1.0, nBinsY=100, ymin=-1.0, ymax=1.0): 
   
    # Create 2D histogram with one bin on the y-axis 
    hist, xEdge_, yEdge_ = np.histogram2d(x, y, bins=[nBinsX, nBinsY], range=[[xmin, xmax], [ymin, ymax]])

    # hist, xEdge_ = np.histogram(x, bins=nBinsX, range=[xmin, xmax]) # , [ymin, ymax]])

    # bin widths
    xBinWidths = xEdge_[1]-xEdge_[0]

    # Calculate the mean and RMS values of each vertical slice of the 2D distribution
    # xSlice_, xSliceErr_, ySlice_, ySliceErr_, ySliceRMS_ = [], [], [], [], []
    xSlice_,  ySlice_, ySliceErr_, = [], [], [] 

    for i in range(len(xEdge_) - 1):

        # Average x-value
        xSlice = x[ (xEdge_[i] < x) & (x <= xEdge_[i+1]) ]

        # Get y-slice within current x-bin
        ySlice = y[ (xEdge_[i] < x) & (x <= xEdge_[i+1]) ]

        # Filter out np.nan values
        ySlice = ySlice[~np.isnan(ySlice)]

        # Avoid empty slices
        if len(xSlice) == 0 or len(ySlice) == 0:
            continue

        # Central values are means and errors are standard errors on the mean
        xSlice_.append(np.mean(xSlice))
        # xSliceErr_.append(stats.sem(xSlice)) # RMS/sqrt(n)
        ySlice_.append(ySlice.mean()) 
        ySliceErr_.append(stats.sem(ySlice)) 
        # ySliceRMS_.append(np.std(ySlice))

    return np.array(xSlice_), np.array(ySlice_), np.array(ySliceErr_)

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
    if stats: ax.legend([legend_text], loc="upper right", frameon=False, fontsize=13)

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
        ax.xaxis.offsetText.set_fontsize(13)
    if ax.get_ylim()[1] > 9999 or ax.get_ylim()[1] < 9.999e-3:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(13)

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
        ax.xaxis.offsetText.set_fontsize(13)
    if ax.get_ylim()[1] > 9999 or ax.get_ylim()[1] < 9.999e-3:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(13)

    # Save the figure
    plt.savefig(fout, dpi=300, bbox_inches="tight")
    print("---> Written", fout)

    # Clear memory
    plt.clf()
    plt.close()

    return

    plt.close()

# Scatter plot
def PlotGraphErrors(x, y, xerr=np.array([]), yerr=np.array([]), title=None, xlabel=None, ylabel=None, fout="gr.png"):

   # Create a scatter plot with error bars using NumPy arrays 

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot scatter with error bars
    if len(xerr)==0: xerr = [0] * len(x) # Sometimes we only use yerr
    if len(yerr)==0: yerr = [0] * len(y) # Sometimes we only use xerr

    if len(x) != len(y): print("Warning: x has length", len(x),", while y has length", len(y))

    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', color='black', markersize=4, ecolor='black', capsize=2, elinewidth=1, linestyle='None')

    # Set title, xlabel, and ylabel
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel(xlabel, fontsize=13, labelpad=10) 
    ax.set_ylabel(ylabel, fontsize=13, labelpad=10) 

    # Set font size of tick labels on x and y axes
    ax.tick_params(axis='x', labelsize=13)  # Set x-axis tick label font size
    ax.tick_params(axis='y', labelsize=13)  # Set y-axis tick label font size

     # Scientific notation
    if ax.get_xlim()[1] > 9999:
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.xaxis.offsetText.set_fontsize(13)
    if ax.get_ylim()[1] > 9999:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(13)

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
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel(xlabel, fontsize=13, labelpad=10) 
    ax.set_ylabel(ylabel, fontsize=13, labelpad=10) 

    # Set font size of tick labels on x and y axes
    ax.tick_params(axis='x', labelsize=13)  # Set x-axis tick label font size
    ax.tick_params(axis='y', labelsize=13)  # Set y-axis tick label font size

    if ax.get_xlim()[1] > 9999 or ax.get_xlim()[1] < 9.999e-3:
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.xaxis.offsetText.set_fontsize(13)
    if ax.get_ylim()[1] > 9999 or ax.get_ylim()[1] < 9.999e-3:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(13)

    # Add legend to the plot
    ax.legend(loc="best", frameon=False, fontsize="14", markerscale=10)

    # Save the figure
    plt.savefig(fout, dpi=NDPI, bbox_inches="tight")
    print("---> Written", fout)

    # Clear memory
    plt.clf()
    plt.close()

    
def Plot2D(x, y, nBinsX=100, xmin=-1.0, xmax=1.0, nBinsY=100, ymin=-1.0, ymax=1.0, title=None, xlabel=None, ylabel=None, fout="hist.png", log=False, cb=True):

    # Filter out empty entries from x and y
    valid_indices = [i for i in range(len(x)) if np.any(x[i]) and np.any(y[i])]

    # Extract valid data points based on the indices
    x = [x[i] for i in valid_indices]
    y = [y[i] for i in valid_indices]

    # Check if the input arrays are not empty and have the same length
    if len(x) == 0 or len(y) == 0:
        print("Input arrays are empty.")
        return
    if len(x) != len(y):
        print("Input arrays x and y have different lengths.")
        return

    # Create 2D histogram
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=[nBinsX, nBinsY], range=[[xmin, xmax], [ymin, ymax]])
    x_bin_centres_ = (x_edges[:-1] + x_edges[1:]) / 2

    # Set up the plot
    fig, ax = plt.subplots()

    norm = colors.Normalize(vmin=0, vmax=np.max(hist))  
    if log: norm = colors.LogNorm(vmin=1, vmax=np.max(hist)) 

    # Plot the 2D histogram
    im = ax.imshow(hist.T, cmap='inferno', extent=[xmin, xmax, ymin, ymax], aspect='auto', origin='lower', norm=norm) 
    # im = ax.imshow(hist.T, extent=[xmin, xmax, ymin, ymax], aspect='auto', origin='lower', vmax=np.max(hist))

    # Add colourbar
    if cb: plt.colorbar(im)

    plt.title(title, fontsize=15, pad=10)
    plt.xlabel(xlabel, fontsize=13, labelpad=10)
    plt.ylabel(ylabel, fontsize=13, labelpad=10)

    # Scientific notation
    if ax.get_xlim()[1] > 99999:
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.xaxis.offsetText.set_fontsize(13)
    if ax.get_ylim()[1] > 99999:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(13)

    plt.savefig(fout, dpi=300, bbox_inches="tight")
    print("---> Written", fout)

    # Clear memory
    plt.close()

    return 

# ------------------------------------------------
# Fitting
# ------------------------------------------------

# Simple sine fit, should be good enough
def FitFunction(t, A, phi_g2, c):
    return A * np.sin(OMEGA_A*t + phi_g2) + c 
 
def FitAndPlotGraph(x, y, xerr=np.array([]), yerr=np.array([]), pi_=[], phi_g2=0.0, fitMin=0, fitMax=1, title=None, xlabel=None, ylabel=None, fout="gr.png"):

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot scatter with error bars
    if len(xerr) == 0:
        xerr = [0] * len(x)  # Sometimes we only use yerr
    if len(yerr) == 0:
        yerr = [0] * len(y)  # Sometimes we only use yerr

    if len(x) != len(y):
        print("Warning: x has length", len(x), ", while y has length", len(y))

    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', color='black', markersize=4, ecolor='black', capsize=2, elinewidth=1, linestyle='None', zorder=2)

    # Set title, xlabel, and ylabel
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlabel(xlabel, fontsize=14, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=10)

    # Set font size of tick labels on x and y axes
    ax.tick_params(axis='x', labelsize=14)  # Set x-axis tick label font size
    ax.tick_params(axis='y', labelsize=14)  # Set y-axis tick label font size

    # Scientific notation
    if ax.get_xlim()[1] > 9999:
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.xaxis.offsetText.set_fontsize(14)
    if ax.get_ylim()[1] > 9999:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.yaxis.offsetText.set_fontsize(14)

    # Set fit range
    x_f = x[(x >= fitMin) & (x <= fitMax)]
    y_f = y[(x >= fitMin) & (x <= fitMax)]
    # xerr_f = xerr[(x >= fitMin) & (x <= fitMax)]
    yerr_f = yerr[(x >= fitMin) & (x <= fitMax)]

    # Fix g-2 phase by setting bounds 
    lower_bounds = [-np.inf, pi_[1] - 1e-15, -np.inf]
    upper_bounds = [np.inf, pi_[1] + 1e-15, np.inf]

    # Calculate fit parameters
    pf_, cov = curve_fit(FitFunction, x_f, y_f, sigma=yerr_f, absolute_sigma=True, p0=pi_, bounds=(lower_bounds, upper_bounds))

    # Extract errors
    pferr_ = np.sqrt(np.diag(cov))

    # Calculate the chi-squared value
    fit_residuals = y_f - FitFunction(x_f, *pf_)
    chi_squared = np.sum((fit_residuals / yerr_f)**2)

    # Calculate the degrees of freedom
    dof = len(x_f) - len(pf_)

    # Calculate the reduced chi-squared value
    chi_squared_per_dof = chi_squared / dof

    # Plot the fitted line
    fit_x = np.linspace(float(fitMin), float(fitMax), 100)
    fit_y = FitFunction(fit_x, *pf_)

    ax.plot(fit_x, fit_y, color='red', linestyle='-', label=f"$\\alpha(t)=\delta\cdot\sin(\omega_{{a}} t+\phi)+c$\n$\chi^2/ndf={Round(chi_squared_per_dof, 3)}$\n$\delta={Round(pf_[0], 3)}\pm{Round(pferr_[0], 3)}$\n$\phi={Round(pf_[1], 3)}\pm{Round(pferr_[1], 3)}$\n$c={Round(pf_[2], 3)}\pm{Round(pferr_[2], 3)}$", zorder=3)
    # ax.plot(fit_x, fit_y, color='red', linestyle='-', label=f"$\\alpha(t)=\delta\cdot\sin(\omega_{{a}} t)$\n$\chi^2/ndf={Round(chi_squared_per_dof, 3)}$\n$\delta={Round(pf_[0], 3)}\pm{Round(pferr_[0], 1)}$", zorder=3) 

    # Set the desired y-range
    ax.set_ylim(-42.5, 42.5)  # Set the range you want

    # Save the figure
    plt.legend(loc="best", frameon=False, fontsize=13)
    plt.savefig(fout, dpi=300, bbox_inches="tight")
    print("---> Written", fout)

    print("A =", pf_[0], "+/-", pferr_[0])
    # print("phi =", pf_[1], "+/-", pferr_[1])
    # print("c =", pf_[1], "+/-", pferr_[1])

    # Clear memory
    plt.clf()
    plt.close()

    return

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
# Vertical polarisation angle
# ------------------------------------------------

# Vertical polarisation
# Assume everything is lab frame
def PolY(t, phi_g2 = 0, boundaries=np.array([])): # , labFrame=False):

    # Modulate
    phi = OMEGA_C*t % (2 * np.pi)

    # Tilt angle at this azimuth
    tilt = Br(phi) 

    # Polarisation angle oscillation, phase is zero
    pol_y = (0 * np.cos(OMEGA_A * t + phi_g2)) + (tilt * np.sin(OMEGA_A * t + phi_g2)) # g-2 is cosine, Br is sine 

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

# Input number of decays and g-2 phase
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
    PlotGraphErrors(x=phi_, y=Br_, xlabel="Ring azimuth [rad]", ylabel="$B_{r}$ [ppm] / 0.01 rad", fout="Images/gr_Br_vs_phi.png")

    # Generate decay times by sampling from a exponential distribution
    # I think this is important so you're not just sampling the same g-2 phase every cyclotron period
    np.random.seed(12345)
    lifetime = TAU*GMAGIC
    t_ = np.random.exponential(scale=lifetime, size=int(n_decays)) 
    t_ = np.clip(t_, 0, 700) # Clip the samples between 0 and 700 us
    # Sanity plot
    Plot1D_A(t_, 70, 0, 700, xlabel="Time [$\mu$s]", ylabel="Decays / 10 $\mu$s", fout="Images/h1_decay_times_"+str(int(n_decays))+".png")

    # Modulate these times over the g-2 period
    t_mod_ = t_ % G2PERIOD

    # Sanity plot
    Plot1D_A(t_mod_, 700, 0, G2PERIOD, xlabel="Time modulo $g-2$ [$\mu$s]", ylabel="Decays / $\mu$s", fout="Images/h1_t_mod_"+str(int(n_decays))+".png")

    # ----------------------------------------

    # Now we set up the acceptance 
    acc_weights_ = GetAcceptanceWeights(acc_)

    # Accepted Br, for illustration
    Br_acc_ = Br_ * acc_weights_ 
    PlotGraphOverlay(x=phi_, y1=Br_, y2=Br_acc_, xlabel="Ring azimuth [rad]", ylabel="$B_{r}$ [ppm] / 0.01 rad", fout="Images/gr_Br_vs_phi_acc_"+str(int(n_decays))+".png")

    # Get acceptance boundaries
    boundaries_ = GetAcceptanceBoundaries(Br_acc_)

    # This is dumb. Convert the boundaries into phi here
    boundaries_phi_ = np.array([phi_[boundary] for boundary in boundaries_])

    # ----------------------------------------

    # Get the vertical polarisation over a range of g-2 phases
    phi_g2_ = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]

    for phi_g2 in phi_g2_:

        phi_g2_str = str(int(phi_g2 * 180 / np.pi))
        print(phi_g2_str)

        # Generate the vertical polarisation over these times with a varying radial field
        pol_y_ = np.array([PolY(t, phi_g2=phi_g2) for t in t_]) 

        # Plot the vertical polarisation with full acceptance
        PlotGraphErrors(x=t_, y=pol_y_, xlabel="Time [$\mu$s]", ylabel=r"$\alpha$ [$\mu$rad]", fout="Images/gr_pol_y_vs_t_"+str(int(n_decays))+"_"+phi_g2_str+"deg.png")
        PlotGraphErrors(x=t_mod_, y=pol_y_, xlabel="Time modulo $g-2$ [$\mu$s]", ylabel=r"${\alpha}$ [$\mu$deg]", fout="Images/gr_pol_y_vs_t_mod_"+str(int(n_decays))+"_"+phi_g2_str+"deg.png")
        Plot1D_A(pol_y_, nBins=240, xmin=-120., xmax=120., xlabel=r"${\alpha}$ [$\mu$deg]", ylabel="Decays / $\mu$rad", fout="Images/h1_pol_y_"+str(int(n_decays))+"_"+phi_g2_str+"deg.png", errors=True, underOver=False)
        Plot2D(x=t_mod_, y=pol_y_, nBinsX=1000, xmin=0, xmax=G2PERIOD, nBinsY=480, ymin=-110, ymax=110, xlabel="Time modulo $g-2$ [$\mu$s]", ylabel=r"${\alpha}$ [$\mu$rad]", fout="Images/h2_pol_y_vs_t_"+str(int(n_decays))+"_"+phi_g2_str+"deg.png", cb=False, log=False) 
        Plot2D(x=t_mod_, y=pol_y_, nBinsX=int(G2PERIOD/T_c), xmin=0, xmax=G2PERIOD, nBinsY=480, ymin=-110, ymax=110, xlabel="Time modulo $g-2$ [$\mu$s]", ylabel=r"${\alpha}$ [$\mu$rad]", fout="Images/h2_pol_y_vs_t_rebin_"+str(int(n_decays))+"_"+phi_g2_str+"deg.png", cb=False, log=False) 
        
        # Now generate the polarisation for the accepted Br 
        pol_y_acc_ = np.array([PolY(t, phi_g2=phi_g2, boundaries=boundaries_phi_) for t in t_]) 

        # Plot the vertical polarisation with partial acceptance
        PlotGraphErrors(x=t_, y=pol_y_acc_, xlabel="Time [$\mu$s]", ylabel=r"${\alpha}$ [$\mu$rad]", fout="Images/gr_pol_y_vs_t_acc_"+str(int(n_decays))+"_"+phi_g2_str+"deg.png")
        PlotGraphErrors(x=t_mod_, y=pol_y_acc_, xlabel="Time modulo $g-2$ [$\mu$s]", ylabel=r"${\alpha}$ [$\mu$rad]", fout="Images/gr_pol_y_vs_t_mod_acc_"+str(int(n_decays))+"_"+phi_g2_str+"deg.png")
        Plot1D_A(pol_y_acc_, nBins=240, xmin=-120., xmax=120., xlabel=r"${\alpha}$ [$\mu$rad]", ylabel="Decays / $\mu$rad", fout="Images/h1_pol_y_acc_"+str(int(n_decays))+"_"+phi_g2_str+"deg.png", errors=True, underOver=False)
        Plot2D(x=t_mod_, y=pol_y_acc_, nBinsX=1000, xmin=0, xmax=G2PERIOD, nBinsY=240, ymin=-110, ymax=110, xlabel="Time modulo $g-2$ [$\mu$s]", ylabel=r"${\alpha}$ [$\mu$rad]", fout="Images/h2_pol_y_acc_vs_t_"+str(int(n_decays))+"_"+phi_g2_str+"deg.png", cb=False, log=False) 
        Plot2D(x=t_mod_, y=pol_y_acc_, nBinsX=int(G2PERIOD/T_c), xmin=0, xmax=G2PERIOD, nBinsY=240, ymin=-110, ymax=110, xlabel="Time modulo $g-2$ [$\mu$s]", ylabel=r"${\alpha}$ [$\mu$rad]", fout="Images/h2_pol_y_acc_vs_t_rebin_"+str(int(n_decays))+"_"+phi_g2_str+"deg.png", cb=False, log=False)

        # Rebin pol_y vs t_mod 
        t_mod_rebin_, pol_y_rebin_, pol_y_rebin_err_ = ProfileX(x=t_mod_, y=pol_y_, nBinsX=int(G2PERIOD/T_c), xmin=0, xmax=G2PERIOD, nBinsY=240, ymin=-110, ymax=110)
        t_mod_acc_rebin_, pol_y_acc_rebin_, pol_y_acc_rebin_err_ = ProfileX(x=t_mod_, y=pol_y_acc_, nBinsX=int(G2PERIOD/T_c), xmin=0, xmax=G2PERIOD, nBinsY=240, ymin=-110, ymax=110)
    
        PlotGraphErrors(x=t_mod_rebin_, y=pol_y_rebin_, yerr=pol_y_rebin_err_, xlabel="Time modulo $g-2$ [$\mu$s]", ylabel=r"${\alpha}$ [$\mu$rad]", fout="Images/gr_pol_y_vs_t_mod_rebin_"+str(int(n_decays))+"_"+phi_g2_str+"deg.png")
        PlotGraphErrors(x=t_mod_acc_rebin_, y=pol_y_acc_rebin_, yerr=pol_y_acc_rebin_err_, xlabel="Time modulo $g-2$ [$\mu$s]", ylabel=r"${\alpha}$ [$\mu$rad]", fout="Images/gr_pol_y_acc_vs_t_mod_rebin_"+str(int(n_decays))+"_"+phi_g2_str+"deg.png")

        # Fit
        FitAndPlotGraph(x=t_mod_rebin_, y=pol_y_rebin_, yerr=pol_y_rebin_err_, pi_=[0.0, phi_g2, 0.0], fitMin=0, fitMax=G2PERIOD, xlabel="Time modulo $g-2$ [$\mu$s]", ylabel=r"${\alpha}$ [$\mu$rad]", fout="Images/gr_fit_pol_y_vs_t_mod_rebin_"+str(int(n_decays))+"_"+phi_g2_str+"deg.png")
        FitAndPlotGraph(x=t_mod_acc_rebin_, y=pol_y_acc_rebin_, yerr=pol_y_acc_rebin_err_, pi_=[-50, phi_g2, 0.0], fitMin=0, fitMax=G2PERIOD, xlabel="Time modulo $g-2$ [$\mu$s]", ylabel=r"${\alpha}$ [$\mu$rad]", fout="Images/gr_fit_pol_y_acc_vs_t_mod_rebin_"+str(int(n_decays))+"_"+phi_g2_str+"deg.png")

    return

# ------------------------------------------------
# main 
# ------------------------------------------------

def main():

    n_samples = 1e6
    RunToyBr(n_samples) 


    return

if __name__ == "__main__":
    main()