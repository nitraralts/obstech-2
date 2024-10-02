import pylab as pl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from numpy import genfromtxt
from scipy.signal import savgol_filter
from astropy.io import fits
from astropy import units as u
from astropy.modeling.polynomial import Polynomial1D
from astropy.modeling.fitting import LinearLSQFitter
from astropy.modeling.models import Linear1D
from astroquery.nist import Nist
import glob
import warnings
plt.rcParams['image.origin'] = 'lower'
plt.matplotlib.style.use('dark_background')

# found this beautiful extrema code online. will find local, non-endpoint maxima for use as pixel guesses, assuming the PSD is roughly normal
def find_maxima(array):
    slope = np.sign(np.diff(array))  # 1 if ascending,0 if flat, -1 if descending
    not_flat, = slope.nonzero() # Indices where data is not flat.
    local_max_inds, = np.where(np.diff(slope[not_flat])==-2)
    return local_max_inds + 1

def check_array_closeness(array1:np.ndarray, array2:np.ndarray, seperation_skepticism:int):
    # this is designed for flattened arrays.
    array1 = array1.flatten()
    length_1 = np.size(array1)
    array2 = array2.flatten()
    length_2 = np.size(array2)
    diff_matrix = np.ones(length_1*length_2)
    for i in range(np.size(diff_matrix)):
        diff_matrix[i] = np.abs(array1[i%length_1] - array2[i//length_1 - 1])
    diff_matrix = np.reshape(diff_matrix, (length_2,length_1))
    check_closeness_axis1 = np.min(diff_matrix,axis=0)/seperation_skepticism
    check_closeness_axis2 = np.min(diff_matrix,axis=1)/seperation_skepticism
    if(all(check_closeness_axis1 < 1) or all(check_closeness_axis2 <= 1)):
        return True
    else:
        return False

def pick_close_indices(array1:np.ndarray, array2:np.ndarray):
    #like the above code, but returns indices of the second array that are closest to the first array
    array1 = array1.flatten()
    length_1 = np.size(array1)
    array2 = array2.flatten()
    length_2 = np.size(array2)
    larger_axes = np.argsort(np.array([length_1,length_2]))
    inds = []
    for i in array1:
        inds.append(np.argmin(np.abs(array2 - i)))
    return np.array(inds)
    
def check_lineups(data_array, guesspix, appx_wavelength, guess_depth = 1, csv_name='bright_neon_lines.csv', supervised = False, linecount_leeway = 4, pixel_seperation_skepticism = 150):
    # I need the closest to central guess we have.
    data_width = np.shape(data_array)[1]
    closest_guess_to_center = guesspix[np.argmin(np.abs(guesspix-data_width/2))]
    
    # I would like to sort the guesses closest to that for scaling
    guesspix_closest = guesspix[np.argsort(np.abs(guesspix-closest_guess_to_center))[1:]]
    
    # Next I want to figure out what lines are close to my center.
    line_proportions = genfromtxt(csv_name, delimiter=',')[1:,1:]
    visible_neon_lines = genfromtxt(csv_name,delimiter=',')[1:,0]
    wavelength_args_by_prox = np.argsort(np.abs(visible_neon_lines - appx_wavelength))
    wavelengths_by_prox = visible_neon_lines[wavelength_args_by_prox]
    
    # Finally I would like to try each of those lines as my center and ask about a plot with this as the case.
    for i, wavelength in enumerate(wavelengths_by_prox):
        for near_wavelength in guesspix_closest[:guess_depth]:
            wavelength_ratio_scale = near_wavelength - closest_guess_to_center
            flipped = np.sign(wavelength_ratio_scale)
            wavelength_ratio_scale *= flipped
            test_wavelength_lineup = (closest_guess_to_center + (line_proportions[:,wavelength_args_by_prox[i]] * wavelength_ratio_scale * flipped))
            windowed_guesses_args = np.asarray(np.abs(test_wavelength_lineup - data_width/2) < data_width/2).nonzero()
            # If supervised then ask - does this look good? Put those human eyes to use!
            if(supervised):
                # plot solution to be checked
                plt.imshow(data_array, cmap='gray', norm=LogNorm())
                for j, line in enumerate(test_wavelength_lineup[windowed_guesses_args]):
                    plt.axvline(line, linestyle='--')
                    plt.text(line, 4400, str(visible_neon_lines[windowed_guesses_args][j]), rotation=90, ha='right', va='top', color='C0')
                plt.title(wavelength)
                plt.xlim(0, data_width)
                plt.show()
                # ask if this lineup is good
                checking_this_lineup = True
                while(checking_this_lineup):
                    is_good = input("is this lined up? Y/N").upper()
                    if(is_good == "Y" or is_good == "YES"):
                        guesses = visible_neon_lines[windowed_guesses_args]
                        return(guesses)
                        break
                    elif(is_good == 'N' or is_good == "NO"):
                        checking_this_lineup = False
                    else:
                        print("Please answer 'Y' or 'N'! 'Y' means the right lines have been placed, even if the angle is off. Hopefully, it catches all visible lines! Otherwise, put 'N'. It should be pretty obvious :)")
            # If not supervised, do your best to supervise yourself
            else:
                # Criterion one: there should be roughly the same number of lines
                if(np.abs(np.size(windowed_guesses_args) - np.size(guesspix)) <= linecount_leeway):
                    # Criterion two: Each guess pixel should have a line close to it (within "pixel seperation skepticism")
                    # Discrepancy should be explainable by the above leeway - this function checks that teh
                    if(check_array_closeness(test_wavelength_lineup, guesspix, pixel_seperation_skepticism)):
                        # Cut down on lines - only have ones corresponding to output
                        possible_wl = visible_neon_lines[windowed_guesses_args]
                        guess_wl = possible_wl[pick_close_indices(guesspix, test_wavelength_lineup[windowed_guesses_args])]
                        return(guess_wl)
                        break           
                
# Input a micrometer setting, output an approximate (center) wavelength - need advising on the best way to do this
# Until refinement, I have just gone with a linear fit of wavelength/micrometer setting with the current calibration of LHIRES based on my own data and some from other students' fits
def approximate_wavelength(x):
    return(255.4*x + 1471.5)

# This function was just in the original wavelength solution jupyter notebook
def inverse_polymodel(wl, xaxis, wavelengths):
    return np.interp(wl, wavelengths, xaxis)

# Combine the above: Input a micrometer setting and a data array, finds bright lines and matches them to Rice's bright lines
def guess_pixel_wavelengths(data_array, micrometer_setting, reasonable_line_separation = 30, view_plots = True, **kwargs):
    # figure out what y-values we're looking at - just use the bright row. please input horizontally scattered lines until i make a trace program (NO PROMISES!!!)
    median_measurement = np.median(data_array)
    std_measurement = np.std(data_array)
    bright_rows = data_array[np.where(np.average(data_array, axis=1) > median_measurement)]
    y_value = np.floor(np.shape(bright_rows)[0]*.3).astype('int') # this is what should be replaced with a trace. right now it just picks something in the lower half of the brightest half rows.
    bright_strip = bright_rows[y_value,:].astype(np.int32)
    
    # mark bright lines x-values
    bright_strip_smoothed = savgol_filter(bright_strip, 5, 1)
    bright_line_inds = np.asarray(bright_strip_smoothed - median_measurement > 0.5 * std_measurement).nonzero()[0]
    
    bright_line_maxima_inds = bright_line_inds[find_maxima(bright_strip[bright_line_inds])]
    reasonably_separated_inds = np.where(np.diff(bright_line_maxima_inds, prepend=0, append=0) > reasonable_line_separation)
    
    guesspix = bright_line_maxima_inds[reasonably_separated_inds]
    
    if(view_plots):
        plt.imshow(data_array, cmap='gray', norm=LogNorm())
        for x in guesspix:
            plt.axvline(x, alpha=0.7, linestyle='--')
            plt.title("Guessed Pixel Locations")
        plt.show()
    
    # this is the hardest part of this - "guessing wavelengths" does the most work in this whole process, and humans are so much better at it than (my) code! 
    # idea: solve for expected wavelength at micrometer value. luckily i have data! then I just check if the guess is right
    guess_wl = check_lineups(data_array, guesspix, approximate_wavelength(micrometer_setting), **kwargs)
    
    # give back a tuple of findings
    return {'guesspix':guesspix, 'guess_wl':guess_wl, 'bright_row':bright_strip}

def solve_LHIRES_wavelength(calibration_data_folder:str, micrometer_setting:float, view_status=True, view_plots=True, polynomial_order = 2, relative_intensity_cutoff = 1000, guesspix = None, guess_wl = None):
    
    ## Step Zero: Load and Look at Data
    # potential addition - make this also take in flats, biases, and darks, subtract those out of calibration data
    
    # grab the files
    calibration_files = glob.glob(calibration_data_folder + "/*.fit")
    
    # initialize an array for the size of all the data
    first_file_header = fits.getheader(calibration_files[0])
    all_image_data = np.zeros((len(calibration_files), first_file_header['NAXIS2'], first_file_header['NAXIS1']))
    
    # get the data!
    for i, file in enumerate(calibration_files):
        all_image_data[i] = fits.getdata(file)
    
    # take the average of the data - I assume the calibration data is all the same exposure time and this is a reasonable mean to take
    averaged_calibration_data = np.mean(all_image_data,0)
    
    # look at data
    if(view_plots):
        plt.imshow(averaged_calibration_data, cmap='gray')
        plt.title("Averaged Calibration Data")
        plt.show()
    
    ## Step One: Get Guesses - 
    # we only need to do this if we don't already have guesses!
    if(guess_wl is None or guesspix is None):
        guesses = guess_pixel_wavelengths(averaged_calibration_data, micrometer_setting)
        guesspix = guesses['guesspix']
        guess_wl = guesses['guess_wl']
        ne_spectrum = guesses['bright_row']
    else:
        ne_spectrum = averaged_calibration_data[int(averaged_calibration_data.shape[0]/2)]
    xaxis = np.arange(averaged_calibration_data.shape[1])

    if(np.size(guesspix) <= 3):
        warnings.warn('WARNING: Only three lines detected. Wavelength guesses may be inaccurate.')

    
    ## Step Two: Fix Initial Guesses
    linfitter = LinearLSQFitter()
    wlmodel = Linear1D()
    linfit_wlmodel = linfitter(model=wlmodel, x=guesspix, y=guess_wl)
    wavelengths = linfit_wlmodel(xaxis) * u.AA
    if(view_status):
        print('guesspix:' + str(guesspix))
        print('guess_wl:' + str(guess_wl))
        print("Initial Guesses Linear Fit: " + str(linfit_wlmodel))
    
    ## Step Three: Improve Guesses
    npixels=15
    improved_xval_guesses = [np.average(xaxis[g-npixels:g+npixels],
                                    weights=ne_spectrum[g-npixels:g+npixels] - np.median(ne_spectrum))
                         for g in guesspix]
    linfit_wlmodel = linfitter(model=wlmodel, x=improved_xval_guesses, y=guess_wl)
    wavelengths = linfit_wlmodel(xaxis) * u.AA
    if(view_status):    
        print("Improved Guesses (Linear Model): " + str(improved_xval_guesses))
    
    ## Step Four: Plot Fit and Residual
    if(view_plots):
        plt.plot(guesspix, guess_wl, 'o')
        plt.plot(xaxis, wavelengths, '-')
        plt.plot(improved_xval_guesses, guess_wl, 's', zorder=-5)
        plt.title('Linear Fit')
        plt.show()
        plt.figure()
        plt.plot(improved_xval_guesses, (guess_wl - linfit_wlmodel(improved_xval_guesses)), 'x')
        plt.title('Linear Fit Residuals')
        plt.show()
    
    ## Step Four Point Five: Refit with Higher Order
    polymodel = Polynomial1D(degree=polynomial_order)
    linfitter = LinearLSQFitter()
    fitted_polymodel = linfitter(polymodel, improved_xval_guesses, guess_wl)
    wavelengths = fitted_polymodel(xaxis) * u.AA
    if(view_plots):
        plt.plot(guesspix, guess_wl, 'o')
        plt.plot(xaxis, fitted_polymodel(xaxis), '-')
        plt.plot(improved_xval_guesses, guess_wl, 's', zorder=-5)
        plt.title('Higher Order Fit')
        plt.show()
        plt.figure()
        plt.plot(improved_xval_guesses, (guess_wl - fitted_polymodel(improved_xval_guesses)), 'x')
        plt.title('Higher Order Residuals')
        plt.show()
    if(view_status):    
        print("Higher order polynomial: " + str(fitted_polymodel))
    
    ## Step Five: Query Line Catalog
    minwave = wavelengths.min()
    maxwave = wavelengths.max()
    neon_lines = Nist.query(minwav=minwave,
                        maxwav=maxwave,
                        wavelength_type='vac+air',
                        linename='Ne I')
    
    if(view_plots):
        plt.plot(wavelengths.value, ne_spectrum)
        plt.vlines(neon_lines['Observed'].value, 0, 62500, 'k', alpha=0.25)
        #plt.xlim(minwave.value,maxwave.value)
        plt.title('Neon Lines')
        plt.show()
    
    ## Step Six: Downselect Bright Lines
    # these lines downselect from the table to keep only those that have usable "Relative Intensity" measurements
    # first, we get rid of those whose 'Rel.' column is masked out or is an asterisk
    ne_keep = (neon_lines['Rel.'].astype('str') != "*")
    ne_wl_tbl = neon_lines['Observed'][ne_keep]
    # then, we collect the 'Rel.' values and convert them from strings to floats
    ne_rel_tbl = np.array([float(x) for x in neon_lines['Rel.'][ne_keep]])
    ne_rel_intens = ne_rel_tbl / ne_rel_tbl.max() * ne_spectrum.max()
    # we normalize the relative intensities to match the intensity of the spectrum so we can see both on the same plot
    # since they're just relative intensities, their amplitudes are arbitrary anyway
    ne_keep_final = ne_rel_intens > relative_intensity_cutoff
    if(view_plots):
        plt.plot(wavelengths, ne_spectrum)
        plt.plot(ne_wl_tbl, ne_rel_intens, 'x')
        plt.plot(ne_wl_tbl[ne_keep_final], ne_rel_intens[ne_keep_final], 'x')
        plt.title('Relative Intensity Wavelengths')
        plt.show()
    
    ## Step Seven: Convert Bright Lines to Pixel Predictions
    # select down to just those lines we want to keep
    ne_wl_final = ne_wl_tbl[ne_keep_final]
    # accounts for if the data is decreasing (np.interp doesn't like that) tcwpud = "this_camera_was_placed_upside_down"
    tcwpud = np.sign(fitted_polymodel(1) - fitted_polymodel(0))
    # linear isn't good enough.  ne_pixel_vals = linfit_wlmodel.inverse(ne_wl_final)
    ne_pixel_vals = inverse_polymodel(tcwpud*ne_wl_final, xaxis=xaxis, wavelengths=tcwpud*wavelengths)
    ## Step Eight: Measure Bright Line Pixel Locations
    # this ugly exception block tries to catch any coincidental weights sum to 0. please tell me if you know a better way to do this -p 
    try:
        npixels = 10
        improved_xval_guesses_ne = [np.average(xaxis[g-npixels:g+npixels],
                                    weights=ne_spectrum[g-npixels:g+npixels] - np.median(ne_spectrum))
                                    for g in map(int, ne_pixel_vals)]
    except ZeroDivisionError:
        try:
            npixels = 9
            improved_xval_guesses_ne = [np.average(xaxis[g-npixels:g+npixels],
                                        weights=ne_spectrum[g-npixels:g+npixels] - np.median(ne_spectrum))
                                        for g in map(int, ne_pixel_vals)]
        except ZeroDivisionError:
            try:
                npixels = 8
                improved_xval_guesses_ne = [np.average(xaxis[g-npixels:g+npixels],
                                            weights=ne_spectrum[g-npixels:g+npixels] - np.median(ne_spectrum))
                                            for g in map(int, ne_pixel_vals)]
            except ZeroDivisionError:
                npixels = 14
                improved_xval_guesses_ne = [np.average(xaxis[g-npixels:g+npixels],
                                            weights=ne_spectrum[g-npixels:g+npixels] - np.median(ne_spectrum))
                                            for g in map(int, ne_pixel_vals)]                
    if(view_status):        
        print("Bright line pixel locations: " + str(improved_xval_guesses_ne))
    
    ## Step Nine: Refit with Line List
    refitted_polymodel = linfitter(polymodel, improved_xval_guesses_ne, ne_wl_final)
    wavelengths_refit = refitted_polymodel(xaxis) * u.AA
    if(view_plots):
        plt.plot(improved_xval_guesses_ne, ne_wl_final, '^', label='Neon')
        plt.plot(xaxis, wavelengths_refit, zorder=-5)
        plt.plot(xaxis, linfit_wlmodel(xaxis), zorder=-5)
        plt.plot(xaxis, fitted_polymodel(xaxis), zorder=-5)
        plt.legend(loc='best')
        plt.xlabel("Pixel Coordinate")
        plt.ylabel("Wavelength (Angstroms)");
        plt.title("Final Model")
        plt.show()
        
        plt.plot(improved_xval_guesses_ne, ne_wl_final - linfit_wlmodel(improved_xval_guesses_ne), '.', label='Linear')
        plt.plot(improved_xval_guesses_ne, ne_wl_final - fitted_polymodel(improved_xval_guesses_ne), '+', label='Higher Order')
        plt.plot(improved_xval_guesses_ne, ne_wl_final - refitted_polymodel(improved_xval_guesses_ne), 'x', label='Refitted')
        plt.legend(loc='lower right')
        plt.xlabel("Pixel Coordinate")
        plt.ylabel("Wavelength residual (data minus model; nm)")
        plt.title("Final Residuals")
        plt.show()
    
    ## Step Ten: Plot and Return Final Wavelength Solution
    if(view_plots):
        plt.plot(wavelengths_refit, ne_spectrum)
        pl.vlines(neon_lines['Observed'][ne_keep_final], 0, 1000 + np.max(ne_spectrum), 'k', alpha=0.25, linestyle='--')
        for wl in neon_lines['Observed'][ne_keep_final]:
            plt.text(wl, 1000 + np.median(ne_spectrum), str(wl), rotation=90, ha='right', va='top')
        plt.xlabel("Air Wavelength [Angstroms]");
    if(view_status):
        print("Wavelengths calibrated!")
    return(wavelengths_refit)