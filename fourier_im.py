import numpy as np
import scipy.ndimage
from scipy import interpolate, stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import warnings
import data_readin as dim

def spatialft(image, cosine_window=True, rmdc=True):
    """Take the fourier transform of an image (or flow field).
    shift the quadrants around so that low spatial frequencies are in
    the center of the 2D fourier transformed image"""

    #raised cosyne window on image to avoid border artifacts
    (dim1,dim2) = np.shape(image)
    if(cosine_window):
        cosfilter = np.tile(np.hanning(dim2), (dim1,1))*(np.tile(np.hanning(dim1), (dim2,1)).T)
        image = image * cosfilter
    
    # remove DC component
    if(rmdc):
        image =  image - np.mean(image)
    
    ps = np.abs(np.fft.fftshift(np.fft.fft2(image)))**2
    fqs = [np.fft.fftshift(np.fft.fftfreq(np.shape(image)[0])),
           np.fft.fftshift(np.fft.fftfreq(np.shape(image)[1]))]
    return(ps, fqs)


def get_logspace_contours(raw_field, c_levels=None):
    """Get the contour locations by percentage of max power in logspace"""
    
    # settings for contour plots
    if not c_levels:
        c_levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9]) # energy percentages to contour

    log_field = np.log10(raw_field)
    t_contours = c_levels * log_field.max()
    linear_t_contours = 10**t_contours
    
    return(c_levels, linear_t_contours)

def interp_get_contours(ps):
    interp_val = 5
    interp = scipy.ndimage.filters.gaussian_filter(ps, interp_val)
    clevels, contours = get_logspace_contours(interp)
    return(interp, contours, clevels)

def contour_plotter(ps, fqs, bar=True, ticks=True, title='Power Contour'):
    """Function to make contour plots"""
    
    c_cmap = 'brg' # colormap for contour plots
    
    interp, contours, clevels = interp_get_contours(ps)
    ratio = len(fqs[0])/len(fqs[1])
    
    plt.figure(figsize=(20, 20*ratio))
    
    cont_pl = plt.pcolormesh(fqs[1],
                             fqs[0],
                             interp,
                             cmap="gray",
                             norm=mpl.colors.LogNorm())#,extent=extents)
    
    if(bar):
        cont_pl = plt.colorbar()
        
    if not(ticks):
        plt.axis('off')
    
    pl = plt.contour(fqs[1],
                     fqs[0],
                     interp,
                     levels=contours,
                     cmap=c_cmap,
                     norm=mpl.colors.LogNorm())
    
    for i in range(len(clevels)):
        pl.collections[i].set_label(f'{clevels[i]*100:0.0f}%')

    plt.legend(loc='upper left')
    
    plt.gca().set_aspect(ratio)
    
    plt.title(title)
    return(pl)

def get_mean_ps(filelist, cosine_window, imtype, ims_raw=False):
    '''
    Calculate the mean power spectrum of a natural image dataset (each iamge as its own file)
    Params:
        filelist: list of filenames corresponding to natural images (full path)
        OR a list of images
        cosine_window: apply a hanning window before calculating spectrum?
        ims_raw (bool): Flag if filelist is instead the files themselves (already read in)
    Returns:
        mean_power_spec: A single 2d power spectrum represeing the mean power spectrum.
        fq_power_spec: Spectrum of frequencies
    '''
    
    try:
        mean_power_spec = pickle.load(open(f'./output/mean_power_coswin_{cosine_window}.pkl'))
        mean_fq_spec = pickle.load(open(f'./output/mean_fqspec_coswin_{cosine_window}.pkl'))

    except:
        print(f'Couldn\'t find mean power spectrum: Calculting now from {len(filelist)}  frames...')
        #book keeping
        if(ims_raw):
            frameshape = np.shape(spatialft(filelist[0],
                                            cosine_window=cosine_window)[0])
        else:
            frameshape = np.shape(spatialft(dim.readin_img(filelist[0], imtype),
                                            cosine_window=cosine_window)[0])
        mean_power_spec = np.zeros((frameshape)).astype('float32')

        #loop through video and get chunks of chunklen
        i = 0 #frame num sampling from
        for i in range(len(filelist)):
            if(ims_raw):
                im = filelist[i]
            else:
                im = dim.readin_img(filelist[i], imtype)
            p, fq_spec = spatialft(im, cosine_window=cosine_window)
            mean_power_spec += p

        mean_power_spec /= (len(filelist))
        
        print('Saving...',end='')
        pickle.dump(mean_power_spec, open(f'./output/mean_power_coswin_{cosine_window}.pkl', 'wb'))
        pickle.dump(fq_spec, open(f'./output/mean_fqspec_coswin_{cosine_window}.pkl', 'wb'))
        
    return(mean_power_spec, fq_spec)


def get_mean_ps_flow(f, data,frameshape, flow_type, data_subset, cosine_window):
    '''
    Load in or calcuate the mean flow power spectrum from the dataset
    Params:
        f: flow files
        data: dataset of flow data
        chunklen: how large of chunks can we read in at a time? *I think we may not need this...*
        frameshape: width and height of flow fields
        flow_type: 'raw' or 'retinal'
        data_subset: 'all' or 'walking'
        cosine_window: Mutiply by a cosine window (to avoid edge artifacts)?
    Returns:
        mean_flow: one flow frame representign the mean of data.    
    '''
    
    try:
        mean_power_spec_u = pickle.load(open(f'./output/matthis/flow/mean_flow_power_{flow_type}_{data_subset}_u_coswin_{cosine_window}.pkl','rb'))
        mean_power_spec_v = pickle.load(open(f'./output/matthis/flow/mean_flow_power_{flow_type}_{data_subset}_v_coswin_{cosine_window}.pkl','rb'))
        fq_spec_u = pickle.load(open(f'./output/matthis/flow/mean_flow_spec_{flow_type}_{data_subset}_u_coswin_{cosine_window}.pkl','rb'))
        fq_spec_v = pickle.load(open(f'./output/matthis/flow/mean_flow_spec_{flow_type}_{data_subset}_v_coswin_{cosine_window}.pkl','rb'))
        print(f'Found Mean Flow Power Already Calculated for {flow_type}! Done!')
    except:
        print(f'Couldn\'t find mean flow frame for flow type \'{flow_type}\',\'{data_subset}\': Calculting now from {max(data) - min(data)} flow frames...')
        #book keeping
        mean_power_spec_u = np.zeros((frameshape)).astype('float32')
        mean_power_spec_v = np.zeros((frameshape)).astype('float32')

        #loop through video and get chunks of chunklen
        i = 0 #frame num sampling from
        for i in range(min(data), max(data)):
            flow = np.nan_to_num(f[str(i)])
            pu, fq_spec_u = spatialft(flow[:,:,0], cosine_window=cosine_window)
            pv, fq_spec_v = spatialft(flow[:,:,1], cosine_window=cosine_window)
            mean_power_spec_u += pu
            mean_power_spec_v += pv

        mean_power_spec_u /= (max(data) - min(data))
        mean_power_spec_v /= (max(data) - min(data))
        
        print('Saving...',end='')
        pickle.dump(mean_power_spec_u, open(f'./output/matthis/flow/mean_flow_power_{flow_type}_{data_subset}_u_coswin_{cosine_window}.pkl', 'wb'))       
        pickle.dump(mean_power_spec_v, open(f'./output/matthis/flow/mean_flow_power_{flow_type}_{data_subset}_v_coswin_{cosine_window}.pkl', 'wb'))
        pickle.dump(fq_spec_u, open(f'./output/matthis/flow/mean_flow_spec_{flow_type}_{data_subset}_u_coswin_{cosine_window}.pkl', 'wb'))       
        pickle.dump(fq_spec_v, open(f'./output/matthis/flow/mean_flow_spec_{flow_type}_{data_subset}_v_coswin_{cosine_window}.pkl', 'wb'))
        print('Done!')
    return(mean_power_spec_u, mean_power_spec_v, fq_spec_u, fq_spec_v)


def azimuthalAverage(image, nyquist, center=None, bin_in_log=False):
    """      
    Calculate the azimuthally averaged radial profile. (Intended for 2d Power Spectra)
    image - The 2D image (2d power spectrum)
    nyquist - max frequency value (assume same for x and y)
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)
    num_bins = np.min(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    #ASSUME HERE THAT MAX FREQUENCY IS EQUAL ON BOTH AXES & GRANULARITY VARIES ***
    normalized = ((x-center[0])/np.max(x),(y-center[1])/np.max(y))
    r = np.hypot(normalized[0], normalized[1])
    #don't calculate corners
    keep_circle = np.where(r<=np.max(y))
    r = r[keep_circle]
    image = image[keep_circle]

    # number of bins should be equivalent to the number of bins along the shortest axis of the image.
    if(bin_in_log):
        bin_edges = np.histogram_bin_edges(np.log(r), num_bins)
        bin_edges = np.exp(bin_edges)
    else:
        bin_edges = np.histogram_bin_edges(r,num_bins)
    
    r_binned = np.digitize(r, bin_edges)
    binmean = np.zeros(num_bins)
    for i in range(num_bins):
        binmean[i] = np.mean(image[np.where(r_binned==i+1)])
    bin_centers = bin_edges[:-1] + ((bin_edges[1]-bin_edges[0])/2)
    bin_centers = bin_centers/np.max(bin_centers)*nyquist

    return(binmean, bin_centers)


def filt_gauss_step(f, fc):
    '''
    Return a 2d filter where fc is the cuttoff frequency in cycles per something.
    Input:
        f 
    '''
    #cuttoff amplitude for frequencies above cuttoff based on gernalized gaussian
    beta=2 #gaussian for now (beta=1 for Laplacian)
    alpha=0.05
    #calculate half width at half max: relationship between cuttoff fq and frequency where gauss taper is centerd.
    hwhm = alpha * (np.log(2))**(1./beta)
    #center of gaussian is cuttoff minus half witch half max
    fd = fc - hwhm
    #check if top of gaussian is negative.
    if fd < 0:
        warnings.warn('Taper Top Negative - Won\'t reach full contrast.')
        warn_flag = True
    #generic gauusian scaing function
    filt = np.exp(-1*(np.abs(f-fd)/alpha)**beta)
    filt /= np.max(filt)
    filt[f<fd] = 1.
    return(filt, warn_flag)

def filt_cosine_step(f, fc):
    #cosine step taper, full power at fd, zero power at 2*fd
    warn_flag = False
    #calc fd (taper start) and fz (taper end)
    fd = fc*np.pi/(np.arccos(np.sqrt(2)-1)+np.pi)
    #end taper at 2*fd for a power scale
    fz = 2*fd
    #can now define function
    filt = 0.5*(1+np.cos(np.pi*(f-fd)/(fz-fd+0.01)))

    filt[f < fd] = 1
    filt[f > fz] = 0

    if fz > np.max(f):
        warn_flag = True
        warnings.warn('Zero point is beyond Nyquist - Nothing is completally cuttoff.')
    return(filt, warn_flag)

def filt_gauss_step(f, fc):
    #cuttoff amplitude for frequencies above cuttoff based on gernalized gaussian
    warn_flag=False
    beta=2 #gaussian for now (beta=1 for Laplacian)
    alpha=0.05
    #calculate half width at half max: relationship between cuttoff fq and frequency where gauss taper is centerd.
    hwhm = alpha * (np.log(2))**(1./beta)
    #center of gaussian is cuttoff minus half witch half max
    fd = fc - hwhm
    #check if top of gaussian is negative.
    if fd < 0:
        warnings.warn('Taper Top Negative - Won\'t reach full contrast.')
        warn_flag = True
    #generic gauusian scaing function
    filt = np.exp(-1*(np.abs(f-fd)/alpha)**beta)
    filt /= np.max(filt)
    filt[f<fd] = 1.
    return(filt, warn_flag)


def fft_lowpass(img_in, cpd_cutoff, stim_cpd, filt_name='sharp', rescale=True):
    '''
    Lowpass filter an image at a given cpd cuttoff using Fourier representation, for a given cpd of the img stimulus
    
    Args:
        img_in (2d numpy array):   stimluius img
        cpd_cuttoff (float):  maximum CPD value present in output img
        stim_cpd (float):    CPD of stimulus (should be larger than cpd_cuttoff)
        filt_name (str):        define the type of filtering desired (sharp, cosine_step, gauss_step, gauss_taper)
        
    Returns:
        stim (2d numpy array):   stimlulus image fourier filtered and no frequencies higher than cpd_cuttoff
        mag (2d numpy array):   magnitude of stimulus
        phase (2d numpy array): global phase angle of stimulus
        filt (2d numpy array): fiter used to create filtered img
        warn_flag (bool):   flag  if we had a warning in generating image
    '''
    
    # warn flag is false by default
    warn_flag = False
    #make sure parameters make sense
    if cpd_cutoff > stim_cpd:
        warnings.warn(f'Cutoff CPD ({stim_cpd}) is higher than stimulus CPD ({stim_cpd})')
        warn_flag=True

    #find ratio of cuttoff to max cpd so we know where to stop in fourier space
    fft_diameter_fc = cpd_cutoff/(stim_cpd)
    fft = np.fft.fftshift(np.fft.fft2(img_in))
    mag = np.abs(fft)
    yy, xx = np.meshgrid(np.linspace(-1, 1, img_in.shape[1]),
                         np.linspace(-1, 1, img_in.shape[0])) #THIS IS NOT VALID IF STIM IS NOT SQUARE111
    fft_diameters = np.sqrt(xx**2 + yy**2)
    
    #calculate filter
    if(filt_name=='sharp'):
    # Anything greater than cpd_cutoff set to 0 in the mag
        filt = (fft_diameters <= fft_diameter_fc)
        
    elif(filt_name=='gauss_taper'):
        #calculate sigma needed for HWHM value to be at fc
        #sigma = fft_diameter_fc / (np.sqrt(-np.log(np.sqrt(0.5))))
        #rescale to 100 so we don't get numerical instability
        sigma = fft_diameter_fc*100000 / np.sqrt(-2*np.log(np.sqrt(0.5)))
        filt = np.exp(-1.5*((fft_diameters*100000)/(sigma))**2)
    elif(filt_name=='cosine_step'):
        filt, warn_flag = filt_cosine_step(fft_diameters, fft_diameter_fc)
    elif(filt_name=='gauss_step'):
        filt, warn_flag = filt_gauss_step(fft_diameters, fft_diameter_fc)
    
    else:
        raise ValueError(f'{filt_name} is an unknown filtering type! Currently Supported Decompositions are \'fourier_sharp\', \'fourier_gauss\' and \'wavelet\'.  Returning original Signal.')
        filt = 1
        
    #multiply by arbitrary fourier filter
    mag = np.multiply(mag,filt)
    phase = np.angle(fft)
    
    # Reconstruct the image using the inverse fourier transform
    ifft = np.fft.fftshift(mag * np.exp(phase * 1.0j))
    img_ifft = np.fft.ifft2(ifft).real
    
    # Rescale to [0,255]
    if(rescale):
        img_ifft = imtools.rescale_255(img_ifft)
    
    return img_ifft, mag, phase, filt, warn_flag