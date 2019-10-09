import numpy as np

def crop_ims(ims, edgesize=None, area='middle'):
    '''
    funtion to resize image to smallest of all images, or edgesize is supplied.
    
    Params:
        ims (list of 2d numpy arrays): set of images to crop to the same shape
        edgesize (optional, tuple int): size of image desired
        area (string): Crop from the top, middle, or bottom?
    '''

    if(edgesize==None):
        dims = np.array([(np.shape(im)) for im in ims])
        crop_h = np.min(dims[:,0])
        crop_w = np.min(dims[:,1])
    
    h_crop_h = np.int(crop_h/2)
    h_crop_w = np.int(crop_w/2)
    # newims = np.ones((np.shape(ims)[0],crop_h,crop_w,3))
    newims = []
    for i, im in enumerate(ims):
        # print(np.shape(im))
        im_h = np.shape(im)[0]
        im_w = np.shape(im)[1]
        h_im_h = np.int(im_h / 2) # half image height
        h_im_w = np.int(im_w / 2) # half image width
        
        #check to see if we have large enough images to crop
        if(im_w < 2*crop_w):
            lr_replicates=False
        
        if(area=='middle'):
            addon = [d%2for d in  np.shape(ims[0])]
            newim = im[h_im_h - h_crop_h:h_im_h + h_crop_h+addon[0], # crop height
                       h_im_w - h_crop_w:h_im_w + h_crop_w+addon[0]] # crop width
            newims.append(newim)

        elif(area=='top'):
            newim = im[:crop_h, # crop height
                       h_im_w - h_crop_w-1:h_im_w + h_crop_w] # crop width
            newims.append(newim)

        elif(area=='bottom'):
            newim = im[im_h - crop_h:im_h, # crop height
                       h_im_w - h_crop_w-1:h_im_w + h_crop_w] # crop width
            newims.append(newim)

        else:
            raise ValueError('area not well specified!')
        
    return(newims)