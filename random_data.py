import csv
import cv2
import numpy as np
import numpy.random
import sklearn
import os
import shutil

def flip(image, steering):
    r = np.random.randint(0,2)
    if (r==0):
        return np.fliplr(image),-steering
    else:
        return image,steering

def brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = 0.8 + 0.4*(2*np.random.uniform()-1.0)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def crop(image,steering=0.0,tx_lower=-20,tx_upper=20,ty_lower=-2,ty_upper=2,rand=True):
    # we will randomly crop subsections of the image and use them as our data set.
    # also the input to the network will need to be cropped, but of course not randomly and centered.
    shape = image.shape
    col_start,col_end =abs(tx_lower),shape[1]-tx_upper
    horizon=50
    bonnet=140
    if rand:
        tx= np.random.randint(tx_lower,tx_upper+1)
        ty= np.random.randint(ty_lower,ty_upper+1)
    else:
        tx,ty=0,0

    #    print('tx = ',tx,'ty = ',ty)
    random_crop = image[horizon+ty:bonnet+ty,col_start+tx:col_end+tx,:]
    image = cv2.resize(random_crop,(320,90),cv2.INTER_AREA)
    # the steering variable needs to be updated to counteract the shift
    if tx_lower != tx_upper:
        dsteering = -tx/(tx_upper-tx_lower)/3.0
    else:
        dsteering = 0
    steering += dsteering

    return image,steering

def shear(image,steering,shear_range):
    rows,cols,ch = image.shape
    dx = np.random.randint(-shear_range,shear_range+1)
    #    print('dx',dx)
    random_point = [cols/2+dx,rows/2]
    pts1 = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
    pts2 = np.float32([[0,rows],[cols,rows],random_point])
    dsteering = dx/(rows/2) * 360/(2*np.pi*25.0) / 6.0
    M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image,M,(cols,rows),borderMode=1)
    steering +=dsteering

    return image,steering

def getImage(line):
    correction = 1/20 * 360/( 2*np.pi) / 25.0
    which = np.random.randint(0,3)
    src_path=line[which]
    if (which==0):
        steering = float(line[3])
    elif (which==1):
        src_path=line[1]
        steering = float(line[3]) + correction
    else:
        steering = float(line[3]) - correction
    filename=src_path.split('/')[-1]
    cur_path='data/IMG/'+filename
    image=cv2.imread(cur_path)
    image,steering = shear(image,steering,shear_range=100)
    image,steering = crop(image,steering,tx_lower=-20,tx_upper=20,ty_lower=-10,ty_upper=10)
    image,steering = flip(image,steering)
    image = brightness(image)
    image,steering=flip(image,steering)
    return image,steering

def calculate_steering_keep_rate(lines):
    steerings = lines[:,3].astype(np.float)
    hist, bin_edges= np.histogram(steerings,bins=30)
    median=np.median(hist)
    # get the random keep % from the histogram
    keep_rate = []
    for i in range(len(hist)):
        if (hist[i]>median):
            keep_rate.append(median/hist[i])
        else:
            keep_rate.append(1)
    keep_rate = np.asarray(keep_rate)
    return bin_edges, keep_rate

def normalize_steering_lines(bin_edges, keep_rate, lines):
    lines_keep=[]
    for i in range(len(lines)):
        h = -1
        for j in range(len(bin_edges)):
            if (bin_edges[j]>= steerings[i]):
                h = j - 1
                break
        if (h!=-1 and np.random.randint(100000) < keep_rate[h] * 100000):
            lines_keep.append(lines[i])
    lines_keep=np.asarray(lines_keep)
    return lines_keep

def generateData(target, outdir):
    try:
        shutil.rmtree(outdir)
    except FileNotFoundError:
        print("no previous data dir")
    else:
        print("good")
    os.makedirs(outdir)
    os.mkdir(outdir+"/IMG")

    count = 0
    lines =[]
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        skip_first=False
        for line in reader:
            if (skip_first):
                lines.append(line)
            else:
                skip_first=True
    lines = np.asarray(lines)
    bin_edges, keep_rate = calculate_steering_keep_rate(lines)
    with open(outdir+'/driving_log.csv', 'w', newline='') as csvfile:
        csvfile=csv.writer(csvfile)
        while(count<target):
            lines_keep = normalize_steering_lines(bin_edges, keep_rate, lines)
            for line in lines_keep:
                count=count+1
                image,steering=getImage(line)
                filename=outdir+"/IMG/"+str(count)+".jpg"
                cv2.imwrite(filename,image)
                csvfile.writerow([filename, filename, filename, steering, 0, 0, 0])
                break

generateData(10000, "/opt/data")
