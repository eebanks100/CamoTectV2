#doby added code commenting on logic
from PIL import Image, ImageDraw        #import image library 
import os, argparse                     #util/system libraries 
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--advStats', action='store_true', help='Shows total number and percentages of true positive, false positive, true negative, false negative (negative images must be named with keywords "no" or "neg")')
parser.add_argument('--waste', type=float, default=0.7, help='Set the waste ratio cutoff for head classification.')

opt = parser.parse_args()

def isWhite(x, y):		#checks if a pixel is white (255 grescale value)	(x and y are coords of the pixel)
    if 0 <= x < maxX and 0 <= y < maxY:     #if it is in the bounds of the image (to avoid out of bounds error)
        r, g, b = image.getpixel((x, y))		#get the red value of the pixel  (for our purposes all the RBG values are the same)
        if r == 255:				#if it is 255
            return True
    return False

def isWhitish(x, y):	#checks if a pixel is whitish (200 greyscale value)
    if 0 <= x < maxX and 0 <= y < maxY:         #if it is in the bounds of the image (to avoid out of bounds error)
        r, g, b = image.getpixel((x, y))		#get the red value of the pixel
        if r >= 200:				#if red val of pixel is over 200
            return True
    return False

def isBlack(x, y):			#checked if a pixel is black (0 greyscale value)
    if 0 <= x < maxX and 0 <= y < maxY:     #if in bounds of the image
        r, g, b = image.getpixel((x, y))		
        if r == 0:				
            return True
        return False

def findWhite(x, y, x0, y0, x1, y1):
    while x < maxX and y < maxY and not isWhite(x, y):	    #scans left to right, top to bottom until it finds a white pixel
            x += 4					#going 4 pixels at a time by x just to speed it up (white head clusters generally > 4 pixels, if it isnt > 4 pixels the waste ratio will be low anyways making it not be classified as a head)
            if (x >= maxX):		#if you reach the right side of the border jump back to the start and move down
                x = 0			#back to left side
                y += 4          #go down
    if(x < maxX and y < maxY):      #if in bounds
        max = len(x0)       #how many bounding boxes there are to check
        for i in range(0, max):         #check to see if the detected white pixel is in a previously detected bounding box
              if (x >= x0[i] and x <= x1[i] and y >= y0[i] and y <= y1[i]):     #if it is
                    return findWhite(x1[i], y, x0, y0, x1, y1)                  #resume scanning on the right side of the bounding box(collided from the left)
        return x, y         #return coords of not previously detected white pixel
    return -1, -1           #return -1 -1 if there are no more undetected white pixels
    
def findBounds(x, y):       #find the bounding box of the object once u detect a white pixel
    x0b = x
    y0b = y
    x1b = x
    y1b = y

    def checkLR(x, y0b, y1b, inc):          #method checks left/right bounds
        flag = False                    #flag to track if no new pixels are added
        while not flag:
            flag = True     #set flag to true, if bounding box is extended by 1 pixel flag gets set to false, to repeat for pixel after that, etc.
            max = y1b - y0b + 1         #height of the up/down edge we are checking along in the left/right dirs
            for i in range(0, max):
                y0bi = y0b + i
                if 0 <= x < maxX and 0 <= y0bi < maxY and not isBlack(x, y0bi):	#if pixel checking is in bounds and not black extend bounds by 1 pixel
                    x += inc			
                    i = max     #end the for loop -> start checking edge again in +1 l/r pixel direction
                    flag = False        #set flag to false to continue the cycle
        return x
    
    def checkUD(y, x0b, x1b, inc):          #checkLR except checking up/down bounds along left/right edge
        flag = False
        while not flag:
            flag = True
            max = x1b - x0b + 1
            for i in range(0, max):
                x0bi = x0b + i
                if 0 <= y < maxY and 0 <= x0bi < maxX and not isBlack(x0b + i, y):	
                    y += inc		
                    i = max
                    flag = False
        return y
    
    flag = True
    while flag:
        x1b2 = checkLR(x1b, y0b, y1b, 1)            #check bounds in clockwise square
        y1b2 = checkUD(y1b, x0b, x1b2, 1)
        x0b2 = checkLR(x0b, y0b, y1b2, -1)
        y0b2 = checkUD(y0b, x0b2, x1b2, -1)

        if(x0b == x0b2 and x1b == x1b2 and y1b == y1b2 and y0b == y0b2):            #if bounds were not updated in a cycle of checking
              flag = False          #break out of the loop
        
        x0b = x0b2          #update vars to detected bounds
        y0b = y0b2
        x1b = x1b2
        y1b = y1b2

    updatedb = True             #bounds were updated (not needed(?) artifact of when i was looking at making the method recursive instead of flag while loop)
    return updatedb, x0b, y0b, x1b, y1b         #return updated status and bounds



# moreStats = 0
# if(moreStats == 0):
#     testFold = "./sinet_output/"
#     saveFold = "./sinetProc_output/res_output/"
#     frameFold = "./Dataset/Test/Imgs/"
#     fSaveFold = "./sinetProc_output/imgs_output/"
# else:
#     testFold = "./sinet_output/"
#     saveFold = "./sinetProc_output/res_output/"
#     frameFold = "./Dataset/Test/Imgs/"
#     fSaveFold = "./sinetProc_output/imgs_output/"

testFold = "./sinet_output/"                            #file paths
saveFold = "./sinetProc_output/res_output/"
frameFold = "./Dataset/Test/Imgs/"
fSaveFold = "./sinetProc_output/imgs_output/"
logPos = "./sinetProc_output/PosClassification.txt"
logNeg = "./sinetProc_output/NegClassification.txt"

abc = os.listdir(testFold)                  #sinet outputs read/write
abcF = os.listdir(frameFold)                #frames (write)

# Elisha -  create the necessary output save folders if they dont exist
if not os.path.exists(saveFold):
    os.makedirs(saveFold, exist_ok=True)
    print(f"New folder created at {saveFold}")
else:
    shutil.rmtree(saveFold, ignore_errors=True)
    os.makedirs(saveFold, exist_ok=True)
if not os.path.exists(fSaveFold):
    os.makedirs(fSaveFold, exist_ok=True)
    print(f"New folder created at {fSaveFold}")
else:
    shutil.rmtree(fSaveFold, ignore_errors=True)
    os.makedirs(fSaveFold, exist_ok=True)
if os.path.isfile(logPos):
    os.remove(logPos)
if os.path.isfile(logNeg):
    os.remove(logNeg)

heads = []          #array of detected heads
positiveImage = 0       # num of pos images
negativeImage = 0
falsePositive = False
for q in range(0, len(abc)):                #for every image in the dir
    image = Image.open(testFold + abc[q])           #open image, init drawing
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)

    frame = Image.open(frameFold + abcF[q])         #open image, init drawing (frame)
    frame = frame.convert('RGB')
    drawF = ImageDraw.Draw(frame)

    maxX, maxY = image.size                 #set image dimensions (assuming sinet output and frame its based on are same dimensions)

    # Elisha - log text file for which images were identified as pos or neg images
    logP = open(logPos, 'a')
    logN =  open(logNeg, 'a')

    x0 = []             #arrays of the bound box 2 opposite corners (all thats needed since its a rectangle)
    x1 = []
    y0 = []
    y1 = []

    cy = 0              #current x/y -> start in top left corner since scanning right/down
    cx = 0
    x1t = 0             

    while cy <= maxY:       #while still in image bounds(y)
        cx = x1t
        cx, cy = findWhite(cx, cy, x0, y0, x1, y1)          #set current x/y to detected white pixel
        if(cx == -1):           #if no white pixel detected
            break
        print(str(cx) + "," + str(cy))

        upd, x0t, y0t, x1t, y1t = findBounds(cx, cy)        #get bounds of detected pixel
        x0.append(x0t)          #append them to the arrays ---x 0 temp
        y0.append(y0t)          #y 1 temp
        x1.append(x1t)          #x 1 temp
        y1.append(y1t)          #y 1 temp           temp since just using to grab values of bounding box and immediantly storing into arrays 

    print("-------------")

    waste = []          #array of waste ratios
    for k in range(0, len(x0)):     #for each detected bounds (all 4 arrays same length)
        black = 0           #counter of black pixels (waste, bad, not head) and white pixels
        white = 0              #counter of white pixels (signal, good, head)
        for i in range(0, x1[k] - x0[k]):       #for the range of x pixels
            for j in range(0, y1[k] - y0[k]):         #for the range of y pixels            (x * y = area of bound box, checking each pixel in box)
                r, g, b = image.getpixel((x0[k] + i, y0[k] + j))        #get val of pixel
                if r > 0:           # if it aint pure black(we dont want pure black pixels in count, if we included them it wouldnt be noise/waste filter as much as noise/waste + how circular filter)
                        black += 1      
                        if r > 200:
                            white += 1              # so if you have a 250 pixel both black and white get incd by 1
                            
                            for l in range(-1, 2):      #checks 3x3 around pixel(including pixel itself which is inefficient)
                                 for m in range(-1, 2):
                                      x = x0[k] + i
                                      y = y0[k] + j
                                      if(not isWhitish(x + l, y + m)):          #if its not white we will mark the pixel as an edge pixel      
                                           image.putpixel((x0[k] + i, y0[k] + j), (255, 0, 0))      #make it red
                                           frame.putpixel((x0[k] + i, y0[k] + j), (255, 0, 0))
                                           continue         #continue out of the for loop- not sure if this skips out of just inner or both but skipping out both would be good
                            
                            """                                                         #garbage
                            if(not isWhite(x0[k] + i - 1, y0[k] + j)):
                                 image.putpixel((x0[k] + i, y0[k] + j), (255, 0, 0))
                            elif(not isWhite(x0[k] + i, y0[k] + j)):
                                 image.putpixel((x0[k] + i, y0[k] + j), (255, 0, 0))
                            elif(not isWhite(x0[k] + i, y0[k] + j + 1)):
                                 image.putpixel((x0[k] + i, y0[k] + j), (255, 0, 0))
                            elif(not isWhite(x0[k] + i, y0[k] + j - 1)):
                                 image.putpixel((x0[k] + i, y0[k] + j), (255, 0, 0))
                            elif(not isWhite(x0[k] + i - 1, y0[k] + j - 1)):
                                 image.putpixel((x0[k] + i, y0[k] + j), (255, 0, 0))
                            elif(not isWhite(x0[k] + i + 1, y0[k] + j - 1)):
                                 image.putpixel((x0[k] + i, y0[k] + j), (255, 0, 0))
                            elif(not isWhite(x0[k] + i + 1, y0[k] + j + 1)):
                                 image.putpixel((x0[k] + i, y0[k] + j), (255, 0, 0))
                            elif(not isWhite(x0[k] + i - 1, y0[k] + j + 1)):
                                 image.putpixel((x0[k] + i, y0[k] + j), (255, 0, 0))
                            """
                        #elif r > 100:
                            #image.putpixel((x0[k] + i, y0[k] + j), (255, 0, 0))

        w = white/black     #the waste ratio, closer to 1 implies less waste -> higher signal to noise ratio
        waste.append(w)         #add it to the list
        print(w)                #print it to console
    
    #wasteMax = 0       #this was to factor in relative noise of heads and objects per image but removed -> similar logic might be usefull to better classify dir of images that vary in total noise
    #for i in range(0, len(waste)):
    #     wasteMax = max(wasteMax, waste[i])

    print(str(q) + "   XXXXXXXXXXXXX")      #tracking which waste val we are on
    hasHead = False
    heads.append(0)         #add val to list of heads, start at 0 detected heads
    for i in range(0, len(x0)):     #for each bounding box/detected object
        print(str(x0[i]) + ", " + str(y0[i]))
        print(str(x1[i]) + ", " + str(y1[i]))
        print("#############")
        #if(waste[i] > max(0.75, wasteMax * 0.95)):             #0.65 to 0.75 100%
        if(waste[i] >= opt.waste):            #if its waste ratio is > 0.7 or selected amount
            heads[q] += 1                  #head detected
            hasHead = True
            draw.rectangle(					#draw the bounding box on the img
                    (x0[i], y0[i], x1[i], y1[i]),
                    fill=None,
                    outline=(0, 255, 0),					#red outline
                    width=1)
            drawF.rectangle(					#draw the bounding box on the img
                    (x0[i], y0[i], x1[i], y1[i]),
                    fill=None,
                    outline=(0, 255, 0),					#red outline
                    width=1)       

    if hasHead:
        positiveImage += 1
        logP.write('{}\n'.format(abc[q]))      # image.info
    else:
        negativeImage += 1
        logN.write('{}\n'.format(abc[q]))  

    image.save(saveFold + abc[q])
    frame.save(fSaveFold + abcF[q])
    hasHead = False
    falsePositive = False
    
# tp = 0
# fp = 0
# tn = 0
# fn = 0    

# Elisha - Old classification, for the else statement, the fn is supposed to be tn?. 
# if moreStats == 0:
#     for i in range(0, len(heads)):
#         if(heads[i] == 0):
#             tn += 1
#         else:
#             fp += 1     
# else:
#     for i in range(0, len(heads)):
#         if(heads[i] == 0):
#             fn += 1
#         elif (heads[i] == 1):
#             tp += 1
#         else:
#             tp += 1
#             fp += heads[i] - 1   

# Elisha - set moreStats to 0 for basic img classification, moreStats 1 gives true pos, false pos, true neg, and false neg stats based on the naming of the test dataset imgs
# moreStats = 0

# # Elisha - this is primarily for test dataset mixed with neg and pos frames, may not work with other datasets with different naming
# if moreStats == 1:
if opt.advStats:
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(0, len(heads)):
        if(heads[i] >= 1):
            tp += 1
            if (heads[i] >= 1 and (abc[i].__contains__('no') or abc[i].__contains__('neg'))):
                tp -= 1
                fp += 1
        elif (heads[i] == 0):
            tn += 1
            if (heads[i] == 0 and (not(abc[i].__contains__('no') or abc[i].__contains__('neg')))):
                tn -= 1
                fn += 1
        else:
             continue

    print("\n\n###########################################################")            #stat logics
    print("Detected as Positive images: " + str(positiveImage))
    print("Detected as Negative images: " + str(negativeImage))
    print("tp: " + str(tp))
    print("fp: " + str(fp))
    print("tn: " + str(tn))
    print("fn: " + str(fn))
    print("###########################################################")
    if(tp == 0):
        print("True positive: 0.00")
    else:
        print("True positive: " + str(tp/(tp + fn)))
    if(fp == 0):
        print("False positive: 0.00")
    else:
        print("False positive: " + str(fp / (fp + tn)))
    if(tn == 0):
        print("True negative: 0.00")
    else:
        print("True negative: " + str(tn / (tn + fp)))
    if(fn == 0):
        print("False negative: 0.00")
    else:
        print("False negative: " + str(fn / (tp + fn)))

    print("\nImage classification logs saved under the: ./" + str(saveFold.split("/")[1]) + " folder")
else:
    print("###########################################################")
    print("\n\nDetected as Positive images: " + str(positiveImage))
    print("Detected as Negative images: " + str(negativeImage))
    print("\nImage classification logs saved under the: ./" + str(saveFold.split("/")[1]) + " folder")

logP.write('\nNumber of images detected as positive images: {}'.format(positiveImage))
logN.write('\nNumber of images detected as negative images: {}'.format(negativeImage))
logP.close()
logN.close()