
## Creating labels for our dataset : 

## The array contains the indx of every cauliflowers we would like to pick 

## We only considers picture at the moment (no video frames)

mur_indx = [1,2,3,4,5,6,7,8,9,10,11,13,15,16,18,19,21,22,23,25,26,28,32,36,42,43,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,62,70,71,72,73,74,84,86,87,88,89,90,91,92,93,94,96,97,99,100,101,102,103,106,107,108,109,110,112,114,115,118,119,120]

class Labels:

    def __init__(self,isVideo):
        self.isVideo = isVideo 
        if isVideo:
            self.labels = []
        else:
            self.labels = mur_indx

def get_labels(video):
    labels = Labels(video)
    return labels.labels 
