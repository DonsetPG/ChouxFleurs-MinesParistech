import numpy as np 

def find_center(img,taux):
    pointX = []
    pointY = []
    for i in range(len(img)):
        for j in range(len(img)):
            taux_red = img[i][j][0]
            if taux_red > taux:
                pointX.append(i/224.)
                pointY.append(j/224.)
    if len(pointX) > 0:          
        X=sum(pointX)/len(pointX)
        Y=sum(pointY)/len(pointY)
    else:
        X = 0.5
        Y = 0.5
    return [X,Y]
    
    
