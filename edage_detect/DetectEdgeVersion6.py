## coding = utf-8
"""
this version is based on Canny and Houghlines
Find Cross Points of these lines
"""
import cv2
import math
import numpy as np
import requests

from edage_detect.FindCrossPoints import getPointsWithOutOrder

ERROR_NUM = -9999

def DegreeTrans(avgAngle):
    angle = int(avgAngle / math.pi* 180)
    return angle

def ImageEdage(originInput,savePath):

    midImage = cv2.Canny(originInput,10,50,3)

    cv2.imwrite('./image/midImage.jpg',midImage)

    linesOri = cv2.HoughLines(midImage,1,math.pi/180,50,0,0)

    try:
        linesOri = linesOri.tolist()
        lines = []
        while True:
            if len(linesOri) == 0:
                break
            item = linesOri.pop(0)
            rhoMain = item[0][0]
            thetaMain = item[0][1]
            lines.append([rhoMain,thetaMain])

            if len(linesOri) == 0:
                break


            for other in linesOri:
                rhoGuest = other[0][0]
                thetaGuest = other[0][1]

                isClose1 = math.fabs(rhoMain - rhoGuest) + \
                          math.fabs(thetaMain - thetaGuest) * 100

                if isClose1 < 170:
                    linesOri.remove(other)
        for item in lines:
            rho = item[0]
            theta = item[1]
        #
            cosValue = math.cos(theta)
            sinValue = math.sin(theta)
        #
            x0 = cosValue * rho
            y0 = sinValue * rho

            x1 = int(round(x0 + 1000 * (-sinValue)))
            y1 = int(round(y0 + 1000 * cosValue))

            x2 = int(round(x0 - 1000 * (-sinValue)))
            y2 = int(round(y0 - 1000 * cosValue))

            p1 = (x1,y1)
            p2 = (x2,y2)

            # cv2.line(originInput,p1,p2,(255,255,255),5)
        #
        # cv2.imwrite(savePath, originInput)
        print("Edge detect success")
        print(lines)
        return lines
    except Exception as e:
        print("No edge in Image" + str(e))
        return[]

def getCrossLines(edgeLines):
    lines1 = []
    lines2 = []

    lineRoot = edgeLines.pop(0)
    lines1.append(lineRoot)
    thetaRoot = lineRoot[1]

    while len(edgeLines) > 0:
        lineCheck = edgeLines.pop(0)
        thetaCheck = lineCheck[1]

        isOneClass = math.fabs(thetaRoot - thetaCheck)

        if isOneClass > (math.pi/2):
            isOneClass = math.pi - isOneClass

        if isOneClass < 0.2:
            lines1.append(lineCheck)
        else:
            lines2.append(lineCheck)

    lines1 = sortArray(lines1)
    lines2 = sortArray(lines2)

    return lines1,lines2

def sortArray(arr):
    if len(arr) > 1:
        lines= np.array(arr)
        lines = lines[lines[:,0].argsort()].tolist()
        return lines
    else:
        return arr

def getOrderedPoints(srcTi):

    arr = np.float32(srcTi)
    seq = arr[:,1].argsort()

    t1 = seq.tolist()[0]
    t2 = seq.tolist()[1]
    result = []
    if arr[t1][0] < arr[t2][0]:
        min = t1
    else:
        min = t2

    for i in range(4):
        result.append(srcTi[(min + i) % 4])
    return np.float32(result)


def Main(imagePath,savePath):
    originInput = cv2.imread(imagePath)

    rows,cols,chans = originInput.shape

    while rows < 1000 or cols < 1000:
        originInput = cv2.resize(originInput, (int(cols * 2), int(rows * 2)))
        rows, cols, chan = originInput.shape

    while rows > 1000 or cols > 1000:
        originInput = cv2.resize(originInput, (int(cols / 2), int(rows / 2)))
        rows, cols, chan = originInput.shape

    (B, G, R) = cv2.split(originInput)
    B = cv2.GaussianBlur(B, (5, 5), 10)

    ret, th = cv2.threshold(B, 0, 255, cv2.THRESH_OTSU)

    g = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_open = cv2.morphologyEx(th, cv2.MORPH_CLOSE, g)

    edgeLines = ImageEdage(img_open,savePath)
    if len(edgeLines) > 0:
        lines1,lines2 = getCrossLines(edgeLines)
        pointsNeed = getPointsWithOutOrder(lines1,lines2,rows,cols)
        pointsNeed = getOrderedPoints(pointsNeed)
    else:
        pointsNeed = [[cols, rows], [0, rows], [0, 0], [cols, 0]]

    pointsNeed = [
        [pointsNeed[0][0] / cols, pointsNeed[0][1] / rows],
        [pointsNeed[1][0] / cols, pointsNeed[1][1] / rows],
        [pointsNeed[2][0] / cols, pointsNeed[2][1] / rows],
        [pointsNeed[3][0] / cols, pointsNeed[3][1] / rows]
    ]

    return pointsNeed


def getImage(IMG_URL,LOCAL_ADDR):
    imgresponse = requests.get(IMG_URL, stream=True)
    image = imgresponse.content
    filePath = LOCAL_ADDR + "/" + IMG_URL.split("/")[-1]

    try:
        nparr = np.fromstring(image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(filePath,image)
        return filePath
    except IOError:
        return ""


# if __name__ == "__main__":
#     #parm ;IMG_URL from chandra
#     # IMG_URL = "http://tapsbooktest.u.qiniudn.com/tmp/wx11aa5d122dcc41bf.o6zAJs3wa2Ul30yhGIZ_sJ3-iVtc.hxnGWJsgQ8H90f1a735037c1b4ee4582edba7ad3261b.jpg"
#     # #replace ~ with absolut path of test server
#     # LOCAL_ADDR = "~/data/img"
#     #
#     # imageURL = getImage(IMG_URL,LOCAL_ADDR)
#
#     imageURL = "/Users/developer/Downloads/image/11.jpg"
#
#     try:
#         rect = Main(imageURL, "/Users/developer/Downloads/image/11_result.jpg")
#         result = {
#             'success': 0,
#             'result': rect
#         }
#
#         #replace print to your https return
#         print(json.dumps(result))
#     except:
#         result = {
#             'success': 1,
#             'result': []
#         }
#         print(json.dumps(result))


