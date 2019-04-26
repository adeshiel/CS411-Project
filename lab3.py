import cv2
import cognitive_face as CF
import datetime
import numpy as np
import os
import requests
from io import BytesIO
from flask import Flask, render_template, jsonify, request
from KEYS import CF_KEY
from pymongo import MongoClient
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/Uploads'

# if (__name__ == '__main__'):
#    app.run(
#        debug=True,
#        threaded=False,
#        processes=3,
#    )


client = MongoClient('localhost', 27017)
db = client.user
posts = db.posts



CF.Key.set(CF_KEY)
BASE_URL = 'https://eastus.api.cognitive.microsoft.com/face/v1.0/'  # Replace with your regional Base URL
CF.BaseUrl.set(BASE_URL)

TotalEmotionAverage = {'anger': [], 'contempt': [], 'disgust': [], 'fear': [], 'happiness': [], 'neutral': [], 'sadness': [], 'surprise': []}
TotalAvg = {'anger': 0, 'contempt': 0, 'disgust': 0, 'fear': 0, 'happiness': 0, 'neutral': 0, 'sadness': 0, 'surprise': 0}
exclamation = cv2.imread("effects\\exclamation.png")



@app.route("/", methods=['GET', 'POST'])
def retpage():
    if request.method == 'POST':
        vid= request.files['vide']
        vid.save(secure_filename(vid.filename))
        analysis = findEmotions(vid.filename, 20)

        video_data = {
            'user': 'name/email_here',
            'title': secure_filename(vid.filename),
            'highestAvgEmotion': analysis[0],
            'time_submitted': datetime.datetime.utcnow()
        }
        post = posts.insert_one(video_data)

        os.remove(secure_filename(vid.filename))

        return render_template("index.html", vid=analysis[1], highest=analysis[0])
    else:
        return render_template("index.html")

def findEmotions(vid, turns):
    """ calculates highest emotion using Azure and applies effects with OpenCV """
    assert turns >= 0, "Cannot have negative turn value"
    cap = cv2.VideoCapture(vid)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    highestEmotion = ""
    # out = cv2.VideoWriter('static\\outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 24, (frame_width,frame_height))
    out = cv2.VideoWriter('static\\outpy.mp4',0x7634706d, 24, (frame_width,frame_height))

    turnsLeft = 0
    faceloc = []
    print("loading...")

    angerAlpha = 0.1
    tilt = 'left'

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            cv2.imwrite("current.png", frame)

            if turnsLeft != 0:
                turnsLeft -= 1
            else:
                result = CF.face.detect("current.png", attributes='emotion')


                try:
                    if result == []:
                        continue

                    elif len(result) > 1:
                        emoList = []
                        loc_face = []
                        for face in result:
                            emotes = face['faceAttributes']['emotion']
                            # print(emotes)
                            emoList.append(emotes)
                            loc_face.append(getRectangle(face))

                            # TODO: if there is more than one face, average out emotion
                            # for now we'll take the first one

                        emo = emoList[0]
                        faceloc = loc_face[0]
                    elif len(result) == 1:
                        emo = result[0]['faceAttributes']['emotion']
                        faceloc = getRectangle(result[0])
                        # print(emo)

                    for key in emo:
                        TotalEmotionAverage[key].append(emo[key])


                    # TODO: emotion calculation -> effects
                    highestEmotion = max(emo, key=emo.get)
                    turnsLeft = turns

                except KeyboardInterrupt:
                    cap.release()
                    cv2.destroyAllWindows()

            if highestEmotion != 'anger' and angerAlpha != 0.1:
                angerAlpha = 0.1

            if highestEmotion == 'sadness':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite("gray.png", frame)
                cur_frame = cv2.imread("gray.png")
                out.write(cur_frame)

            elif highestEmotion == 'surprise':
                f_left = faceloc[0]
                f_top = faceloc[1]
                frame[f_top:f_top+300, f_left:f_left+100, : ] = exclamation
                out.write(frame)

            elif highestEmotion == 'anger':
                cur_frame = cv2.imread("current.png")
                overlay = cur_frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), (0, 0, 255), -1)
                cv2.addWeighted(overlay, angerAlpha, cur_frame, 1 - angerAlpha, 0, cur_frame)
                if angerAlpha < 0.5:
                    angerAlpha += 0.05

                if tilt == 'left':
                    rows,cols = cur_frame.shape
                    M = cv2.getRotationMatrix2D((cols/2,rows/2),10,1)
                    cur_frame = cv2.warpAffine(cur_frame,M,(cols,rows))
                    tilt = 'right'
                else:
                    rows,cols = cur_frame.shape
                    M = cv2.getRotationMatrix2D((cols/2,rows/2),350,1)
                    cur_frame = cv2.warpAffine(cur_frame,M,(cols,rows))
                    tilt = 'left'

                out.write(cur_frame)

            elif highestEmotion == 'contempt':

                f_left = faceloc[0]
                f_top = faceloc[1]
                l = np.random.randint(f_left-20, f_left+20)
                t = np.random.randint(f_top-20, f_top+20)
                cv2.putText(frame, 'Really? wtf', (t, l), cv2.FONT_HERSHEY_PLAIN, 3, 255)

                out.write(frame)

            elif highestEmotion == 'happiness':
                cur_frame = cv2.imread("current.png")
                overlay = cur_frame.copy()
                c = faceloc[-3]
                h = faceloc[-2]
                w = faceloc[-1]
                cv2.ellipse(overlay, (c[0]-150, c[1]), (h//2, w//10), 0,0,360,(191, 144,228), -1) #blushu
                cv2.ellipse(overlay, (c[0]+150, c[1]), (h//2, w//10), 0,0,360,(191, 144,228), -1)
                cv2.addWeighted(overlay, 0.35, cur_frame, 0.65, 0, cur_frame)

                out.write(cur_frame)

            elif highestEmotion == 'fear':
                print("fear...")
                cur_frame = cv2.imread("current.png")
                cur_frame = cv2.bitwise_not(cur_frame)

                out.write(cur_frame)

            elif highestEmotion == 'disgust':
                cur_frame = cv2.imread("current.png")
                overlay = cur_frame.copy()
                c = faceloc[-3]
                h = faceloc[-2]
                w = faceloc[-1]
                f_left = faceloc[0]
                f_top = faceloc[1]
                cv2.ellipse(overlay, c, (h//2, w//2), 0,0,360,(0, 255,0), -1)
                overlay[f_top:f_top+h, f_left:f_left+w] = cv2.GaussianBlur(overlay[f_top:f_top+h, f_left:f_left+w], (25,25), 0)
                cv2.addWeighted(overlay, 0.35, cur_frame, 0.65, 0, cur_frame)

                out.write(cur_frame)

            else:
                f_left = faceloc[0]
                f_top = faceloc[1]

                cv2.putText(frame, 'Meh', (f_top, f_left), cv2.FONT_HERSHEY_PLAIN, 10, 255)
                out.write(frame)


            ## this block shows a separate window with live frames and edits
            ## Use it for testing!
            # cv2.putText(frame, str(highestEmotion), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            # cv2.imshow('Test', frame)
            # cv2.waitKey(2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    ## and when finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    for key in TotalEmotionAverage:
        TotalAvg[key] = Average(TotalEmotionAverage[key])

    highestAvgEmotion =  max(TotalAvg, key=TotalEmotionAverage.get)
    print('The overall emotion of people in this video is: ' + highestAvgEmotion)
    return [highestAvgEmotion, TotalAvg]

def forLab(vid):
        """ Videos can only be read if they are present in the project folder """
        cap = cv2.VideoCapture(vid)
        emo = {}
        while(cap.isOpened()):
            ret, frame = cap.read()
            cv2.imwrite("current.png", frame)

            result = CF.face.detect("current.png", attributes='emotion')

            try:
                if result == []:
                    continue

                elif len(result) > 1:
                    emoList = []

                    for face in result:
                        emotes = face['faceAttributes']['emotion']
                        print(emotes)
                        emoList.append(emotes)

                        # TODO: if there is more than one face, average out emotion
                        # for now we'll take the first one

                    emo = emoList[0]
                elif len(result) == 1:
                    emo = result[0]['faceAttributes']['emotion']
                    print(emo)

                ## this block shows a separate window with live frames and edits
                ## Use it for testing!
                # cv2.putText(frame, str(highestEmotion), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                # cv2.imshow('Test', frame)
                # cv2.waitKey(2)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if emo != {}:
                    print("breaking")
                    break

            except KeyboardInterrupt:
                cap.release()
                cv2.destroyAllWindows()

        ## and when finished
        cap.release()
        cv2.destroyAllWindows()

        return emo

def getRectangle(faceDictionary):
    rect = faceDictionary['faceRectangle']
    left = rect['left']
    top = rect['top']
    bottom = top + rect['height']
    right = left + rect['width']
    center = ((right + left)//2, (top+bottom)//2)
    return [left, top, bottom, right, center, rect['height'],  rect['width']]

def Average(lst):
    return sum(lst) / len(lst)
