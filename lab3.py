# import cognitive_face as CF
import requests
import cv2
import cognitive_face as CF
from io import BytesIO
# from PIL import Image, ImageDraw
from flask import Flask, render_template, jsonify, request
from KEYS import CF_KEY


app = Flask(__name__)

CF.Key.set(CF_KEY)

BASE_URL = 'https://eastus.api.cognitive.microsoft.com/face/v1.0/'  # Replace with your regional Base URL
CF.BaseUrl.set(BASE_URL)

TotalEmotionAverage = {'anger': [], 'contempt': [], 'disgust': [], 'fear': [], 'happiness': [], 'neutral': [], 'sadness': [], 'surprise': []}

@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/return", methods=['GET', 'POST'])
def retpage():
    return render_template("return.html")

def findEmotions(vid, turns):
    assert turns >= 0, "Cannot have negative turn value"
    cap = cv2.VideoCapture(vid)
    turnsLeft = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imwrite("current.png", frame)

        # TODO: only make API call once every x turns
        if turnsLeft != 0:
            turnsLeft -= 1
        else:
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

                for key in emo:
                    TotalEmotionAverage[key].append(emo[key])


                # TODO: emotion calculation -> effects
                highestEmotion = max(emo, key=emo.get)

                ## this block shows a separate window with live frames and edits
                ## Use it for testing!
                cv2.putText(frame, str(highestEmotion), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                cv2.imshow('Test', frame)
                cv2.waitKey(2)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except KeyboardInterrupt:
                cap.release()
                cv2.destroyAllWindows()

            turnsLeft = turns

    ## and when finished
    cap.release()
    cv2.destroyAllWindows()

    for key in TotalEmotionAverage:
        TotalEmotionAverage[key] = Average(TotalEmotionAverage[key]) 

    highestAvgEmotion =  max(TotalEmotionAverage, key=TotalEmotionAverage.get)
    print('The overall emotion of people in this video is: ' + highestAvgEmotion)




def Average(lst):
    return sum(lst) / len(lst)


findEmotions('caolanTest.mp4', 60)
