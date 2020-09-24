import cv2
import math

    
def faceTracking():
    
    res1 = (320,240)
    res2 = (640,480)
    res3 = (1280,720)
    res = res3

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    frameCounter = 0
    currentID = 0   
    faceTrackers = {}
    
    WIDTH = res[0]/2
    HEIGHT = res[1]/2
    EYE_DEPTH = 2
    hFOV = 62/2
    vFOV = 49/2
    ppcm = WIDTH*2/15.5
    term = False
    
    while not term:
        ret, frame = cap.read()
        #qframe = cv2.rotate(frame, cv2.ROTATE_180)
        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frameCounter += 1
        if frameCounter % 1 == 0:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                grey,
                scaleFactor = 1.1,
                minNeighbors = 5,
                minSize = (30, 30),                           
                flags = cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in faces:
                center = (int(x+w*0.5), int(y+h*0.5))
                fidMatch = False
                for fid in faceTrackers.keys():
                    (tx, ty, tw, th, n, u) =  faceTrackers.get(fid)
                    if tx <= center[0] <= tx+tw and ty <= center[1] <= ty+th:
                        if n < 50: n += 1
                        faceTrackers.update({fid:(x,y,w,h,n,True)})
                        fidMatch = True
                        break
                if not fidMatch:
                    faceTrackers.update({currentID:(x,y,w,h,1,True)})
                    currentID += 1
                    
        trackID = -1
        fidsToDelete = []
        for fid in faceTrackers.keys():
            (tx, ty, tw, th, n, u) =  faceTrackers.get(fid)
            if not u: n -= 1
            if n < 1: fidsToDelete.append(fid)
            else:
                faceTrackers.update({fid:(tx,ty,tw,th,n,False)})
                if n < 25:
                    pass
                else:
                    trackID = fid
       
        for fid in fidsToDelete:
            faceTrackers.pop(fid, None)
    
            
        if trackID != -1:
            
            # determine who to track
            (x, y, w, h, n, u) = faceTrackers.get(trackID)
            center = (int(x+w*0.5), int(y+h*0.5))
            hAngle = (1 - center[0]/WIDTH) * hFOV
            vAngle = (1 - center[1]/HEIGHT) * vFOV
            # Linear regression to find distance from camera to person
            c = -0.26*w+103
            
        
            # Left Eye - Horizontal
            b = 4 # distance from camera to eye
            # Angle from middel of FOV to person
            angleA = (90 - hAngle)*math.pi/180
            angleAdegrees = angleA * 180/math.pi
            print("Angle: ", angleAdegrees)
            # Distance from eye to person
            a = math.sqrt(b*b + c*c - 2*b*c*math.cos(angleA))
            # Angle from eye to person
            angleC = math.acos((a*a + b*b - c*c)/(2*a*b))
            # Rad to degrees
            angleCdegrees = angleC * 180/math.pi
            

            
            
            # Right eye - Horizontal direction
            b_hat = 2*b # distance from camera to eye
            # Distance from eye to person
            c_hat = math.sqrt(a*a + b_hat*b_hat - 2*a*b_hat*math.cos(angleC))
            # Angle from eye to person
            angleA_hat = math.acos((b_hat*b_hat + c_hat*c_hat - a*a)/(2*b_hat*c_hat))
            # Rad to degrees
            angleA_hatdegree = angleA_hat * 180/math.pi
    
            
            
            # Both eyes - Vertical direction
            b = 6 # distance from camera to eye
            # Angle from middel of FOV to person
            angleAV = (90 - vAngle)*math.pi/180
            # Distance from eye to person
            a = math.sqrt(b*b + c*c - 2*b*c*math.cos(angleAV))
            # Angle from eye to person
            angleCV = math.acos((a*a + b*b - c*c)/(2*a*b))
            # Rad to degrees
            angleCVdegrees = angleCV * 180/math.pi

                
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Release the VideoCapture object
            cap.release()
            cv2.destroyAllWindows()
            break


    
faceTracking()

    