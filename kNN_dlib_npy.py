#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ===== Import necessary libraries =====
import cv2
import numpy as np
import os, sys
import dlib
import glob
import face_recognition
from skimage import io
import time 


# In[2]:


# ===== Initialize varialbes =====
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

count = 1
dataset_path = './knn_dlib/'
offset = 10


# # 1) ========== capture image ==========
# webcam = cv2.VideoCapture(0)
# # ===== Ask user to enter name to create his/her image folder =====
# file_name = input("Enter the name of the person :  ")
# # ===== capture 30 → 20 images =====
# while (webcam.isOpened() and count <= 10) :
#     ret,frame = webcam.read()
#     if(ret == False):
#         continue
#     # ===== convert to gray frame =====
#     gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)    
#     faces = face_cascade.detectMultiScale(frame, 1.3, 5)
#     faces = sorted(faces, key=lambda f:f[2]*f[3])    # it was used in openCV -> not sure about deleting it
#     for (x, y, w, h) in faces : #face in faces : 
#         #print(face)
#         #x, y, w, h =  face 
#         # ===== draw bounding box around face =====
#         #cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
#     # ===== extract only face in different window =====
#         face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
#         face_section = cv2.resize(face_section,(150, 150))
#         #print('face section', face_section)
#     # ===== create file_name folder under knn_dlib =====
#     if not os.path.exists(dataset_path + "/" + file_name):
#         os.makedirs(dataset_path + "/" + file_name)
#     # ===== save key image as frame size =====
#     #fileName = dataset_path + "/" + file_name + "/" + file_name + "_" + str(count) + ".jpg"
#     #cv2.imwrite(fileName, frame)
#     # ===== save key image as face_section size =====
#     fileName = dataset_path + "/" + file_name + "/" + file_name + "_" + str(count) + ".jpg"
#     cv2.imwrite(fileName, face_section)
#     # ===== increment count =====
#     count += 1
#     print('count', count)
#     # ===== Display both window =====
#     cv2.imshow("FACE CROP",face_section) 
#     cv2.imshow("CAPTURE IMG",frame)
#     # ===== Hit 'q' to QUIT =====
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# # ===== Reset count to 1 & close webcam + window =====
# #count = 1
# webcam.release()
# cv2.destroyAllWindows()        

# In[3]:


# ========== 2) load image and save its info as a key to compare ==========
def get_face_encodings(face):
    """
        return np.array of face recognition model which contains location, landmarks for face encoding
    """
    bounds = face_detector(face, 1) # detect face rectangles 
    faces_landmarks = [shape_predictor(face, face_bounds) for face_bounds in bounds]
    return [np.array(face_recognition_model.compute_face_descriptor(face, face_pose, 1)) for face_pose in faces_landmarks]

def get_face_matches(known_faces, face):
    """
        return euclidean distance 
    """
    return np.linalg.norm(known_faces - face, axis=1)


# In[4]:


def find_match(known_faces, person_name, face):
    """
        min distance is the best prediction 
    """
    matches = get_face_matches(known_faces, face) 
    print('matches ', matches)
    min_index = matches.argmin() # min distance index
    print('min index ', min_index)
    min_value = matches[min_index] # min distance
    print('min value ', min_value)
    matchPercent = 100 - (min_value * 100) # convert to percentage
    print('matchPercent ', matchPercent, ' person name ', person_name)
    if matchPercent >= 70 : # at least 80% of correction -> change to 70
        return person_name +" {0:.2f}%".format(matchPercent)
    return 'Not Found'


# In[5]:


def load_face_encodings(faces_folder_path):
    """
        Load face images in person's name folder in a separate window 
    """
    image_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir(faces_folder_path))
    image_filenames = sorted(image_filenames)
    person_names = []
    for x in image_filenames :
        #print('image file name ', x)
        index = x.find('_')
        person_names.append(x[:index]) # exclude from '_'
    full_paths_to_images = [faces_folder_path + x for x in image_filenames]
    print('full paths to images ', full_paths_to_images)
    face_encodings = []
    
    for path_to_image in full_paths_to_images:
        face = io.imread(path_to_image)
        faces_bounds = face_detector(face, 1)
        if len(faces_bounds) != 1:
            print("Expected one and only one face per image: " + path_to_image + " - it has " + str(len(faces_bounds)))
        face_bounds = faces_bounds[0]
        # Get pose/landmarks of those faces
        # Will be used as an input to the function that computes face encodings
        # This allows the neural network to be able to produce similar numbers for faces of the same people, regardless of camera angle and/or face positioning in the image
        face_landmarks = shape_predictor(face, face_bounds)
        
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype='int')
        
        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (face_landmarks.part(i).x, face_landmarks.part(i).y)
        print('coords ', coords)
        face_encoding = np.array(face_recognition_model.compute_face_descriptor(face, face_landmarks, 1))
        face_encodings.append(face_encoding)
    #print('face encoding result ', face_encoding)

    # ===== save this data into numpy array file & text file =====
    np.save(dataset_path + person_names[0] + '.npy', face_encoding)
    print("data successfully saved at " + dataset_path + person_names[0] + '.npy')
    return person_names[0]


# In[6]:


def data_preparation(data_dir) :
    """
        Data Preparation by loading npy file 
    """
    face_data = []
    for dataset in os.listdir(data_dir):
        #print("Loaded "+ dataset)
        if dataset.endswith('.npy'):
            data_item = np.load(dataset_path + dataset)
            face_data.append(data_item)
    return face_data


# In[8]:


# ========== 3) initialize webcam to compare key image features ==========

data_dir = os.path.expanduser('./knn_dlib')
faces_folder_path = data_dir + '/hy/'  # just need to change folder's name to encode 
        
# ===== use timer to get how long it takes to load face encoding =====
start_loading = time.time()
person_name = load_face_encodings(faces_folder_path)
end_loading = time.time()
total_loading = end_loading - start_loading
print('Took {0:.2f} seconds to load face encodings'.format(total_loading))

# ===== load npy file =====
face_data = data_preparation(data_dir)

# ===== initialize webcam =====
camera = cv2.VideoCapture(0)
old_faces = []
cnt = 1
similarity_threshold = 0.4
while True:
    ret, frame = camera.read()
    if not ret:
        break
    frame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)    
    faces = face_detector(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5, minSize = (50, 50), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(old_faces) < len(faces) :
        old_faces = []
        for face in faces :
            tracker = dlib.correlation_tracker()
            tracker.start_track(frame, face)
            old_faces.append(tracker)
    else :
        for i, tracker in enumerate(old_faces) :
            quality = tracker.update(frame)
            if quality > 8 :
                pos = tracker.get_position()
                pos = dlib.rectangle(int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))
                face = frame[pos.top():pos.top() + pos.bottom(), pos.left():pos.left() + pos.right()]
                start = time.time() 
                face_encodings_in_image = get_face_encodings(face)
                if (face_encodings_in_image) :
                    match = find_match(face_data, person_name, face_encodings_in_image[0])
                    end = time.time()
                    total = end - start
                    print('Encoding Image Match Found took {0:.2f} seconds'.format(total))
                    cv2.putText(frame, match, (pos.left()-50, pos.top()-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else : 
                    # save unknown faces 
                    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
                    faces = sorted(faces, key=lambda f:f[2]*f[3])    # it was used in openCV -> not sure about deleting it
                    for (x, y, w, h) in faces : #face in faces : 
                        # ===== extract only face in different window =====
                        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
                    face_section = cv2.resize(face_section,(150, 150))
                        #print('face section', face_section)
                    # ===== create file_name folder under knn_dlib =====
                    if not os.path.exists(dataset_path + "/unknown"):
                        os.makedirs(dataset_path + "/unknown")
                    fileName = dataset_path + "/unknown" + "/person" + "_" + str(cnt) + ".jpg"
                    cv2.imwrite(fileName, face_section)
                    cnt += 1
                    # compare unknown faces and known faces 
                    start_loading = time.time()
                    person_name = load_face_encodings(data_dir + '/unknown/')
                    end_loading = time.time()
                    total_loading = end_loading - start_loading
                    print('Took {0:.2f} seconds to load face encodings'.format(total_loading))
                    # ===== get face encoding =====
                    pos = tracker.get_position()
                    pos = dlib.rectangle(int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))
                    face = frame[pos.top():pos.top() + pos.bottom(), pos.left():pos.left() + pos.right()]
                    face_encodings_in_image = get_face_encodings(face)
                    # ===== load npy file =====
                    #unknown_face_data = data_preparation(data_dir)
                    face_encodings_in_image = get_face_encodings(face)
                    match = find_match(face_data, person_name, face_encodings_in_image[0])
                    end = time.time()
                    total = end - start
                    print('Encoding Unknown Image Match Found took {0:.2f} seconds'.format(total))
                    cv2.putText(frame, match, (pos.left()-50, pos.top()-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    #print('Match Not Found took {0:.2f} seconds'.format(total))
                    #cv2.putText(frame, "Unknown", (pos.left()-15, pos.top()-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.rectangle(frame, (pos.left(), pos.top()), (pos.right(), pos.bottom()), (0, 255, 255), 2)
            else:
                old_faces.pop(i)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break
camera.release()
cv2.destroyAllWindows()

