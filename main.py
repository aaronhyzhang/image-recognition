#Used to import al libraries, I used the facenet_pytorch library to detect faces and extract embeddings. Might change later since facenet feels like cheating.
import os
import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm
from types import MethodType

#Makes embedding of the face
def encode(img):
    res = resnet(torch.Tensor(img))
    return res

#Used to detect the bounds of the face
def detect_box(self, img, save_path=None):
    # Detect the faces
    batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
    # Select the faces
    if not self.keep_all:
        batch_boxes, batch_probs, batch_points = self.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=self.selection_method
        )
    # Extract the faces
    faces = self.extract(img, batch_boxes, save_path)
    return batch_boxes, faces

#Load the pretrained model (OP)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(
  image_size=224, keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60
)
mtcnn.detect_box = MethodType(detect_box, mtcnn)

#Reading the saved pictures, iterates through each file in the folder and saves the embeddings in a dictionary
saved_pictures = "photos"
total_faces = {}
file = "facedetector.py"
#What this is doing is splitting the file name and extension so it's assuming that it's jpg file and that the name is the name of the person.
for file in os.listdir(saved_pictures):
    person_face, extension = os.path.splitext(file)
    if extension == 'jpg':
        img = cv2.imread(f'{saved_pictures}/{file}')
        cropped = mtcnn(img)
        #This is why down here, it's saving the embedding of the face in the dictionary to the name.
        if cropped is not None:
            total_faces[person_face] = encode(cropped)[0, :]



#Opening webcam
def detect(cam=0, thres=0.7):
    vdo = cv2.VideoCapture(cam)
    while vdo.grab():
        _, img0 = vdo.retrieve()
        batch_boxes, cropped_images = mtcnn.detect_box(img0)

        if cropped_images is not None:
            #Iterates through each detected face and it's corresponding bounding box on the webcam frame
            for box, cropped in zip(batch_boxes, cropped_images):
                x, y, x2, y2 = [int(x) for x in box]
                img_embedding = encode(cropped.unsqueeze(0))
                detect_dict = {}
                #Right now it's undetected, so might need more piectures to make it more accurate.
                for k, v in total_faces.items():
                    detect_dict[k] = (v - img_embedding).norm().item()
                if detect_dict:
                    min_key = min(detect_dict, key=detect_dict.get)

                    if detect_dict[min_key] >= thres:
                        min_key = 'Undetected'
                else:
                    min_key = 'Undetected'
                
                #Making the actual red box w font
                cv2.rectangle(img0, (x, y), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                  img0, min_key, (x + 5, y + 10), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                
        # Show the image
        cv2.imshow("output", img0)
        # if cv2.waitKey(1) == ord('q'): #Quit (not working rn)
        #     cv2.destroyAllWindows()
        #     break
        k = cv2.waitKey(1)
        if k%256==27: # Esc to exit
            print('Esc pressed, closing...')
            break

if __name__ == "__main__":
    detect(0)
