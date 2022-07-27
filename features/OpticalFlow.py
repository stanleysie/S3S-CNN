import time
import subprocess
import cv2
import numpy as np

class OpticalFlow:
    def __init__(self, emotions, db_OF):
        self.emotions = emotions
        self.db_OF = db_OF
        self.dataset = {
            "X": [],
            "y": []
        }
        self.face_bounding_box = {}
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')    


    def read_image(self, path):
        img = cv2.imread(path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray_img
    

    def generate_bounding_box(self, gray_img):
        faces = self.face_cascade.detectMultiScale(gray_img, 1.1, 4)
        x, y, w, h = faces[0]
        return str(x), str(y), str(w), str(h)
    

    def generate_optical_strain(self, flow):
        u = flow[...,0]
        v = flow[...,1]

        ux, uy = np.gradient(u)
        vx, vy = np.gradient(v)

        e_xy = 0.5*(uy + vx)
        e_xx = ux
        e_yy = vy
        e_m = e_xx ** 2 + 2 * e_xy ** 2 + e_yy ** 2
        e_m = np.sqrt(e_m)
        e_m = cv2.normalize(e_m, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        e_m = e_m.astype(np.uint8)
        
        return e_m
    

    def generate_optical_flow(self, index, path, db, emotion, subject, onset, apex):   
        start_time = time.process_time()
        # compute optical flow (TVL1Flow)
        subprocess.check_call(f"bash ../run.sh '{onset}' '{apex}'", shell=True)
        end_time = time.process_time()
        print(f"{db} {emotion} {subject} {end_time - start_time}s")

        onset_img, onset_img_gray = self.read_image(onset)

        # creating mask based on the image
        hsv = np.zeros_like(onset_img)
        hsv[..., 1] = 255

        # reading optical flow
        flow = cv2.readOpticalFlow("flow.flo")

        # compute for the optical flow's magnitude and direction from the flow vector
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # compute for the horizontal and vertical components of the optical flow
        horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)     
        vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
        horz = horz.astype('uint8')
        vert = vert.astype('uint8')

        # generate optical strain
        optical_strain = self.generate_optical_strain(flow)

        of_path = f"{path}/{index} {emotion} {subject} OF.jpg"
        os_path = f"{path}/{index} {emotion} {subject} OS.jpg"
        horz_path = f"{path}/{index} {emotion} {subject} H.jpg"
        vert_path = f"{path}/{index} {emotion} {subject} V.jpg"

        self.dataset['X'].append((of_path, os_path, horz_path, vert_path))
        self.dataset['y'].append(self.emotions.index(emotion))

        if db in ['SAMM', 'CASME_II']:
            x, y, w, h = self.generate_bounding_box(onset_img_gray)
            self.face_bounding_box[of_path] = (x, y, w, h)
            self.face_bounding_box[os_path] = (x, y, w, h)
            self.face_bounding_box[horz_path] = (x, y, w, h)
            self.face_bounding_box[vert_path] = (x, y, w, h)

        cv2.imwrite(of_path, rgb)
        cv2.imwrite(os_path, optical_strain)
        cv2.imwrite(horz_path, horz)
        cv2.imwrite(vert_path, vert)
    

    def crop_resize(self, path, img, x, y, w, h, dim, crop=True):
        if crop:
            cropped_image = img[y:y+h, x:x+w]
            resized_image = cv2.resize(cropped_image, dim, interpolation=cv2.INTER_AREA)
            cv2.imwrite(path, resized_image)
        else:
            resized_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            cv2.imwrite(path, resized_image)
    

    def get_dataset(self):
        return self.dataset
    

    def get_face_bounding_box(self):
        return self.face_bounding_box