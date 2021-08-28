import numpy as np
import cv2 as cv
import argparse
from scipy.spatial.distance import directed_hausdorff

def symmetric_hausdorff(A: np.ndarray, B: np.ndarray):
    _A = A.reshape((-1,2))
    _B = B.reshape((-1,2))
    return max(directed_hausdorff(_A, _B)[0], directed_hausdorff(_B, _A)[0])


class KLT:
    def __init__(self, *args):
        # super(KLT, self).__init__(*args)  
        # self.feature_params = dict( maxCorners = 100,
        #                 qualityLevel = 0.3,
        #                 minDistance = 7,
        #                 blockSize = 7 )
        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15,15),
                        maxLevel = 2,
                        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        
        self._old_frame = None
        self._feature_points = np.array([])

    # --------------------- 
    def _check_image(self, img):
        assert type(img) is np.ndarray, "img must be np.ndarray"
        assert len(img.shape) in [2,3], f"img must be gray or rgb, img.shape in [2,3]. you passed an array of shape {img.shape}"
        
    def setOldframe(self, old_frame: np.ndarray):
        self._check_image(old_frame)
        # 
        self._old_frame = old_frame.copy()
        if len(old_frame.shape) == 3:
            self._old_frame = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        
    # --------------------- 
    def _check_points(self, feature_points):
        assert type(feature_points) is np.ndarray, "feature_points must be np.ndarray"
        assert feature_points.shape[1:] == (1,2), f"feature_point ust have the following shape: [N, 1, 2] where N is the number of points, you passed an array of shape {feature_points.shape}"

    def setFeatures(self, feature_points: np.ndarray):
        self._check_points(feature_points)
        # 
        self._feature_points = feature_points.copy()

    # --------------------- 
    def _replace(self, frame, feature_points):
        self._old_frame = frame.copy()
        self._feature_points = feature_points.reshape(-1,1,2)
    
    # --------------------- 

    # def track_once(old_frame: np.ndarray, new_frame: np.ndarray, old_feature_points: np.ndarray):
    #     # self._check_points(self._feature_points)
    #     assert type(new_frame) is np.ndarray, "new_frame must be np.ndarray"
    #     assert len(new_frame.shape) in [2,3], f"new_frame must be gray or rgb, new_frame.shape in [2,3]. you passed an array of shape {new_frame.shape}"

    #     new_frame_gray = new_frame
    #     if len(new_frame.shape) == 3:
    #         new_frame_gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
    #     if len(old_frame.shape) == 3:
    #         old_frame_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    #     # 
    #     # normal tracking workflow
    #     # 
    #     # calculate optical flow
    #     _new_points, st, err = cv.calcOpticalFlowPyrLK(self._old_frame, new_frame_gray, self._feature_points, None, **self.lk_params)
    #     # Select good points
    #     resp = None
    #     status = 0
    #     if _new_points is not None and len(_new_points[st==1]) > 0:
    #         # good_new = _new_points[st==1]
    #         # good_old = self._feature_points[st==1]
    #         status = 1
    #         resp = {"good_new": _new_points[st==1], "good_old": self._feature_points[st==1]}

        
    #     # Now update the previous frame and previous points
    #     if replace_points and status: self._replace(new_frame_gray, resp['good_new'])
        
    #     return resp, status

    # --------------------- 
    def track(self, new_frame: np.ndarray, replace_points: bool=False):
        # self._check_points(self._feature_points)
        assert type(new_frame) is np.ndarray, "new_frame must be np.ndarray"
        assert len(new_frame.shape) in [2,3], f"new_frame must be gray or rgb, new_frame.shape in [2,3]. you passed an array of shape {new_frame.shape}"

        new_frame_gray = new_frame.copy()
        if len(new_frame.shape) == 3:
            new_frame_gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
        # if it is the first frame
        if self._old_frame is None:
            self._old_frame = new_frame_gray.copy()
            return self._feature_points
        # 
        # normal tracking workflow
        # 
        # calculate optical flow
        _new_points, st, err = cv.calcOpticalFlowPyrLK(self._old_frame, new_frame_gray, self._feature_points, None, **self.lk_params)
        # Select good points
        resp = None
        status = 0
        if _new_points is not None and len(_new_points[st==1]) > 0:
            # good_new = _new_points[st==1]
            # good_old = self._feature_points[st==1]
            status = 1
            resp = {"good_new": _new_points[st==1].copy(), "good_old": self._feature_points[st==1].copy()}

        
        # Now update the previous frame and previous points
        if replace_points and status: self._replace(new_frame_gray, resp['good_new'])
        
        return resp, status
        



        



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                                The example file can be downloaded from: \
                                                https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
    parser.add_argument('image', type=str, help='path to image file')
    args = parser.parse_args()

    cap = cv.VideoCapture(args.image)
    # params for ShiTomasi corner detection
    # feature_params = dict( maxCorners = 100,
    #                     qualityLevel = 0.3,
    #                     minDistance = 7,
    #                     blockSize = 7 )
    # Parameters for lucas kanade optical flow
    # lk_params = dict( winSize  = (15,15),
    #                 maxLevel = 2,
    #                 criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    klt_tracker = KLT()
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    # 1 -----------------------------
    klt_tracker.setOldframe(old_frame)
    

    # old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    
    x, y, w, h = 620, 261, 722-620, 377-261
    # frame[y:y+h, x:x+w]
    # p0 = cv.goodFeaturesToTrack(old_gray[y:y+h, x:x+w], mask = None, **feature_params)# + np.array([[y, x]])
    p0 = np.array([
        [[635.5033, 310.20938]],
        [[674.6537, 328.89133]],
        [[638.3484, 346.07788]],
        [[637.49963, 346.75482]],
        [[666.41864, 359.55716]]
    ], dtype=np.float32)
    # 2 -----------------------------
    klt_tracker.setFeatures(feature_points=p0)

    # _img = old_gray[y:y+h, x:x+w]
    # for _r in p0:
    #     a,b = _r.ravel()
    #     _img = cv.circle(_img,(int(a),int(b)),5,color[0].tolist(),-1)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    _max = 0
    while(1):
        ret,frame = cap.read()
        # 3 -----------------------------
        resp, status = klt_tracker.track(new_frame=frame, replace_points=True)
        if status:
            good_new = resp['good_new']
            good_old = resp['good_old']
        # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # # calculate optical flow
        # p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        # if p1 is not None:
        #     good_new = p1[st==1]
        #     good_old = p0[st==1]

        # draw the tracks
        # print(good_old)
        tmp1 = good_new.reshape(-1,1,2)
        # print(good_new.reshape((-1,2)))
        # print(good_new.reshape((-1,2)).shape)
        # exit()
        _dist = symmetric_hausdorff(good_new, good_old)
        if _dist > _max: _max = _dist
        print("dist:", _dist, "max:", _max,"status:",status)
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
        img = cv.add(frame,mask)
        cv.imshow('frame',img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        # old_gray = frame_gray.copy()
        # p0 = good_new.reshape(-1,1,2)