import cv2
import sys
import dlib
import time
import socket
import struct
import numpy as np
import tensorflow as tf
from threading import Thread, Lock
import multiprocessing as mp
from config import get_config
import pickle
import math
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="gaze correction")

    parser.add_argument("-v", '--video',
                        type=str,
                        default='basicvideo.mp4',
                        help='video path')
    parser.add_argument('--P_c_y', type=eval, default=0,
                        help='vertical distance in cm of center of screen from camera (negative if camera is on top')
    parser.add_argument('--system', "-s", type=str, default="home", help="")
    return check_args(parser.parse_args())


def check_args(arguments):
    try:
        assert type(arguments.video) is str or arguments.video == 0
    except ValueError:
        print('video path must be a string, instead got:\n{}\n{}'.format(arguments.video, type(args.video)))

    return arguments


class GazeRedirectionSystem:
    def __init__(self, video_path, P_c_y, overwriting_config):

        self.video_file_name = video_path
        self.vs = cv2.VideoCapture(self.video_file_name)

        width = int(self.vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.input_res = [width, height]
        self.size_video = self.input_res
        # Landmark identifier. Set the filename to whatever you named the downloaded file
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./lm_feat/shape_predictor_68_face_landmarks.dat")
        self.size_df = (320, 240)
        self.size_I = (48, 64)
        # initial value
        self.Rw = [0, 0]
        self.Pe_z = -60
        # get configurations
        self.f = conf.f
        self.Ps = (conf.S_W, conf.S_H)  # screen size cm [w,h], used for shifting angle estimation
        self.Pc = (conf.P_c_x, conf.P_c_y, conf.P_c_z)
        if overwriting_config != [None]:
            self.f = overwriting_config[-2]
            self.Ps = (overwriting_config[0], overwriting_config[1])
            self.Pc = (conf.P_c_x, overwriting_config[-1], conf.P_c_z)

        elif P_c_y != 0:
            self.Pc = (conf.P_c_x, P_c_y, conf.P_c_z)

        self.Pe = [self.Pc[0], self.Pc[1], self.Pe_z]  # H,V,D
        # start video sender
        # self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # self.client_socket.connect((conf.tar_ip, conf.sender_port))
        # self.encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]

        # one process

        self.face_detect_size = [320, 240]
        self.x_ratio = self.size_video[0] / self.face_detect_size[0]
        self.y_ratio = self.size_video[1] / self.face_detect_size[1]

        # load model to gpu
        print("Loading model of [L] eye to GPU")
        with tf.Graph().as_default() as g:
            # define placeholder for inputs to network
            with tf.name_scope('inputs'):
                self.LE_input_img = tf.placeholder(tf.float32, [None, conf.height, conf.width, conf.channel],
                                                   name="input_img")
                self.LE_input_fp = tf.placeholder(tf.float32, [None, conf.height, conf.width, conf.ef_dim],
                                                  name="input_fp")
                self.LE_input_ang = tf.placeholder(tf.float32, [None, conf.agl_dim], name="input_ang")
                self.LE_phase_train = tf.placeholder(tf.bool, name='phase_train')  # a bool for batch_normalization

            self.LE_img_pred, _, _ = model.inference(self.LE_input_img, self.LE_input_fp, self.LE_input_ang,
                                                     self.LE_phase_train, conf)

            # split modle here
            self.L_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False),
                                     graph=g)
            # load model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(model_dir + 'L/')
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(self.L_sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')

        print("Loading model of [R] eye to GPU")
        with tf.Graph().as_default() as g2:
            # define placeholder for inputs to network
            with tf.name_scope('inputs'):
                self.RE_input_img = tf.placeholder(tf.float32, [None, conf.height, conf.width, conf.channel],
                                                   name="input_img")
                self.RE_input_fp = tf.placeholder(tf.float32, [None, conf.height, conf.width, conf.ef_dim],
                                                  name="input_fp")
                self.RE_input_ang = tf.placeholder(tf.float32, [None, conf.agl_dim], name="input_ang")
                self.RE_phase_train = tf.placeholder(tf.bool, name='phase_train')  # a bool for batch_normalization

            self.RE_img_pred, _, _ = model.inference(self.RE_input_img, self.RE_input_fp, self.RE_input_ang,
                                                     self.RE_phase_train, conf)

            # split modle here
            self.R_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False),
                                     graph=g2)
            # load model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(model_dir + 'R/')
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(self.R_sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')

        self.run()

    def run(self):

        global concat_writer
        concat = True  # TODO connect to args
        size_window = [659, 528]
        # self.vs = cv2.VideoCapture(self.video_file_name)
        self.vs.set(3, self.size_video[0])
        self.vs.set(4, self.size_video[1])
        cv2.namedWindow(conf.uid)
        cv2.moveWindow(conf.uid, int(Rs[0] / 2) - int(size_window[0] / 2), int(Rs[1] / 2) - int(size_window[1] / 2))

        # new video
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(self.vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv2.namedWindow('output')
        # cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('output', width, height)
        output_video_path = "output_{}.mp4".format(self.video_file_name[:-4])

        out = cv2.VideoWriter(output_video_path, codec, 24, (width, height))
        if concat:
            concat_output_video_path = "concat_output_{}.mp4".format(self.video_file_name[:-4])
            concat_writer = cv2.VideoWriter(concat_output_video_path, codec, 24, (width * 2, height))

        while True:
            ret, recv_frame = self.vs.read()
            if ret:
                cv2.imshow(conf.uid, recv_frame)  # original

                frame = recv_frame.copy()
                shared_v = self.face_detection(frame)
                out_frame = self.redirect_gaze(frame, shared_v)

                cv2.imshow('output', out_frame)
                out.write(out_frame)
                # if (time.time() - t) > 1:
                #     t = time.time()
                if concat:
                    Concatenated_frame = cv2.hconcat([recv_frame, out_frame])
                    concat_writer.write(Concatenated_frame)
                k = cv2.waitKey(10)
                if k == ord('q'):
                    # data = pickle.dumps('stop')
                    # self.client_socket.sendall(struct.pack("L", len(data))+data)
                    time.sleep(1)
                    cv2.destroyWindow(conf.uid)
                    cv2.destroyWindow('output')
                    # self.client_socket.shutdown(socket.SHUT_RDWR)
                    # self.client_socket.close()
                    self.vs.release()
                    self.L_sess.close()
                    self.R_sess.close()
                    break
                    # elif k == ord('r'):
                    #     if redir:
                    #         redir = False
                    #     else:
                    #         redir = True
            else:
                break

    def redirect_gaze(self, frame, shared_v):
        """
        takes a frame and head coords and returns redirected frame
        """
        # head detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detect_gray = cv2.resize(gray, (self.size_df[0], self.size_df[1]))
        detections = self.detector(face_detect_gray, 0)

        # rg_thread = Thread(target=self.flx_gaze, args=(frame, gray, detections))
        # rg_thread.start()
        # return True
        return self.flx_gaze(frame, gray, detections, shared_v)

    def shifting_angles_estimator(self, R_le, R_re, shared_v):
        """
        calculates geometric sizes for redirection
        """
        # get P_w

        """try:
            tar_win = win32gui.FindWindow(None, "Remote")
            #left, top, reight, bottom
            Rw_lt = win32gui.GetWindowRect(tar_win)
            size_window = (Rw_lt[2]-Rw_lt[0], Rw_lt[3]-Rw_lt[1])
        except:"""
        # Rw_lt = [int(Rs[0])-int(size_window[0]/2),int(Rs[1])-int(size_window[1]/2)]
        size_window = (659, 528)
        Rw_lt = [int(Rs[0]) / 2 - int(size_window[0] / 2), int(Rs[1]) / 2 - int(size_window[1] / 2),
                 int(Rs[0]) / 2 + int(size_window[0] / 2), int(Rs[1]) / 2 + int(size_window[1] / 2)]
        # print("Missing the window")
        # get pos head

        pos_remote_head = [int(size_window[0] / 2), int(size_window[1] / 2)]
        if (shared_v[0] != 0) & (shared_v[1] != 0):
            pos_remote_head[0] = shared_v[0]
            pos_remote_head[1] = shared_v[1]
        else:
            pos_remote_head = (int(size_window[0] / 2), int(size_window[1] / 2))

        R_w = (Rw_lt[0] + pos_remote_head[0], Rw_lt[1] + pos_remote_head[1])
        Pw = (self.Ps[0] * (R_w[0] - Rs[0] / 2) / Rs[0], self.Ps[1] * (R_w[1] - Rs[1] / 2) / Rs[1], 0)

        # get Pe
        self.Pe[2] = -(self.f * conf.P_IDP) / np.sqrt((R_le[0] - R_re[0]) ** 2 + (R_le[1] - R_re[1]) ** 2)
        # x-axis needs flip
        self.Pe[0] = -np.abs(self.Pe[2]) * (R_le[0] + R_re[0] - self.size_video[0]) / (2 * self.f) + self.Pc[0]
        self.Pe[1] = np.abs(self.Pe[2]) * (R_le[1] + R_re[1] - self.size_video[1]) / (2 * self.f) + self.Pc[1]

        # calcualte alpha
        a_w2z_x = math.degrees(math.atan((Pw[0] - self.Pe[0]) / (Pw[2] - self.Pe[2])))
        a_w2z_y = math.degrees(math.atan((Pw[1] - self.Pe[1]) / (Pw[2] - self.Pe[2])))

        a_z2c_x = math.degrees(math.atan((self.Pe[0] - self.Pc[0]) / (self.Pc[2] - self.Pe[2])))
        a_z2c_y = math.degrees(math.atan((self.Pe[1] - self.Pc[1]) / (self.Pc[2] - self.Pe[2])))

        alpha = [int(a_w2z_y + a_z2c_y), int(a_w2z_x + a_z2c_x)]  # (V,H)

        return alpha, self.Pe, R_w # TODO only alpha is used

    @staticmethod
    def get_inputs(frame, shape, pos="L", size_I=(48, 64)):
        """
        Takes a frame, the anchors of the eyes (need only 6), L/R (which eye) and desired res of eye image cutout
        and returns:

        normalized image of the eye cutout,
        anchor maps for the eye, center coord,
        center coord of eye,
        original shape of the cutout,
        top left coord of the cutout

        """
        # depending on which eye, the six anchors around the eye will have different indices
        if pos not in ["L", "R"]:
            print("Error: Wrong Eye")
            exit()
        if pos == "R":
            lc = 36  # left anchor
            rc = 39
            FP_seq = [36, 37, 38, 39, 40, 41]
        else:  # pos == "L":
            lc = 42  # left anchor
            rc = 45
            FP_seq = [45, 44, 43, 42, 47, 46]

        # center of the eye
        eye_cx = (shape.part(rc).x + shape.part(lc).x) * 0.5
        eye_cy = (shape.part(rc).y + shape.part(lc).y) * 0.5
        eye_center = [eye_cx, eye_cy]

        # eye parameters for the cutout
        eye_len = np.absolute(shape.part(rc).x - shape.part(lc).x)
        bx_d5w = eye_len * 3 / 4
        bx_h = 1.5 * bx_d5w
        sft_up = bx_h * 7 / 12
        sft_low = bx_h * 5 / 12

        # segmented eye image, a cutout
        img_eye = frame[int(eye_cy - sft_up):int(eye_cy + sft_low), int(eye_cx - bx_d5w):int(eye_cx + bx_d5w)]

        ori_size = [img_eye.shape[0], img_eye.shape[1]]  # shape of eye cutout
        LT_coor = [int(eye_cy - sft_up), int(eye_cx - bx_d5w)]  # (y,x) top left coord of cutout
        img_eye = cv2.resize(img_eye, (size_I[1], size_I[0]))  # resize cutout to desired size
        # create anchor maps
        ach_map = []
        for i, d in enumerate(FP_seq):  # for each anchor around the eye (out of 6)
            resize_x = int((shape.part(d).x - LT_coor[1]) * size_I[1] / ori_size[1]) # local x coord of anchor d
            resize_y = int((shape.part(d).y - LT_coor[0]) * size_I[0] / ori_size[0]) # local y coord of anchor d
            # y
            ach_map_y = np.expand_dims(np.expand_dims(np.arange(0, size_I[0]) - resize_y, axis=1), axis=2)
            ach_map_y = np.tile(ach_map_y, [1, size_I[1], 1])
            # x
            ach_map_x = np.expand_dims(np.expand_dims(np.arange(0, size_I[1]) - resize_x, axis=0), axis=2)
            ach_map_x = np.tile(ach_map_x, [size_I[0], 1, 1])
            if i == 0:
                ach_map = np.concatenate((ach_map_x, ach_map_y), axis=2)
            else:
                ach_map = np.concatenate((ach_map, ach_map_x, ach_map_y), axis=2)

        return img_eye / 255, ach_map, eye_center, ori_size, LT_coor

    def flx_gaze(self, frame, gray, detections, shared_v, pixel_cut=(3, 4), size_I=(48, 64)):
        """
        replaces eyes in frame with redirected ones
        """
        alpha_w2c = [0, 0]
        x_ratio = self.size_video[0] / self.size_df[0]
        y_ratio = self.size_video[1] / self.size_df[1]
        LE_M_A = []
        RE_M_A = [] # a not neeed declataion of the anchor maps
        p_e = [0, 0]
        R_w = [0, 0]
        for k, bx in enumerate(detections):
            # Get facial landmarks
            time_start = time.time()
            target_bx = dlib.rectangle(left=int(bx.left() * x_ratio), right=int(bx.right() * x_ratio),
                                       top=int(bx.top() * y_ratio), bottom=int(bx.bottom() * y_ratio)) # dlib rectangle bbox
            shape = self.predictor(gray, target_bx)
            # get eye:
            # normalized image of the eye cutout,
            # anchor maps for the eye, center coord,
            # center coord of eye,
            # original shape of the cutout,
            # top left coord of the cutout
            LE_img, LE_M_A, LE_center, size_le_ori, R_le_LT = self.get_inputs(frame, shape, pos="L", size_I=size_I)
            RE_img, RE_M_A, RE_center, size_re_ori, R_re_LT = self.get_inputs(frame, shape, pos="R", size_I=size_I)
            # shifting angles estimator
            alpha_w2c, _, _ = self.shifting_angles_estimator(LE_center, RE_center,
                                                             shared_v)  # p_e, R_w = self.shifting_angles_estimator(LE_center, RE_center, shared_v)

            time_get_eye = time.time() - time_start
            # gaze manipulation
            time_start = time.time()

            # gaze redirection
            # left Eye
            # inputs:  normalized image of the eye cutout, anchor maps for the eye, rotation angles
            LE_infer_img = self.L_sess.run(self.LE_img_pred, feed_dict={
                self.LE_input_img: np.expand_dims(LE_img, axis=0),
                self.LE_input_fp: np.expand_dims(LE_M_A, axis=0),
                self.LE_input_ang: np.expand_dims(alpha_w2c, axis=0),
                self.LE_phase_train: False
            })
            LE_infer = cv2.resize(LE_infer_img.reshape(size_I[0], size_I[1], 3), (size_le_ori[1], size_le_ori[0]))
            # right Eye
            RE_infer_img = self.R_sess.run(self.RE_img_pred, feed_dict={
                self.RE_input_img: np.expand_dims(RE_img, axis=0),
                self.RE_input_fp: np.expand_dims(RE_M_A, axis=0),
                self.RE_input_ang: np.expand_dims(alpha_w2c, axis=0),
                self.RE_phase_train: False
            })
            RE_infer = cv2.resize(RE_infer_img.reshape(size_I[0], size_I[1], 3), (size_re_ori[1], size_re_ori[0]))

            # replace eyes
            frame[(R_le_LT[0] + pixel_cut[0]):(R_le_LT[0] + size_le_ori[0] - pixel_cut[0]),
                  (R_le_LT[1] + pixel_cut[1]):(R_le_LT[1] + size_le_ori[1] - pixel_cut[1])] = \
                LE_infer[pixel_cut[0]:(-1 * pixel_cut[0]), pixel_cut[1]:-1 * (pixel_cut[1])] * 255
            frame[(R_re_LT[0] + pixel_cut[0]):(R_re_LT[0] + size_re_ori[0] - pixel_cut[0]),
                  (R_re_LT[1] + pixel_cut[1]):(R_re_LT[1] + size_re_ori[1] - pixel_cut[1])] = \
                RE_infer[pixel_cut[0]:(-1 * pixel_cut[0]), pixel_cut[1]:-1 * (pixel_cut[1])] * 255

        # frame = self.monitor_para(frame, alpha_w2c, self.Pe, R_w)

        # result, imgencode = cv2.imencode('.jpg', frame, self.encode_param)
        # data = pickle.dumps(imgencode, 0)
        # self.client_socket.sendall(struct.pack("L", len(data)) + data)
        # return True
        return frame

    def monitor_para(self, frame, fig_alpha, fig_eye_pos, fig_R_w):
        """
        adds parameters on the output frame
        """
        cv2.rectangle(frame,
                      (self.size_video[0] - 150, 0), (self.size_video[0], 55),
                      (255, 255, 255), -1
                      )
        cv2.putText(frame,
                    'Eye:[' + str(int(fig_eye_pos[0])) + ',' + str(int(fig_eye_pos[1])) + ',' + str(
                        int(fig_eye_pos[2])) + ']',
                    (self.size_video[0] - 140, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame,
                    'alpha:[V=' + str(int(fig_alpha[0])) + ',H=' + str(int(fig_alpha[1])) + ']',
                    (self.size_video[0] - 140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame,
                    'R_w:[' + str(int(fig_R_w[0])) + ',' + str(int(fig_R_w[1])) + ']',
                    (self.size_video[0] - 140, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        return frame

    def face_detection(self, frame):
        """
        gets the coords of the head relative to the window (frame)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detect_gray = cv2.resize(gray, (self.face_detect_size[0], self.face_detect_size[1]))
        detections = self.detector(face_detect_gray, 0)
        coor_remote_head_center = [0, 0]
        for k, bx in enumerate(detections):
            coor_remote_head_center = [int((bx.left() + bx.right()) * self.x_ratio / 2),
                                       int((bx.top() + bx.bottom()) * self.y_ratio / 2)]
            break

        shared_v = [0, 0]
        shared_v[0] = coor_remote_head_center[0]
        shared_v[1] = coor_remote_head_center[1]
        return shared_v


if __name__ == '__main__':

    conf, _ = get_config()
    if conf.mod == 'flx':
        import flx as model
    else:
        sys.exit("Wrong Model selection: flx or deepwarp")

    # system parameters
    model_dir = './' + conf.weight_set + '/warping_model/' + conf.mod + '/' + str(conf.ef_dim) + '/'
    # size_video = [640, 480]
    # fps = 0
    # P_IDP = 5 # distance between eyes
    # depth = -50
    # for monitoring

    # environment parameter
    Rs = (1920, 1080)
    args = parse_args()
    _video_path = args.video
    _P_c_y = args.P_c_y
    _sys = args.system
    overwrite_config = [None]
    if _sys == "macbook":
        print("overwriting config for MacBook Pro")
        screen_cm_w = 30
        screen_cm_h = 20
        screen_pix_w = 2560
        screen_pix_h = 1600
        focal_pix = 920  # 700 # 500 # 5333
        _P_c_y = -10
        Rs = (screen_pix_w, screen_pix_h)
        overwrite_config = [screen_cm_w, screen_cm_h, screen_pix_w, screen_pix_h, focal_pix, _P_c_y]
    converter = GazeRedirectionSystem(_video_path, _P_c_y, overwrite_config)
