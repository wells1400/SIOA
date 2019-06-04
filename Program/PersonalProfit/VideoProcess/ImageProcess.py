import os

import cv2
import numpy as np


class VideoProcess:
    def __init__(self,
                 video_path=r'H:\Other Program\Program\PersonalProfit\VideoProcess\SourceData\dark_moving.mp4',
                 image_save_dir=r'H:\Other Program\Program\PersonalProfit\VideoProcess\ImageSaveDir',
                 video_frame_interval=25,
                 dark_threshold=40,
                 dark_percent_threshold=0.8,
                 sensity_val=5000,
                 camera_pic_save_path=r'H:\Other Program\Program\PersonalProfit\VideoProcess\CameraOpen'
                 ):
        # 视频文件位置
        self.video_path = video_path
        # 视频帧图像保存位置
        self.image_save_dir = image_save_dir
        # 视频帧数间隔
        self.video_frame_interval = video_frame_interval
        # 图片灰度值阈值，像素点灰度值低于这个阈值认为是暗点
        self.dark_threshold = dark_threshold
        # 暗点占比阈值，大于这个阈值认为该图片是暗的
        self.dark_percent_threshold = dark_percent_threshold
        # 摄像头拍摄的照片保存的目录
        self.camera_pic_save_path = camera_pic_save_path
        # 截取的图片的路径列表
        self.pic_path = []
        # 当前遍历到的图片的索引
        self.pic_index_pos = 0
        # 判断是否存在移动物品的敏感度（帧差）
        self.sensity_val = sensity_val

    # 清空指定目录下所有文件
    def delete_files(self, path_dir):
        os.chdir(path_dir)
        fileList = list(os.listdir(path_dir))
        for file in fileList:
            if os.path.isfile(file):
                os.remove(file)

                # 按指定帧数间隔截取视频并保存到指定位置

    def video_frame_capture(self):
        # 清空指定目录下所有文件
        self.delete_files(self.image_save_dir)
        vc = cv2.VideoCapture(self.video_path)
        count = 1
        if vc.isOpened():
            rval, frame = vc.read()
            cv2.imwrite(self.image_save_dir + r'\\part_%d' % count + '.jpg', frame)
        else:
            rval = False
        while rval:
            rval, frame = vc.read()
            if count % self.video_frame_interval == 0:
                cv2.imwrite(self.image_save_dir + r'\\part_%d' % count + '.jpg', frame)
            count += 1
            cv2.waitKey(1)
        vc.release()

    # 输入一场图片，以及判断图片是不是暗的那个像素点占比分类阈值，
    # 判断这个图片是不是暗的，如果是暗的返回True，否则返回False
    def check_image_if_dark(self, pic_path):
        # 转化成灰度图
        img = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
        total_pixels = img.shape[0] * img.shape[1]
        dark_pixel_count = 0
        for row in img:
            row_count = np.array(row)
            dark_pixel_count += len(row_count[row_count < self.dark_threshold])
        return dark_pixel_count / total_pixels > self.dark_percent_threshold

    # 读取指定目录，返回目录下所有以.jpg结尾的图片路径
    def load_pic_paths(self, dir_path):
        res_pic_paths = []
        for root, firs, files in os.walk(dir_path):
            for file in files:
                if 'jpg' in file.split('.'):
                    res_pic_paths.append(root + r'\\' + file)
        return res_pic_paths

    # 控制开启摄像头,并拍摄一张照片保存,作为开启灯光的代替
    def open_camera_and_shot(self,
                             camera_pic_save_path=r'H:\Other Program\Program\PersonalProfit\VideoProcess\CameraOpen'):
        # 先清空camera_pic_save_path下已有的文件
        self.delete_files(camera_pic_save_path)
        print('开启灯光！（这里用摄像头拍张照代替，图片存储位置在：%s）' % camera_pic_save_path)
        cap = cv2.VideoCapture(0)
        f, frame = cap.read()  # 拍张照
        cv2.imwrite(camera_pic_save_path + r'\\' + 'camera_opened_flag.jpg', frame)
        cap.release()  # 关闭调用的摄像头

    # 输入视频路径,将这张暗图作为背景，根据后续的图像帧判断是否存在动态实体
    # 存在动态实体返回1， 不存在返回0
    def check_if_moving_conent(self):
        pre_frame = None
        while self.pic_index_pos < len(self.pic_path):
            cur_frame = cv2.imread(self.pic_path[self.pic_index_pos])

            gray_img = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
            gray_img = cv2.resize(gray_img, (500, 500))
            gray_img = cv2.GaussianBlur(gray_img, (21, 21), 0)

            if pre_frame is None:
                pre_frame = gray_img
            else:
                img_delta = cv2.absdiff(pre_frame, gray_img)
                thresh = cv2.threshold(img_delta, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c) < self.sensity_val:  # 设置敏感度
                        continue
                    else:
                        print("昏暗环境下检测到移动物体！!")
                        # 调用摄像头
                        return True
            pre_frame = gray_img
            # 下一张图片
            self.pic_index_pos += 1
        cv2.destroyAllWindows()
        return False

    # 灯光控制主程
    def video_process_engine(self):
        # 先截取视频图片
        self.video_frame_capture()
        # 获取截取的图片的路径列表
        self.pic_path = self.load_pic_paths(self.image_save_dir)
        # 持续读取图片
        while self.pic_index_pos < len(self.pic_path):
            # 如果不是暗图，则啥也不做，
            if not self.check_image_if_dark(self.pic_path[self.pic_index_pos]):
                self.pic_index_pos += 1
                continue
            # 否则截取到暗图
            if self.check_if_moving_conent():
                return self.open_camera_and_shot(camera_pic_save_path=self.camera_pic_save_path)
        print(r'灯光不开启！')
        return


if __name__ == '__main__':
    # video_path是视频的路径
    video_path = r'H:\Other Program\Program\PersonalProfit\VideoProcess\SourceData\light_no_moving.mp4'
    # image_save_dir截取的视频图像帧存放目录
    image_save_dir = r'H:\Other Program\Program\PersonalProfit\VideoProcess\ImageSaveDir'
    video_processor = VideoProcess(video_path=video_path,
                                   image_save_dir=image_save_dir,
                                   #  视频取帧间隔
                                   video_frame_interval=25,
                                   # 判断像素点明暗的阈值，像素点灰度值小于这个值认为该像素点是暗的
                                   dark_threshold=40,
                                   # 判断灰度图是不是暗图或者明亮的阈值，是暗像素点占所有像素点的比例，大于这个阈值就把这个图像认为是昏暗的
                                   dark_percent_threshold=0.8,
                                   # 判断上个视频帧与当前视频帧是否存在较大差异（即存在醒目移动物体）的阈值
                                   sensity_val=5000,
                                   # # 如果是昏暗环境下检测到了移动物体，那么开启摄像头拍张照，camera_pic_save_path是照片存储的位置
                                   camera_pic_save_path=r'H:\Other Program\Program\PersonalProfit\VideoProcess\CameraOpen'
                                   )
    # 算法运行入口
    video_processor.video_process_engine()
