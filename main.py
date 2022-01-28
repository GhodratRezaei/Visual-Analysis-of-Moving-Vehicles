import argparse
from cmath import rect
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import numpy as np

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, is_ascii, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, load_classifier, time_sync


MODEL_FILE_PATH = "yolo_model.pt"
INPUT_DIRECTORY_PATH = "video.mp4"


def center_of_rectangle(rectangle):
    return ((rectangle[0] + rectangle[2]) / 2.0, (rectangle[1] + rectangle[3]) / 2.0)


def is_point_inside_rectangle(point, rectangle):
    if point[0] < rectangle[0]:
        return False
    if point[1] < rectangle[1]:
        return False
    if point[0] > rectangle[2]:
        return False
    if point[1] < rectangle[3]:
        return False
    return True


def calculate_speed(start_point, end_point, vanishing_point, rear_marker, front_marker, markers_distance, time_delta):
    vanishing_point_start_point_norm = np.linalg.norm(vanishing_point - start_point)
    front_marker_start_point_norm = np.linalg.norm(front_marker - start_point)
    vanishing_point_rear_marker_norm = np.linalg.norm(vanishing_point - rear_marker)
    front_marker_rear_marker_norm = np.linalg.norm(front_marker - rear_marker)

    start_point_distance = (vanishing_point_rear_marker_norm / front_marker_rear_marker_norm) * markers_distance / (vanishing_point_start_point_norm / front_marker_start_point_norm)

    vanishing_point_end_point_norm = np.linalg.norm(vanishing_point - end_point)
    front_marker_end_point_norm = np.linalg.norm(front_marker - end_point)
    vanishing_point_rear_marker_norm = np.linalg.norm(vanishing_point - rear_marker)
    front_marker_rear_marker_norm = np.linalg.norm(front_marker - rear_marker)

    end_point_distance = (vanishing_point_rear_marker_norm / front_marker_rear_marker_norm) * markers_distance / (vanishing_point_end_point_norm / front_marker_end_point_norm)

    return abs(end_point_distance - start_point_distance) / time_delta


def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def near_points(point1, point2, threshold_x, threshold_y, threshold_distance):
    if point1[0] - point2[0] > threshold_x:
        return False
    if point1[1] - point2[1] > threshold_y:
        return False
    if distance(point1, point2) > threshold_distance:
        return False
    return True


def preprocess_bounding_boxels_list(bounding_boxes_list, image_width, image_height):
    bounding_boxes_list_copy = bounding_boxes_list.copy()
    bounding_boxes_list = []
    for i in range(len(bounding_boxes_list_copy)):
        if center_of_rectangle(bounding_boxes_list_copy[i])[1] >= 0.5 * image_height:
            bounding_boxes_list.append(bounding_boxes_list_copy[i])
    return bounding_boxes_list


def preprocess_points_list_by_time(points_list_by_time, image_width, image_height):
    points_list_by_time_copy = points_list_by_time.copy()
    points_list_by_time = []
    for i in range(len(points_list_by_time_copy)):
        if points_list_by_time_copy[i][-1][1] >= 0.5 * image_height:
            points_list_by_time.append(points_list_by_time_copy[i])
    return points_list_by_time


def remove_unupdated_points(points_list_by_time, updated_lists_dict):
    points_list_by_time_copy = points_list_by_time.copy()
    points_list_by_time = []
    for i in range(len(points_list_by_time_copy)):
        if i in updated_lists_dict.keys() and updated_lists_dict[i]:
            points_list_by_time.append(points_list_by_time_copy[i])
    return points_list_by_time


points_list_by_time = []
total_objects = 0


def process_bounding_boxes(bounding_boxes_list, image):
    global points_list_by_time
    global total_objects
    image_width = image.shape[1]
    image_height = image.shape[0]
    bounding_boxes_list = preprocess_bounding_boxels_list(bounding_boxes_list, image_width, image_height)
    points_list_by_time = preprocess_points_list_by_time(points_list_by_time, image_width, image_height)
    cv2.rectangle(image, (0, int(0.5 * image_height)), (image_width, image_height), (0, 255, 255), 5)
    cv2.rectangle(image, (0, int(0.5 * image_height)), (int(0.3 * image_width), int(0.5 * image_height) + int(0.06 * image_height)), (255, 255, 255), -1)
    cv2.putText(img=image, text="Number of cars: " + str(len(bounding_boxes_list)),
                org=(int(0.025 * image_width), int(0.5 * image_height) + int(0.045 * image_height)), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=image_width / 1280, color=(0, 0, 0), thickness=int(2 * image_width / 1280 + 0.5))
    updated_lists_dict = {}
    points_list = []
    for i in range(len(bounding_boxes_list)):
        points_list.append(center_of_rectangle(bounding_boxes_list[i]))
    for i in range(len(points_list)):
        min_distance_index = -1
        for j in range(len(points_list_by_time)):
            if near_points(points_list[i], points_list_by_time[j][-1], 50, 50, 50):
                if min_distance_index < 0 or distance(points_list[i], points_list_by_time[j][-1]) < distance(points_list[i], points_list_by_time[min_distance_index][-1]):
                    min_distance_index = j
        if min_distance_index != -1:
            updated_lists_dict[min_distance_index] = True
            points_list_by_time[min_distance_index].append(points_list[i])
        else:
            total_objects += 1
            points_list_by_time.append([points_list[i]])
            updated_lists_dict[len(points_list_by_time) - 1] = True
    points_list_by_time = remove_unupdated_points(points_list_by_time, updated_lists_dict)
    print("")
    print("total cars until now:")
    print(total_objects)
    for i in range(len(points_list_by_time)):
        print("speed:")
        speed = calculate_speed(start_point=np.concatenate((points_list_by_time[i][0], np.array([1]))),
                                end_point=np.concatenate((points_list_by_time[i][-1], np.array([1]))),
                                vanishing_point=np.array([762.764, 4.05, 1]),
                                rear_marker=np.array([569.25, 676.25, 1]),
                                front_marker=np.array([715.8028, 159.6927, 1]),
                                markers_distance=52,
                                time_delta=len(points_list_by_time[i]) * 1 / 30.0)

        cv2.putText(img=image, text=str(speed * 3.6)[:4] + " km/h",
                org=(int(points_list_by_time[i][-1][0] - 75), int(points_list_by_time[i][-1][1])), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=0.75, color=(0, 255, 255), thickness=1)


@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        draw_bounding_boxes=True,
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix = False, Path(w).suffix.lower()
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in ['.pt', '.onnx', '.tflite', '.pb', ''])  # backend
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
            
            bounding_boxes_list = []
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{' ' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    bounding_boxes_list.append(xyxy)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        if draw_bounding_boxes:
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            process_bounding_boxes(bounding_boxes_list, im0)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


run(weights=MODEL_FILE_PATH, source=INPUT_DIRECTORY_PATH, hide_labels=True, hide_conf=True, nosave=False, view_img=True, draw_bounding_boxes=False)
