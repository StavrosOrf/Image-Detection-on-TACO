import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
import cv2
import time
import imutils
import numpy as np
import pandas as pd
import datetime
import mrcnn.model as modellib
from mrcnn import utils, visualize
from imutils.video import WebcamVideoStream
import random
import taco
from tensorflow.keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
import math

# Root directory of the project
from mrcnn.config import Config

ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith("samples/taco"):
    # Go up two levels to the repo root
    ROOT_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))

# Import Mask RCNN
sys.path.append(ROOT_DIR)

# sys.path.append(os.path.join(ROOT_DIR, "samples/plastic/"))  # To find local version

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "/taco_waste_detection20210513T0230/mask_rcnn_taco_waste_detection_0003.h5")
#COCO 60 
# COCO_MODEL_PATH = "../../logs/taco_waste_detection20210513T2303/mask_rcnn_taco_waste_detection_0030.h5"
#Imagenet 60
# COCO_MODEL_PATH = "../../logs/taco_waste_detection20210513T2132/mask_rcnn_taco_waste_detection_0030.h5"

#coco 28 classes
# COCO_MODEL_PATH = "../../logs/taco_waste_detection20210516T0106/mask_rcnn_taco_waste_detection_0030.h5"
COCO_MODEL_PATH = "../../logs/taco_waste_detection20210518T1948/mask_rcnn_taco_waste_detection_0030.h5"

#imagenet 28 classes
# COCO_MODEL_PATH = "../../logs/taco_waste_detection20210516T0245/mask_rcnn_taco_waste_detection_0030.h5"
#trashnet 28 classes
# COCO_MODEL_PATH = "../../logs/taco_waste_detection20210516T2045/mask_rcnn_taco_waste_detection_0030.h5"
#coco 10
# COCO_MODEL_PATH = "../../logs/taco_waste_detection20210516T2332/mask_rcnn_taco_waste_detection_0030.h5"
#imagenet 10
# COCO_MODEL_PATH = "../../logs/taco_waste_detection20210517T0100/mask_rcnn_taco_waste_detection_0030.h5"
# COCO_MODEL_PATH = "../../logs/taco_waste_detection20210516T2045/mask_rcnn_taco_waste_detection_0030.h5"

#trashnet 28
# COCO_MODEL_PATH = "../../logs/taco_waste_detection20210518T1411/mask_rcnn_taco_waste_detection_0030.h5"
# good ones
# COCO_MODEL_PATH = "../../logs/taco_waste_detection20210518T1555/mask_rcnn_taco_waste_detection_0030.h5"
# COCO_MODEL_PATH = "../../logs/taco_waste_detection20210518T2114/mask_rcnn_taco_waste_detection_0025.h5"

# COCO_MODEL_PATH = "best_model_resnet50_v5_2.h5"
# Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     print("Did not find the weights")
#     utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NAME = "taco_waste_detection"
    NUM_CLASSES = 1 + len(taco.classes)

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = pyplot.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

config = InferenceConfig()
# config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
# COCO_MODEL_PATH = "../../logs/taco_waste_detection20210516T0245/mask_rcnn_taco_waste_detection_0030.h5"
# model.load_weights(COCO_MODEL_PATH, by_name=True)
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = taco.classes
class_names.insert(0,"nothing")
# print(class_names)
# print(len(class_names))
colors = visualize.random_colors(len(class_names))

gentle_grey = (45, 65, 79)
white = (255, 255, 255)

OPTIMIZE_CAM = False
SHOW_FPS = False
SHOW_FPS_WO_COUNTER = True  # faster
PROCESS_IMG = True


import json
with open('test_images.json') as json_file:
    data = json.load(json_file)

# print(data)
data1 = ['batch_1/000008.jpg','batch_1/000031.jpg','batch_1/000016.jpg','batch_10/000018.jpg','batch_2/000051.JPG','batch_8/000014.jpg']
# print(taco.classes)
test_images = ['batch_9/000080.jpg', 'batch_9/000081.jpg', 'batch_9/000082.jpg', 'batch_9/000084.jpg', 'batch_9/000086.jpg', 'batch_9/000088.jpg', 'batch_9/000089.jpg', 'batch_9/000090.jpg', 'batch_9/000091.jpg', 'batch_9/000092.jpg', 'batch_9/000093.jpg', 'batch_9/000094.jpg', 'batch_9/000095.jpg', 'batch_9/000097.jpg', 'batch_9/000099.jpg']
# for i,a in enumerate(test_images):
for i,a in enumerate(data1):
    # lllll = [0,2,3]
    # lllll = [1,5]    
    # if i  not in lllll:
    #     continue
    # if i != 1:
    #     continue
    # print(a)    
    image_path = "./dataset/"+a
    print(image_path)

    if OPTIMIZE_CAM:
        vs = WebcamVideoStream(src=0).start()
    else:
        # vs = cv2.VideoCapture("../../Waste_data1_crop.mp4")
        vs = cv2.VideoCapture(image_path)
        # vs = cv2.VideoCapture("./dataset//batch_12/000008.jpg")
        # vs = cv2.imread("./dataset/batch_2/000000.JPG")
        fps = vs.get(cv2.CAP_PROP_FPS)
        width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if SHOW_FPS:
        fps_caption = "FPS: 0"
        fps_counter = 0
        start_time = time.time()

    SCREEN_NAME = 'Mask RCNN LIVE'
    cv2.namedWindow(SCREEN_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(SCREEN_NAME, cv2.WINDOW_NORMAL, cv2.WINDOW_NORMAL)
    # file_name = "waste_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
    # vwriter = cv2.VideoWriter(
    #     file_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0

    total = 0
    my_list=[]

    # TODO


    while True:
        # Capture frame-by-frame
        if OPTIMIZE_CAM:
            frame = vs.read()
        else:
            grabbed, frame = vs.read()
            if not grabbed:
                break
            total += 1

        if SHOW_FPS_WO_COUNTER:
            start_time = time.time()  # start time of the loop
        
        
        # model.keras_model.summary()

        # for i,layer in enumerate(model.keras_model.layers):
        #     print(i,layer.name)

        # ll = [4,8,11,20,23,20,33,]
        listt = [4,8,40,82,144]
        listt = []
        for i in listt:
        # for i,layer in enumerate(model.keras_model.layers):
            # if i < 198:
            #     continue
            # if "activation" not in layer.name:
            #     continue
            molded_images, image_metas, windows = model.mold_inputs([frame])
            image_shape = molded_images[0].shape
            anchors = model.get_anchors(image_shape)        
            anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)
            
            print(i,model.keras_model.layers[i].output)
            model1 = Model(inputs=model.keras_model.inputs,outputs=model.keras_model.layers[i].output)       
            
          

            output = model1.predict([molded_images, image_metas, anchors])
            print(output.shape)
            if i < 198:                
                # plot all feature maps 
                limit = output.shape[3]
                square = math.floor(math.sqrt(limit))
                ix = 1
                for _ in range(square):
                    for _ in range(square):
                        # specify subplot and turn of axis
                        ax = pyplot.subplot(square, square, ix)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        # plot filter channel in grayscale
                        pyplot.imshow(output[0, :, :, ix-1], cmap='gray')
                        ix += 1
                pyplot.show()
        
        if PROCESS_IMG:
            print("--------------------")
            results = model.detect([frame])
            r = results[0]
            # print(r)
            for c in r['class_ids']:
                print(taco.classes[c])

            # print(r['masks'])
            # print(len( r['masks'][0]))
            # print(len( r['masks'][0][0]))
            # print(len( r['masks'][0][1]))
            # print(len( r['masks'][1]))
            # print(len( r['masks'][2]))
            # Run detection
            print_rois = False
            if print_rois:
                image = frame
                pillar = model.keras_model.get_layer("ROI").output  # node to start searching from

                # TF 1.4 and 1.9 introduce new versions of NMS. Search for all names to support TF 1.3~1.10
                nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression:0")
                if nms_node is None:
                    nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV2:0")
                if nms_node is None: #TF 1.9-1.10
                    nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0")

                rpn = model.run_graph([frame], [
                    ("rpn_class", model.keras_model.get_layer("rpn_class").output),
                    ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
                    ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
                    ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
                    ("post_nms_anchor_ix", nms_node),
                    ("proposals", model.keras_model.get_layer("ROI").output),
                ])

                limit = 100
                sorted_anchor_ids = np.argsort(rpn['rpn_class'][:,:,1].flatten())[::-1]
                # visualize.draw_boxes(frame, boxes=model.anchors[sorted_anchor_ids[:limit]], ax=get_ax())

                limit = 50
                ax = get_ax(1, 2)
                pre_nms_anchors = utils.denorm_boxes(rpn["pre_nms_anchors"][0], image.shape[:2])
                refined_anchors = utils.denorm_boxes(rpn["refined_anchors"][0], image.shape[:2])
                refined_anchors_clipped = utils.denorm_boxes(rpn["refined_anchors_clipped"][0], image.shape[:2])
                # visualize.draw_boxes(image, boxes=pre_nms_anchors[:limit],refined_boxes=refined_anchors[:limit], ax=ax[0])
                # visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[:limit], ax=ax[1])
                
                ixs = rpn["post_nms_anchor_ix"][:limit]
                # visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[ixs], ax=get_ax())

                # Convert back to image coordinates for display
                h, w = config.IMAGE_SHAPE[:2]
                proposals = rpn['proposals'][0, :limit] * np.array([h, w, h, w])
                # visualize.draw_boxes(image, refined_boxes=proposals, ax=get_ax())


                mrcnn = model.run_graph([image], [
                    ("proposals", model.keras_model.get_layer("ROI").output),
                    ("probs", model.keras_model.get_layer("mrcnn_class").output),
                    ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
                    ("masks", model.keras_model.get_layer("mrcnn_mask").output),
                    ("detections", model.keras_model.get_layer("mrcnn_detection").output),
                ])
                
                det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
                det_count = np.where(det_class_ids == 0)[0][0]
                det_class_ids = det_class_ids[:det_count]
                detections = mrcnn['detections'][0, :det_count]

                print("{} detections: {}".format(
                    det_count, np.array(class_names)[det_class_ids]))

                captions = ["{} {:.3f}".format(class_names[int(c)], s) if c > 0 else ""
                            for c, s in zip(detections[:, 4], detections[:, 5])]
                # visualize.draw_boxes(
                #     image, 
                #     refined_boxes=utils.denorm_boxes(detections[:, :4], image.shape[:2]),
                #     visibilities=[2] * len(detections),
                #     captions=captions, title="Detections",
                #     ax=get_ax())

                h, w = config.IMAGE_SHAPE[:2]
                proposals = utils.denorm_boxes(mrcnn["proposals"][0], image.shape[:2])
                # proposals = np.around(mrcnn["proposals"][0] * np.array([h, w, h, w])).astype(np.int32)

                # Class ID, score, and mask per proposal
                roi_class_ids = np.argmax(mrcnn["probs"][0], axis=1)
                roi_scores = mrcnn["probs"][0, np.arange(roi_class_ids.shape[0]), roi_class_ids]
                roi_class_names = np.array(class_names)[roi_class_ids]
                roi_positive_ixs = np.where(roi_class_ids > 0)[0]

                # How many ROIs vs empty rows?
                print("{} Valid proposals out of {}".format(np.sum(np.any(proposals, axis=1)), proposals.shape[0]))
                print("{} Positive ROIs".format(len(roi_positive_ixs)))

                # Class counts
                print(list(zip(*np.unique(roi_class_names, return_counts=True))))
                limit = 200
                ixs = np.random.randint(0, proposals.shape[0], limit)
                captions = ["{} {:.3f}".format(class_names[c], s) if c > 0 else ""
                            for c, s in zip(roi_class_ids[ixs], roi_scores[ixs])]
                
                visualize.draw_boxes(image, boxes=proposals[ixs],
                                     visibilities=np.where(roi_class_ids[ixs] > 0, 2, 1),
                                     captions=captions, title="ROIs Before Refinement")

                roi_bbox_specific = mrcnn["deltas"][0, np.arange(proposals.shape[0]), roi_class_ids]
                print("roi_bbox_specific", roi_bbox_specific)

                # Apply bounding box transformations
                # Shape: [N, (y1, x1, y2, x2)]
                refined_proposals = utils.apply_box_deltas(
                    proposals, roi_bbox_specific * config.BBOX_STD_DEV).astype(np.int32)
                print("refined_proposals", refined_proposals)

                # Show positive proposals
                # ids = np.arange(roi_boxes.shape[0])  # Display all
                limit = 5
                if len(roi_positive_ixs) != 0 :
                    ids = np.random.randint(0, len(roi_positive_ixs), limit)  # Display random sample
                    captions = ["{} {:.3f}".format(class_names[c], s) if c > 0 else ""
                                for c, s in zip(roi_class_ids[roi_positive_ixs][ids], roi_scores[roi_positive_ixs][ids])]
                    visualize.draw_boxes(image, boxes=proposals[roi_positive_ixs][ids],
                                         refined_boxes=refined_proposals[roi_positive_ixs][ids],
                                         visibilities=np.where(roi_class_ids[roi_positive_ixs][ids] > 0, 1, 0),
                                         captions=captions, title="ROIs After Refinement")                                             


                keep = np.where(roi_class_ids > 0)[0]
                keep = np.intersect1d(keep, np.where(roi_scores >= config.DETECTION_MIN_CONFIDENCE)[0])

                pre_nms_boxes = refined_proposals[keep]
                pre_nms_scores = roi_scores[keep]
                pre_nms_class_ids = roi_class_ids[keep]

                nms_keep = []
                for class_id in np.unique(pre_nms_class_ids):
                    # Pick detections of this class
                    ixs = np.where(pre_nms_class_ids == class_id)[0]
                    # Apply NMS
                    class_keep = utils.non_max_suppression(pre_nms_boxes[ixs], 
                                                            pre_nms_scores[ixs],
                                                            config.DETECTION_NMS_THRESHOLD)
                    # Map indicies
                    class_keep = keep[ixs[class_keep]]
                    nms_keep = np.union1d(nms_keep, class_keep)
                    print("{:22}: {} -> {}".format(class_names[class_id][:20], 
                                                   keep[ixs], class_keep))

                keep = np.intersect1d(keep, nms_keep).astype(np.int32)
                ixs = np.arange(len(keep))  # Display all
                # ixs = np.random.randint(0, len(keep), 10)  # Display random sample
                captions = ["{} {:.3f}".format(class_names[c], s) if c > 0 else ""
                            for c, s in zip(roi_class_ids[keep][ixs], roi_scores[keep][ixs])]
                visualize.draw_boxes(
                    image, boxes=proposals[keep][ixs],
                    refined_boxes=refined_proposals[keep][ixs],
                    visibilities=np.where(roi_class_ids[keep][ixs] > 0, 1, 0),
                    captions=captions, title="Detections after NMS")


                mrcnn = model.run_graph([image], [
                    ("detections", model.keras_model.get_layer("mrcnn_detection").output),
                    ("masks", model.keras_model.get_layer("mrcnn_mask").output),
                ])

                # Get detection class IDs. Trim zero padding.
                det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
                det_count = np.where(det_class_ids == 0)[0][0]
                det_class_ids = det_class_ids[:det_count]

                print("{} detections: {}".format(
                    det_count, np.array(class_names)[det_class_ids]))

                det_boxes = utils.denorm_boxes(mrcnn["detections"][0, :, :4], image.shape[:2])
                det_mask_specific = np.array([mrcnn["masks"][0, i, :, :, c] 
                                              for i, c in enumerate(det_class_ids)])
                det_masks = np.array([utils.unmold_mask(m, det_boxes[i], image.shape)
                                      for i, m in enumerate(det_mask_specific)])

                # visualize.display_images(det_mask_specific[:4] * 255, cmap="Blues", interpolation="none")
                visualize.display_images(det_masks[:4] * 255, cmap="Blues", interpolation="none")
            # time when we finish processing for this frame
            new_frame_time = time.time()  # Calculating the fps
            # fps will be number of frame processed in given time frame
            # since their will be most of time error of 0.001 second
            # we will be subtracting it to get more accurate result
            fps = 1/(new_frame_time-prev_frame_time)
            my_list.append(new_frame_time-prev_frame_time)
            # print("time processed:"+str(new_frame_time-prev_frame_time))
            prev_frame_time = new_frame_time
            # converting the fps into integer
            fps = int(fps)
            # print("fps="+str(fps))
            masked_image = visualize.display_instances_10fps(
                frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], colors=colors, real_time=True)

        if PROCESS_IMG:
            s = masked_image
        else:
            s = frame
        # print("Image shape: {1}x{0}".format(s.shape[0], s.shape[1]))

        width = s.shape[1]
        height = s.shape[0]
        top_left_corner = (width-120, height-20)
        bott_right_corner = (width, height)
        top_left_corner_cvtext = (width-80, height-5)



        # vwriter.write(s)
        # s = cv2.resize(s,(1080,1440))

        # used to print cv2 complete image
        cv2.imshow(SCREEN_NAME, s)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    input("Press any button to continue..")


print(pd.DataFrame(np.array(my_list)).describe())

# When everything done, release the capture
if OPTIMIZE_CAM:
    vs.stop()
else:
    vs.release()
cv2.destroyAllWindows()
