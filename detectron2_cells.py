# -*- coding: utf-8 -*-
# Detectron2 - cells.ipynb

# Original file is located at https://colab.research.google.com/drive/1rCLduoeFC_UKN5MEKF77nvPVkbxOzt1x

# **Usage of Detectron2**

# Install detectron2
# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
import os
from pathlib import Path

setup_logger()

# import some common libraries
import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from pathlib import Path
import os

scratchdir = os.getenv('SCRATCHDIR', ".")
print(Path(scratchdir).exists())
print(Path(scratchdir))

input_data_dir = Path(scratchdir) / 'data/orig/cells'
print(Path(input_data_dir).exists())
print(Path(input_data_dir))

outputdir = Path(scratchdir) / 'data/processed/'
print(Path(outputdir).exists())
print(Path(outputdir))

trainval = Path(scratchdir) / 'data/orig/cells/trainval.json'
print(Path(trainval).exists())
print(Path(trainval))

# Get the list of all files and directories
path = str(input_data_dir)
dir_list = os.listdir(path)

print("Files and directories in '", path, "' :")

# prints all files
print(dir_list)

from detectron2.data.datasets import register_coco_instances

register_coco_instances("cells", {}, str(input_data_dir / "trainval.json"), str(input_data_dir / "images"))
#register_coco_instances("cells_validation", {}, str(input_data_dir / "trainval.json"), str(input_data_dir / "images")) # TODO: Create validation dataset

cells_metadata = MetadataCatalog.get("cells")
dataset_dicts = DatasetCatalog.get("cells")

print()
print(dataset_dicts)
print()

for d in dataset_dicts:
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=cells_metadata, scale=1)
    vis = visualizer.draw_dataset_dict(d)
    (outputdir / "vis_train").mkdir(parents=True, exist_ok=True)
    print(Path(outputdir/"vis_train").exists())
    print(Path((outputdir/"vis_train")))
    img_name_final = os.path.basename(d["file_name"])
    print(str(outputdir) + "/" + "vis_train" + "/" + img_name_final)
    if not cv2.imwrite(str(outputdir) + "/" + "vis_train" + "/" + img_name_final, vis.get_image()[:, :, ::-1]):
        raise Exception("Could not write image: " + img_name_final)



from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()
cfg.merge_from_file("/auto/plzen1/home/jburian/extern/detectron2/configs/COCO-InstanceSegmentation"
                    "/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("cells",)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset, validation dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 class (cells)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

from pathlib import Path

print("Obsah adresare outputdir: " + str(list(Path(outputdir).glob("**/*"))))
print("Obsah adresare inputdir: " + str(list(Path(input_data_dir).glob("**/*"))))

trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
cfg.DATASETS.TEST = ("cells",)
predictor = DefaultPredictor(cfg)

# Prediction in picture
from detectron2.utils.visualizer import ColorMode

# TODO: change source directory of images for predictions
for d in dataset_dicts:
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=cells_metadata,
                   scale=3,
                   instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    (outputdir / "vis_predictions").mkdir(parents=True, exist_ok=True)
    img_name_final = os.path.basename(d["file_name"])
    if not cv2.imwrite(str(outputdir) + "/" + "vis_predictions" + "/" + img_name_final, v.get_image()[:, :, ::-1]):
        raise Exception("Could not write image: " + img_name_final)
