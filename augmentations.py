from matplotlib import pyplot as plt
from clodsa.augmentors.augmentorFactory import createAugmentor
from clodsa.transformers.transformerFactory import transformerGenerator
from clodsa.techniques.techniqueFactory import createTechnique
import cv2

PROBLEM = "semantic_segmentation"
ANNOTATION_MODE = "folders"
INPUT_PATH = "DATASET"
GENERATION_MODE = "linear"
OUTPUT_MODE = "folders"
OUTPUT_PATH= "augmented_images/"
LABELS_EXTENSION = ".png"
augmentor = createAugmentor(PROBLEM,ANNOTATION_MODE,OUTPUT_MODE,GENERATION_MODE,INPUT_PATH,{"outputPath":OUTPUT_PATH,"labelsExtension":LABELS_EXTENSION})
transformer = transformerGenerator(PROBLEM)

#rotating
rotateRandom = createTechnique("rotate",{})
augmentor.addTransformer(transformer(rotateRandom))
for angle in [90,180,270]:
    rotate = createTechnique("rotate", {"angle" : angle})
    augmentor.addTransformer(transformer(rotate))

#shifting
translation = createTechnique("translation", {"x":15,"y":-5})
augmentor.addTransformer(transformer(translation))

#elastic deformation
elastic = createTechnique("elastic",{"alpha":5,"sigma":0.05})
augmentor.addTransformer(transformer(elastic))

#gamma correction
gamma = createTechnique("gamma",{"gamma":1.5})
augmentor.addTransformer(transformer(gamma))

augmentor.applyAugmentation()
# MOZE se pojaviti problem ovde sa fajlom koji ucitava slike u sklopu biblioteke clodsa
# treba izmeniti labelPath koji sam na brzinu izmenio da bih prilagodio