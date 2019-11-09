from model import *
from data import *


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myHair = trainGenerator(3,'data/train','image','label',data_gen_args,save_to_dir = False)

model = unet()

model_checkpoint = ModelCheckpoint('unet_hair.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myHair,steps_per_epoch=1000,epochs=20,callbacks=[model_checkpoint])
