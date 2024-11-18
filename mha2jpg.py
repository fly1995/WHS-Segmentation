from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import glob
import SimpleITK as sitk
from medpy.io import load
import cv2
import scipy.misc
import itk
import scipy.io as io
from PIL import Image
import scipy.io as scio

def normalization(input):
    max = np.max(input)
    min = np.min(input)
    return (input-min)/(max-min)


def mha2jpg(file_dir, save_dir):
    imgs = glob.glob(file_dir+'./*2001_image.mha')
    for imgname in imgs:
        midname = imgname[imgname.rindex("\\") + 1:]
        print(midname)
        img = sitk.ReadImage(file_dir+midname) #读取文件内容
        img_data = sitk.GetArrayFromImage(img) #获取文件中的图片数据(363, 512, 512)
        for i in range(img_data.shape[0]):
            cv2.imwrite(save_dir+midname[0:-4]+"%d.png" % (i+1), normalization(img_data[i, :, :])*255)
            # img = np.expand_dims(normalization(img_data[i, :, :])*255, axis=2)
            # img_save = array_to_img(img)
            # img_save.save(save_dir+midname[0:-4]+"%d.png" % (i+1))

#
# file_dir = r'./mmwhs_dataset/ct_mha/'
# save_dir =r'./mmwhs_dataset/ct_img/'
# mha2jpg(file_dir, save_dir)


def jpg2mha(mha_dir, file_dir, save_dir):
    mhas = glob.glob(mha_dir+'*2001_image.mha')
    for mhaname in mhas:
        midname = mhaname[mhaname.rindex("\\") + 1:]
        im = sitk.ReadImage(mha_dir+midname) #读取文件内容
        img_data = sitk.GetArrayFromImage(im) #获取文件中的图片数据(363, 512, 512)
        print(midname)
        npy_list = []
        for i in range(img_data.shape[0]):
            img = load_img(file_dir + str(midname[0:-4]) + str(i+1) + '.png', grayscale=True)
            img = img_to_array(img) #(512,512,1)
            npy_list.append(img)
        npy = np.concatenate(npy_list, axis=2)#(512, 512, 224)
        print(npy.shape)
        img_save = sitk.GetImageFromArray(npy.transpose(2, 0, 1))
        img_save.SetSpacing(im.GetSpacing())
        img_save.SetOrigin(im.GetOrigin())
        sitk.WriteImage(img_save, save_dir+midname)


# mha_dir =  r'./mmwhs_dataset/ct_mha/'
# file_dir = r'./mmwhs_dataset/ct_decomposition/lae_layer\\' #r'./mmwhs_dataset/ct_img/'
# save_dir =r'./mmwhs_dataset/ct_decomposition/'
# jpg2mha(mha_dir, file_dir, save_dir)

#以下代码是采用mat格式进行图像转换， 因为以上转成jpg后叠加起来会产生横条纹（信息丢失）
def mha2mat(mha_dir, save_dir):#将原始mha转成mat，好在matlab中进行分解
    imgs = glob.glob(mha_dir + './*.mha')
    for imgname in imgs:
        midname = imgname[imgname.rindex("\\") + 1:]
        print(midname)
        img = sitk.ReadImage(mha_dir + midname)  # 读取文件内容
        img_data = sitk.GetArrayFromImage(img)
        io.savemat(save_dir+midname[0:-4]+'.mat', {'data': normalization(img_data.transpose(2, 1, 0))*255})


# mha_dir = r'./mmwhs_dataset/ct_mha/'
# save_dir =r'./mmwhs_dataset/ct_mat/'
# mha2mat(mha_dir, save_dir)



def mat2mha(mha_dir, mat_dir, save_dir):#matlab中将分解后的mat转为mha好放进网络进行训练
    mats = glob.glob(mat_dir + '*.mat')
    for imgname in mats:
        midname = imgname[imgname.rindex("\\") + 1:]
        matdata = scio.loadmat(mat_dir+midname)

        im = sitk.ReadImage(mha_dir + midname.replace('mat', 'mha'))  # 读取mha文件内容

        img_save = sitk.GetImageFromArray(matdata['out'].transpose(2, 1, 0))
        img_save.SetSpacing(im.GetSpacing())
        img_save.SetOrigin(im.GetOrigin())
        sitk.WriteImage(img_save, save_dir+midname.replace('mat', 'mha'))
#
mha_dir = r'./mmwhs_dataset/ct_mha/'
mat_dir = r'./mmwhs_dataset/ct_decomposition/'
save_dir =r'./mmwhs_dataset/mat2mha/'
mat2mha(mha_dir, mat_dir, save_dir)