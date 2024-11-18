import nibabel as nb
import os
import itk
import glob
import SimpleITK as sitk


root= r'E:\D6_MMWHS_2017\MedicalDataAugmentationTool-MMWHS\inference\GUT\iter_0\\'
imgname = glob.glob(root + '/*' + '.nii.gz')
for filename in imgname:
    midname = filename[filename.rindex("\\") + 1:]
    image = nb.load(root+midname).get_fdata()
    image = sitk.GetImageFromArray(image)
    sitk.WriteImage(image, root+midname)



