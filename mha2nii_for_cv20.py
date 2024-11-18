import itk
import glob
import os
import time
time_start = time.time()
#将预测的mha文件转为nii文件方便计算20个交叉验证的指标，
import numpy as np
def reorient_to_reference(image, reference):
    filter = itk.OrientImageFilter[type(image), type(image)].New(image)
    filter.UseImageDirectionOn()
    filter.SetInput(image)
    filter.SetDesiredCoordinateDirection(reference.GetDirection())
    filter.Update()
    return filter.GetOutput()


def cast(image, reference):
    filter = itk.CastImageFilter[type(image), type(reference)].New()
    filter.SetInput(image)
    filter.Update()
    return filter.GetOutput()


def copy_information(image, reference):
    filter = itk.ChangeInformationImageFilter[type(image)].New()
    filter.SetInput(image)
    filter.SetReferenceImage(reference)
    filter.UseReferenceImageOn()
    filter.ChangeSpacingOn()
    filter.ChangeOriginOn()
    filter.ChangeDirectionOn()
    filter.Update()
    return filter.GetOutput()


def relabel(labels):
    labels_np = itk.GetArrayViewFromImage(labels)
    from_labels = [1, 2, 3, 4, 5, 6, 7]
    to_labels = [500, 600, 420, 550, 205, 820, 850]
    for from_label, to_label in zip(from_labels, to_labels):
        labels_np[labels_np == from_label] = to_label


if __name__ == '__main__':
    # TODO: set to True for CT and False for MR
    is_ct = False#True False
    # TODO: change folder to where the predictions are saved
    input_folder = r'E:\D6_MMWHS_2017\MedicalDataAugmentationTool-MMWHS\output\cross_validation\unet_mr_3\2022-03-17_19-24-01\iter_25000/' #UNet  CT  MR
    # TODO: change folder to where original files (e.g. mr_train_1001_image.nii.gz and mr_train_1001_label.nii.gz) from MMWHS challenge are saved
    reference_folder = 'mmwhs_dataset/ct_nii/' if is_ct else 'mmwhs_dataset/mr_nii/'

    output_folder = 'E:/D6_MMWHS_2017/Cross_Validation20/3DUNet/CT/' if is_ct else 'E:/D6_MMWHS_2017/Cross_Validation20/3DUNet/MR/' #MM_WHS_GUT， MM_WHS_UNet
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filenames = glob.glob(input_folder + '*.mha')
    for filename in sorted(filenames):
        if 'prediction' in filename:#跳过文件名带有prediction的文件
            continue
        basename = os.path.basename(filename)
        basename_wo_ext = basename[:basename.find('.mha')]
        print(basename_wo_ext)
        image = itk.imread(filename)

        reference = itk.imread(os.path.join(reference_folder, basename_wo_ext + '_image.nii.gz'))
        reoriented = cast(image, reference)
        reoriented = reorient_to_reference(reoriented, reference)
        reoriented = copy_information(reoriented, reference)
        relabel(reoriented)
        itk.imwrite(reoriented, os.path.join(output_folder, basename_wo_ext + '_label.nii.gz'))


time_end = time.time()
print('totally cost', time_end - time_start)