import itk
from glob import glob
import os

#nii转mha格式,训练集的20个图像和标签以及测试集的40个图像（ct和mr）

def reorient_to_rai(image):
    filter = itk.OrientImageFilter.New(image)
    filter.UseImageDirectionOn()
    filter.SetInput(image)
    m = itk.Matrix[itk.D, 3, 3]()
    m.SetIdentity()
    filter.SetDesiredCoordinateDirection(m)
    filter.Update()
    return filter.GetOutput()


def relabel(labels):
    labels_np = itk.GetArrayViewFromImage(labels)
    from_labels = [500, 600, 420, 550, 205, 820, 850]# LV RV LA RA Myo AO PA
    to_labels = [1, 2, 3, 4, 5, 6, 7]
    for from_label, to_label in zip(from_labels, to_labels):
        labels_np[labels_np == from_label] = to_label
    # set everything else to zero
    labels_np[labels_np > 7] = 0


if __name__ == '__main__':
    # TODO: set to True for CT and False for MR
    is_ct = True
    # TODO: change input folder
    # input_folder = r'E:\D6_MMWHS_2017\MedicalDataAugmentationTool-MMWHS\TestReults40\MM_WHS_unet30000\CT/'
    # output_folder = './mmwhs_dataset/ct_mha/' if is_ct else './mmwhs_dataset/mr_mha/'
    input_folder = r'E:\D6_MMWHS_2017\whole-heart-segmentation-master\Data_for_MMWHS\ct_region_image_nii\\'
    output_folder = './mmwhs_dataset_region/ct_mha/' if is_ct else './mmwhs_dataset_region/mr_mha/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filenames = glob(input_folder + '*image.nii.gz')
    for filename in sorted(filenames):
        basename = os.path.basename(filename)
        basename_wo_ext = basename[:basename.find('.nii.gz')]
        print(basename_wo_ext)
        image = itk.imread(filename)
        reoriented = reorient_to_rai(image)
        itk.imwrite(reoriented, output_folder + basename_wo_ext + '.mha')

    # filenames_label = glob(input_folder + '*label.nii.gz')
    # for filename in sorted(filenames_label):
    #     basename = os.path.basename(filename)
    #     basename_wo_ext = basename[:basename.find('.nii.gz')]
    #     print(basename_wo_ext)
    #     image = itk.imread(filename)
    #     reoriented = reorient_to_rai(image)
    #     relabel(reoriented)
    #     itk.imwrite(reoriented, output_folder + basename_wo_ext + '.mha')
