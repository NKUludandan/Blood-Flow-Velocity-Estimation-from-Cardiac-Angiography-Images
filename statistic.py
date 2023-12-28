import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from PIL import Image
from data_loading import get_img_list

def get_intensity(img_paths):
    ves = []
    for img_path in img_paths:
        img = Image.open(img_path)
        img = np.array(img)
        count = np.count_nonzero(img)
        ves.append(count/img.size)

    return ves


img_list7_orig = get_img_list(".\AngioData\AngioData\ImsegmentedPt_02 V_7\Orig")
img_list8_orig = get_img_list(".\AngioData\AngioData\ImsegmentedPt_02 V_8\Orig")

img_list0_seg = get_img_list(".\AngioData\AngioData\ImsegmentedPt_02 V_0\Segmented2\RegUnSharpenedB")
img_list1_seg = get_img_list(".\AngioData\AngioData\ImsegmentedPt_02 V_1\Segmented2\RegUnSharpenedB")
img_list3_seg = get_img_list(".\AngioData\AngioData\ImsegmentedPt_02 V_3\Segmented2\RegUnSharpenedB")
img_list4_seg = get_img_list(".\AngioData\AngioData\ImsegmentedPt_02 V_4\Segmented2\RegUnSharpenedB")
img_list6_seg = get_img_list(".\AngioData\AngioData\ImsegmentedPt_02 V_6\Segmented2\RegUnSharpenedB")
img_list7_seg = get_img_list(".\AngioData\AngioData\ImsegmentedPt_02 V_7\Segmented\Corrected")
img_list8_seg = get_img_list(".\AngioData\AngioData\ImsegmentedPt_02 V_8\Segmented\Corrected")

img_list7_sift = get_img_list(".\AngioData\AngioData\ImsegmentedPt_02 V_7\RegisteredSIFT")
img_list8_sift = get_img_list(".\AngioData\AngioData\ImsegmentedPt_02 V_8\RegisteredSIFT")

first_img_list = [img_list0_seg[0],img_list1_seg[0],img_list3_seg[0],img_list4_seg[0],img_list6_seg[0],img_list7_seg[0],img_list8_seg[0]]
last_img_list = [img_list0_seg[-1],img_list1_seg[-1],img_list3_seg[-1],img_list4_seg[-1],img_list6_seg[-1],img_list7_seg[-1],img_list8_seg[-2]]
ax_label = ['set00','set01','set03','set04','set06','set07','set08']

# The percentage of pixels segmented as vessels in the image
distribution_first = get_intensity(first_img_list)
sns.barplot(x=ax_label, y=distribution_first, color="#6495ED")
plt.xlabel('Set')
plt.ylabel('Percentage')
plt.show()


distribution_last = get_intensity(last_img_list)
sns.barplot(x=ax_label, y=distribution_last, color="#6495ED")
plt.xlabel('Set')
plt.ylabel('Percentage')
plt.show()


distribution0 = get_intensity(img_list0_seg)
distribution1 = get_intensity(img_list1_seg)
distribution3 = get_intensity(img_list3_seg)
distribution4 = get_intensity(img_list4_seg)
distribution6 = get_intensity(img_list6_seg)
distribution7 = get_intensity(img_list7_seg)
distribution8 = get_intensity(img_list8_seg)

plt.figure(figsize=(12, 6))
sns.lineplot(x=np.array(range(1,len(distribution0)+1)), y=distribution0, label='set01')
sns.lineplot(x=np.array(range(1,len(distribution1)+1)), y=distribution1, label='set01')
sns.lineplot(x=np.array(range(1,len(distribution3)+1)), y=distribution3, label='set03')
sns.lineplot(x=np.array(range(1,len(distribution4)+1)), y=distribution4, label='set04')
sns.lineplot(x=np.array(range(1,len(distribution6)+1)), y=distribution6, label='set06')
sns.lineplot(x=np.array(range(1,len(distribution7)+1)), y=distribution7, label='set07')
sns.lineplot(x=np.array(range(1,len(distribution8)+1)), y=distribution8, label='set08')
plt.xlabel('Image number')
plt.ylabel('Percentage')
plt.show()


def sigmoid(x, a, b, c, d):
    return d / (1 + np.exp(-(a*x+b))) + c


x_data = np.array(range(1,len(distribution3)+1))
y_data = [i for i in distribution3]
params, covariance = curve_fit(sigmoid, x_data, y_data)
a, b, c, d = params
x_fit = np.linspace(1, 48, 100)
y_fit = sigmoid(x_fit, a, b, c, d)
plt.figure(figsize=(12, 6))
sns.lineplot(x=np.array(range(1,len(distribution3)+1)), y=distribution3, label='set03', color='green')
sns.lineplot(x=x_fit, y=y_fit, label='fit curve',color='red')
plt.legend()
plt.xlabel('Image number')
plt.ylabel('Percentage')
plt.show()




x_data = np.array(range(1,len(distribution7)+1))
y_data = [i for i in distribution7]
initial_guess = [a, b, c, d]
params, covariance = curve_fit(sigmoid, x_data, y_data, p0=initial_guess)
a, b, c, d = params
x_fit = np.linspace(0, 58, 100)
y_fit = sigmoid(x_fit, a, b, c, d)
plt.figure(figsize=(12, 6))
sns.lineplot(x=np.array(range(1,len(distribution7)+1)), y=distribution7, label='set07', color='brown')
sns.lineplot(x=x_fit, y=y_fit, label='fit curve',color='red')
plt.legend()
plt.xlabel('Image number')
plt.ylabel('Percentage')
plt.show()