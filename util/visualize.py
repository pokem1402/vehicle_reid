import matplotlib.pyplot as plt
import cv2
import os

def show_sample_images(df_sample, dataset_path, title):
    h, w = 2, 5

    fig, axs = plt.subplots(h, w, figsize=(15, 7))
    idx = 0
    for _, row in df_sample.iterrows():

        file_path = os.path.join(dataset_path, row["file_name"])
        car_img_bgr = cv2.imread(file_path)
        car_img_rgb = cv2.cvtColor(car_img_bgr, cv2.COLOR_BGR2RGB)

        axs[idx//w, idx % w].imshow(car_img_rgb)
        axs[idx//w, idx % w].set_title(row['file_name'])
        axs[idx//w, idx % w].set_xticks([])
        axs[idx//w, idx % w].set_yticks([])
        idx += 1

    fig.tight_layout()
    plt.suptitle(title)
    plt.show()
