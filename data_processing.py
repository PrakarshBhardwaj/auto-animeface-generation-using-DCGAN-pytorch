import glob
import h5py
import cv2
import numpy as np

def normalize(img):
    # Converting image in [-1,1] range
    img = img.astype(np.float32)
    img /= 128
    img -= 1
    return img

def denormalize(img):
    img += 1
    img *= 127
    img = img.astype(np.uint8)
    return img

def to_hdf5(data_loc_wildcard , dataset_name , size , show_imgs=False):
    # Store all images as hdf5
    img_loc = glob.glob(data_loc_wildcard)
    h_px , w_px , ch = size

    with h5py.File(dataset_name , "w") as f:
        animefaces = f.create_dataset("animefaces" , (len(img_loc) , h_px , w_px , ch) , dtype="f")
    
        for i,img_name in enumerate(img_loc):
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
            img = cv2.resize(img , (h_px , w_px))
            animefaces[i] = normalize(img)

            if i%500 == 0 and show_imgs:
                print(f"Iter: {i} , Name: {img_name}")
                cv2.imshow(f"Iter: {i} , Name: {img_name}" , img)
                cv2.waitKey(500)
                cv2.destroyAllWindows()

def verify_dataset(dataset , target_shape = (224,224,3)):
    # Verifies the shape of all images in a dataset
    with h5py.File(dataset , "r") as f:
        animeface_data = f.get("animefaces")
        for i,face in enumerate(animeface_data):
            assert face.shape == target_shape , f"Image {i} has invalid shape"
        print("All images verified successfully!")
                
if __name__ == "__main__":
    to_hdf5("./animeface-character-dataset/thumb/*/*.png" , "animeface-data.hdf5" , (64,64,3))
    verify_dataset("animeface-data.hdf5" , target_shape=(64 , 64 , 3))

