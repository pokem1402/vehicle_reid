import pandas as pd
import numpy as np
import os
import cv2
from util.util import calc_cosign_similarity
from tqdm import tqdm
import onnxruntime as onnx
from functools import lru_cache



class Evaluation:

    SIZE = (256, 256)
    BATCH_SIZE = 128
    
    def __init__(self,
                 weight_file_path,
                 dataset_path,
                 csv_file_path,
                 remove_one_element=True,
                 data_shuffle=False):
        
        # initiate onnx session 
        model_path = "models/"+weight_file_path
        if not os.path.exists(model_path):
            raise ValueError("onnx file does not exists")
        
        provider = ['CUDAExecutionProvider']
        
        self.sess = onnx.InferenceSession(model_path, providers=provider)
        
        # get dataset 
        self.dataset_path = dataset_path
        
        if not os.path.exists(os.path.join(dataset_path, csv_file_path)):
            raise ValueError("CSV FILE does not exists")
        
        df = pd.read_csv(os.path.join(dataset_path,csv_file_path))
        
        # divide dataset into front-side, back-side, side dataset
        self.df = {}
        
        self.df["F"] = df[df.viewpoint == "F"]
        self.df["B"] = df[df.viewpoint == "B"]
        self.df['S'] = df[df.viewpoint == "S"]
        
        # the group that more than one element
        if remove_one_element:
            for k in self.df.keys():
                self.df[k] = self.df[k].groupby('classId').apply(
                    lambda group:group if len(group) > 1 else None
                ).reset_index(drop=True)
                        
        if data_shuffle:
            
            for k in self.df.keys():
                self.df[k] = self.df[k].sample(frac=1).reset_index(drop=True)
            

        
    def normalize(self, nparray, order=2, axis=-1):
        """Normalize a N-D numpy array along the specified axis."""
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)

    def infer(self, image_np):
        input_name = self.sess.get_inputs()[0].name

        feat = self.sess.run(None, {input_name: image_np})[0]
        feat = self.normalize(feat)

        return feat
    
    @lru_cache(maxsize=12000)
    def get_image(self, img_path):
        return cv2.imread(img_path)
    
    def preprocessing(self, image_np, for_batch=False):
        
        # the model expects RGB inputs
        original_image = image_np[:, :, ::-1]

        # Apply pre-processing to image.
        resize_width = self.SIZE[0]
        resize_height = self.SIZE[1]
        img = cv2.resize(original_image, (resize_width,
                         resize_height), interpolation=cv2.INTER_CUBIC)

        if for_batch:
            img = img.astype("float32").transpose(
                2, 0, 1)  # (3, h, w)
        else:
            img = img.astype("float32").transpose(
                2, 0, 1)[None]  # (1, 3, h, w)

        return img
    
   
    @lru_cache(maxsize=12000)
    def get_images_and_preprocess(self, img_path, for_batch=False):
        image_np = cv2.imread(img_path)

        # the model expects RGB inputs
        original_image = image_np[:, :, ::-1]

        # Apply pre-processing to image.
        resize_width = self.SIZE[0]
        resize_height = self.SIZE[1]
        img = cv2.resize(original_image, (resize_width,
                         resize_height), interpolation=cv2.INTER_CUBIC)

        if for_batch:
            img = img.astype("float32").transpose(
                2, 0, 1)  # (3, h, w)
        else:
            img = img.astype("float32").transpose(
                2, 0, 1)[None]  # (1, 3, h, w)

        return img

                
    def evaluation_rank_k(self, side):
        
        queries = self.df[side].groupby("classId").first()

        gallaries = self.df[side].groupby('classId').apply(
            lambda group:group.iloc[1:]
        ).reset_index(drop=True)
        
        rank1, rank5, rank10, i = 0, 0, 0, 0
        pbar = tqdm(queries.iterrows(), total=queries.shape[0])
        for cid, query in pbar:

            #query image
            query_img = self.get_images_and_preprocess(self.dataset_path+query['file_name'])
            query_img_feat = self.infer(query_img)
            #gallary image
            gallary_similarity = []
            for _, gallary in gallaries.groupby("classId").sample(1).iterrows():

            # for _cid, gallary in gallaries.groupby("classId").agg(pd.DataFrame.sample).iterrows():

                gallary_img = self.get_images_and_preprocess(self.dataset_path+gallary['file_name'])
                gallary_img_feat = self.infer(gallary_img)

                similarity = calc_cosign_similarity(query_img_feat, gallary_img_feat)

                gallary_similarity.append(
                    tuple([similarity, gallary['classId']]))

            gallary_similarity.sort(reverse=True)

            rank1 += 1 if list(filter(lambda x: x[1]
                            == cid, gallary_similarity[:1])) else 0
            rank5 += 1 if list(filter(lambda x: x[1]
                            == cid, gallary_similarity[:5])) else 0
            rank10 += 1 if list(filter(lambda x: x[1]
                                == cid, gallary_similarity[:10])) else 0
            i += 1
            
            pbar.set_postfix(
                {'rank1': rank1, 'rank5': rank5, 'rank10': rank10})
            
        print(
            f"rank1 : {rank1}/{i}::{rank1/i}, rank5 : {rank5}/{i}::{rank5/i}, rank10 : {rank10}/{i}::{rank10/i}")

        return (rank1, rank5, rank10, i)

    
    def evaluation_rank_k_by_batch(self, side):
        
        queries = self.df[side].groupby("classId").first()

        gallaries = self.df[side].groupby('classId').apply(
            lambda group:group.iloc[1:]
        ).reset_index(drop=True)
        
        rank1, rank5, rank10, i = 0, 0, 0, 0
        
        pbar = tqdm(queries.iterrows(), total=queries.shape[0])
        for cid, query in pbar:
            
            #query image
            query_img = self.get_image(self.dataset_path+query['file_name'])
            query_img = self.preprocessing(query_img)
            query_img_feat = self.infer(query_img)
            
            #gallary image
            gallary_images = []
            gallary_img_feats = []
            
            batch_i = 0
            clslist = []
            for _, gallary in gallaries.groupby("classId").sample(1).iterrows():

            # for _cid, gallary in gallaries.groupby("classId").agg(pd.DataFrame.sample).iterrows():
                gallary_img = self.get_image(self.dataset_path+gallary['file_name'])
                gallary_img = self.preprocessing(gallary_img, for_batch=True)
                gallary_images.append(gallary_img)
            
                clslist.append(gallary["classId"])

                if (batch_i + 1) % self.BATCH_SIZE == 0: 
                    gal_images_batch = np.stack(gallary_images)
                    gal_img_feat_part = self.infer(gal_images_batch)
                    gallary_img_feats.append(gal_img_feat_part)
                    gallary_images = []
                batch_i+= 1
            else:
                if gallary_images:
                    gal_images_batch = np.stack(gallary_images)
                    gal_img_feat_part = self.infer(gal_images_batch)
                    gallary_img_feats.append(gal_img_feat_part)
            
            gallary_feat = np.concatenate(gallary_img_feats)
            assert gallary_feat.shape == (queries.shape[0], 2048)
            
            similarity = calc_cosign_similarity(query_img_feat, gallary_feat)
            
            sim_order_cid = np.argsort(similarity)[::-1]
            
            clslist = np.array(clslist)[sim_order_cid[:10]]
            
            rank1 += 1 if clslist[0] == cid else 0
            rank5 += 1 if cid in clslist[:5] else 0
            rank10 += 1 if cid in clslist[:10] else 0
            
            i += 1
            pbar.set_postfix({'rank1':rank1, 'rank5':rank5, 'rank10':rank10})
        
        print(f"rank1 : {rank1}/{i}::{rank1/i}, rank5 : {rank5}/{i}::{rank5/i}, rank10 : {rank10}/{i}::{rank10/i}")

        return (rank1, rank5, rank10, i)
        
        