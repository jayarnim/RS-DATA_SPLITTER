import random
import itertools
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from ..utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)


class PointwiseNegativeSamplingDataset(Dataset):
    def __init__(
        self, 
        origin: pd.DataFrame, 
        neg_items_per_user: dict,
        neg_per_pos: int,
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
    ):
        self.neg_items_per_user = neg_items_per_user
        self.neg_per_pos = neg_per_pos
        self.col_user = col_user
        self.col_item = col_item

        zip_obj = zip(origin[self.col_user], origin[self.col_item])
        self.user_item_pairs = list(zip_obj)

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user, pos = self.user_item_pairs[idx]
        
        # negative sampling
        kwargs = dict(
            population=self.neg_items_per_user[user],
            k=self.neg_per_pos,     
        )
        neg_list = random.sample(**kwargs)

        user_list = [user] * (1 + self.neg_per_pos)
        item_list = [pos] + neg_list
        label_list = [1] + [0] * self.neg_per_pos

        return user_list, item_list, label_list


class PointwiseNegativeSamplingDataLoader:
    def __init__(
        self,
        origin: pd.DataFrame,
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
    ):
        self.col_user = col_user
        self.col_item = col_item

        kwargs = dict(
            origin=origin, 
            col_user=self.col_user, 
            col_item=self.col_item,
        )
        self.neg_items_per_user = self._generate_negative_sample_pool(**kwargs)

    def get(
        self, 
        data: pd.DataFrame,
        neg_per_pos: int,
        batch_size: int,
        shuffle: bool=True,
    ):
        kwargs = dict(
            data=data, 
            neg_items_per_user=self.neg_items_per_user,
            neg_per_pos=neg_per_pos,
            col_user=self.col_user, 
            col_item=self.col_item,     
        )
        dataset = PointwiseNegativeSamplingDataset(**kwargs)

        kwargs = dict(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=self._collate,            
        )
        loader = DataLoader(**kwargs)

        return loader

    def _generate_negative_sample_pool(
        self,
        origin: pd.DataFrame, 
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
    ):
        all_users = sorted(origin[col_user].unique())
        all_items = sorted(origin[col_item].unique())
        
        pos_per_user = {
            user: set(origin[origin[col_user] == user][col_item])
            for user in all_users
        }

        neg_items_per_user = {
            user: list(set(all_items) - pos_per_user[user])
            for user in all_users
        }

        return neg_items_per_user

    def _collate(self, batch):
        # batch: ([(u1),(i1*),(y1*)],[(u2),(i2*),(y2*)],...)
        # user_list: [(u1,u1,u1),(u2,u2,u2),...]
        # item_list: [(i11,i12,i13),(i21,i22,i23),...]
        # label_list: [(y11,y12,y13),(y21,y22,y23),...]
        user_list, item_list, label_list = zip(*batch)

        # [(u1,u1,u1),(u2,u2,u2),...] -> [u1,u1,u1,u2,u2,u2,...]
        user_batch = torch.tensor(
            list(itertools.chain.from_iterable(user_list)), 
            dtype=torch.long
        )

        # [(i11,i12,i13),(i21,i22,i23),...] -> [i11,i12,i13,i21,i22,i23,...]
        item_batch = torch.tensor(
            list(itertools.chain.from_iterable(item_list)), 
            dtype=torch.long
        )

        # [(y11,y12,y13),(y21,y22,y23),...] -> [y11,y12,y13,y21,y22,y23,...]
        label_batch = torch.tensor(
            list(itertools.chain.from_iterable(label_list)),
            dtype=torch.float32
        )
        
        return user_batch, item_batch, label_batch