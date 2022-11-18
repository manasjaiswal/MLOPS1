if self.columns is not None:
                item_visibility_ix=self.columns.index(COLUMN_ITEM_VISIBILITY)
                item_type_ix=self.columns.index(COLUMN_ITEM_TYPE)
                item_weight_ix=self.columns.index(COLUMN_ITEM_WEIGHT)
                outlet_size_ix=self.columns.index(COLUMN_ITEM_WEIGHT)
            self.item_visibility_ix=item_visibility_ix
            self.item_type_ix=item_type_ix    
            self.item_weight_ix=item_weight_ix
            self.outlet_size_ix=outlet_size_ix



 visibility_arr=X[:,self.item_visibility_ix]
            item_type_arr=X[:,self.item_type_ix]
            item_weight_arr=X[:,self.item_weight_ix]
            outlet_size_arr=X[:,self.outlet_size_ix]
            data=np.c_(visibility_arr,item_type_arr,item_weight_arr,outlet_size_arr)            