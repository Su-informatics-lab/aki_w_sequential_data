import os
from typing import List, Optional, Union, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from itertools import starmap



from transformers import PatchTSTConfig, PatchTSTForClassification, Trainer, TrainingArguments


def is_cols_in_df(df, cols):
    """
    Check if all columns in `cols` are present in DataFrame `df`.
    Returns a tuple (True, []) if all are present, or (False, missing_cols).
    """
    missing = [col for col in cols if col not in df.columns]
    return (len(missing) == 0, missing)



class BaseDFDataset(torch.utils.data.Dataset):
    """Base dataset for time series models built upon a pandas dataframe

    Args:
        data_df (pd.DataFrame): Underlying pandas dataframe.
        id_columns (List[str], optional): List of columns which contain id information to separate distinct time series. Defaults to [].
        timestamp_column (Optional[str], optional): Name of the timestamp column. Defaults to None.
        group_id (Optional[Union[List[int], List[str]]], optional): _description_. Defaults to None.
        x_cols (list, optional): Columns to treat as inputs. If an empty list ([]) all the columns in the data_df are taken, except the timestamp column. Defaults to [].
        y_cols (list, optional): Columns to treat as outputs. Defaults to [].
        drop_cols (list, optional): List of columns that are dropped to form the X matrix (input). Defaults to [].
        context_length (int, optional): Length of historical data used when creating individual examples in the torch dataset. Defaults to 1.
        prediction_length (int, optional): Length of prediction (future values). Defaults to 0.
        zero_padding (bool, optional): If True, windows of context_length+prediction_length which are too short are padded with zeros. Defaults to True.
        stride (int, optional): Stride at which windows are produced. Defaults to 1.
        fill_value (Union[float, int], optional): Value used to fill any missing values. Defaults to 0.0.
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        id_columns: List[str] = [],
        timestamp_column: Optional[str] = None,
        group_id: Optional[Union[List[int], List[str]]] = None,
        x_cols: list = [],
        y_cols: list = [],
        drop_cols: list = [],
        context_length: int = 1,
        prediction_length: int = 0,
        zero_padding: bool = True,
        stride: int = 1,
        fill_value: Union[float, int] = 0.0,
    ):
        super().__init__()
        if not isinstance(x_cols, list):
            x_cols = [x_cols]
        if not isinstance(y_cols, list):
            y_cols = [y_cols]

        if len(x_cols) > 0:
            there, missing = is_cols_in_df(data_df, x_cols)
            assert there, f"{missing} given in {x_cols} is not a valid column identifier in the data."

        if len(y_cols) > 0:
            there, missing = is_cols_in_df(data_df, y_cols)
            assert there, f"{missing} given in {y_cols} is not a valid column identifier in the data."

        if timestamp_column:
            assert timestamp_column in list(
                data_df.columns
            ), f"{timestamp_column} is not in the list of data column names provided {data_df.columns}"
            assert (
                timestamp_column not in x_cols
            ), f"{timestamp_column} can not be used as a timestamp column as it also appears in provided collection:{x_cols}."

        self.data_df = data_df
        self.datetime_col = timestamp_column
        self.id_columns = id_columns
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.drop_cols = drop_cols
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.zero_padding = zero_padding
        self.fill_value = fill_value
        self.timestamps = None
        self.group_id = group_id
        self.stride = stride

        # sort the data by datetime
        if timestamp_column in list(data_df.columns):
            # if not isinstance(data_df[timestamp_column].iloc[0], pd.Timestamp):
            #     data_df[timestamp_column] = pd.to_datetime(data_df[timestamp_column])
            data_df = data_df.sort_values(timestamp_column, ignore_index=True)

        # pad zero to the data_df if the len is shorter than seq_len+pred_len
        if zero_padding:
            data_df = self.pad_zero(data_df)
        elif len(data_df) < self.context_length + self.prediction_length:
            LOGGER.warning(
                f"Padding is disabled and input data is shorter than required length. Received {len(data_df)} time point, but require at least {self.context_length + self.prediction_length} time points."
            )

        if timestamp_column in list(data_df.columns):
            self.timestamps = data_df[timestamp_column].to_list()  # .values coerces timestamps
        # get the input data
        if len(x_cols) > 0:
            self.X = data_df[x_cols]
        else:
            drop_cols = self.drop_cols + y_cols
            if timestamp_column:
                drop_cols += [timestamp_column]
            self.X = data_df.drop(drop_cols, axis=1) if len(drop_cols) > 0 else data_df
            self.x_cols = list(self.X.columns)

        # get target data
        if len(y_cols) > 0:
            self.y = data_df[y_cols]
        else:
            self.y = None

        # get number of X variables
        self.n_vars = self.X.shape[1]
        # get number of target
        self.n_targets = len(y_cols) if len(y_cols) > 0 else 0

    def pad_zero(self, data_df):
        # return zero_padding_to_df(data_df, self.seq_len + self.pred_len)
        return ts_padding(
            data_df,
            timestamp_column=self.datetime_col,
            id_columns=self.id_columns,
            context_length=self.context_length + self.prediction_length,
        )

    def __len__(self):
        return max((len(self.X) - self.context_length - self.prediction_length) // self.stride + 1, 0)

    def _check_index(self, index: int) -> int:
        if index >= len(self):
            raise IndexError("Index exceeds dataset length")

        if index < 0:
            if -index > len(self):
                raise ValueError("Absolute value of index should not exceed dataset length")
            index = len(self) + index
        return index

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        raise NotImplementedError

class BaseConcatDFDataset(torch.utils.data.ConcatDataset):
    """A dataset consisting of a concatenation of other datasets, based on torch ConcatDataset.

    Args:
        data_df (pd.DataFrame): Underlying pandas dataframe.
        id_columns (List[str], optional): List of columns which contain id information to separate distinct time series. Defaults to [].
        timestamp_column (Optional[str], optional): Name of the timestamp column. Defaults to None.
        context_length (int, optional): Length of historical data used when creating individual examples in the torch dataset. Defaults to 1.
        prediction_length (int, optional): Length of prediction (future values). Defaults to 0.
        num_workers (int, optional): (Currently not used) Number of workers. Defaults to 1.
        fill_value (Union[float, int], optional): Value used to fill any missing values. Defaults to 0.0.
        cls (_type_, optional): The dataset class used to create the underlying datasets. Defaults to BaseDFDataset.
        stride (int, optional): Stride at which windows are produced. Defaults to 1.
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        id_columns: List[str] = [],
        timestamp_column: Optional[str] = None,
        context_length: int = 1,
        prediction_length: int = 1,
        num_workers: int = 1,
        fill_value: Union[float, int] = 0.0,
        cls=BaseDFDataset,
        stride: int = 1,
        **kwargs,
    ):
        if len(id_columns) > 0:
            there, missing = is_cols_in_df(data_df, id_columns)
            assert there, f"{missing} given in {id_columns} is not a valid column in the data."

        self.timestamp_column = timestamp_column
        self.id_columns = id_columns
        # self.x_cols = x_cols
        # self.y_cols = y_cols
        self.context_length = context_length
        self.num_workers = num_workers
        self.cls = cls
        self.prediction_length = prediction_length
        self.stride = stride
        self.extra_kwargs = kwargs
        self.fill_value = fill_value
        self.cls = cls

        # create groupby object
        if len(id_columns) == 1:
            self.group_df = data_df.groupby(by=self.id_columns[0])
        elif len(id_columns) > 1:
            self.group_df = data_df.groupby(by=self.id_columns)
        else:
            data_df["group"] = 0  # create a artificial group
            self.group_df = data_df.groupby(by="group")

        # add group_ids to the drop_cols
        self.drop_cols = id_columns if len(id_columns) > 0 else ["group"]

        self.group_names = list(self.group_df.groups.keys())
        datasets = self.concat_dataset()
        super().__init__(datasets)
        self.n_vars = self.datasets[0].n_vars
        self.n_targets = self.datasets[0].n_targets

    def concat_dataset(self):
        """Create a list of Datasets

        Returns:
            List of datasets
        """
        group_df = self.group_df
        # print(f'group_df: {group_df}')
        # pool = mp.Pool(self.num_workers)
        # pool.starmap(
        list_dset = starmap(
            get_group_data,
            [
                (
                    self.cls,
                    group,
                    group_id,
                    self.id_columns,
                    self.timestamp_column,
                    self.context_length,
                    self.prediction_length,
                    self.drop_cols,
                    self.stride,
                    self.fill_value,
                    self.extra_kwargs,
                )
                for group_id, group in group_df
            ],
        )

        # pool.close()
        # del group_df
        return list_dset


def get_group_data(
    cls,
    group,
    group_id,
    id_columns: List[str] = [],
    timestamp_column: Optional[str] = None,
    context_length: int = 1,
    prediction_length: int = 1,
    drop_cols: Optional[List[str]] = None,
    stride: int = 1,
    fill_value: Union[float, int] = 0.0,
    extra_kwargs: Dict[str, Any] = {},
):
    return cls(
        data_df=group,
        group_id=group_id if isinstance(group_id, tuple) else (group_id,),
        id_columns=id_columns,
        timestamp_column=timestamp_column,
        context_length=context_length,
        prediction_length=prediction_length,
        drop_cols=drop_cols,
        stride=stride,
        fill_value=fill_value,
        **extra_kwargs,
    )

class ForecastDFDataset(BaseConcatDFDataset):
    """A dataset used for forecasting pretraing and inference

    Args:
        data (pd.DataFrame): Underlying pandas dataframe.
        id_columns (List[str], optional): List of columns which contain id information to separate distinct time series. Defaults
            to [].
        timestamp_column (Optional[str], optional): Name of the timestamp column. Defaults to None.
        target_columns (List[str], optional): List of column names which identify the target channels in the input, these are the
            columns that will be predicted. Defaults to [].
        observable_columns (List[str], optional): List of column names which identify the observable channels in the input.
            Observable channels are channels which we have knowledge about in the past and future. For example, weather
            conditions such as temperature or precipitation may be known or estimated in the future, but cannot be
            changed. Defaults to [].
        control_columns (List[str], optional): List of column names which identify the control channels in the input. Control
            channels are similar to observable channels, except that future values may be controlled. For example, discount
            percentage of a particular product is known and controllable in the future. Defaults to [].
        conditional_columns (List[str], optional): List of column names which identify the conditional channels in the input.
            Conditional channels are channels which we know in the past, but do not know in the future. Defaults to [].
        categorical_columns (List[str]): List of column names which identify time-varying categorical-valued channels in the input.
            Defaults to [].
        static_categorical_columns (List[str], optional): List of column names which identify categorical-valued channels in the
            input which are fixed over time. Defaults to [].
        context_length (int, optional): Length of historical data used when creating individual examples in the torch dataset.
            Defaults to 1.
        prediction_length (int, optional): Length of the future forecast. Defaults to 1.
        num_workers (int, optional): (Currently not used) Number of workers. Defaults to 1.
        frequency_token (Optional[int], optional): An integer representing the frequency of the data. Please see for an example of
            frequency token mappings. Defaults to None.
        autoregressive_modeling (bool, optional): (Experimental) If False, any target values in the context window are masked and
            replaced by 0. If True, the context window contains all the historical target information. Defaults to True.
        stride (int, optional): Stride at which windows are produced. Defaults to 1.
        fill_value (Union[float, int], optional): Value used to fill any missing values. Defaults to 0.0.
        masking_specification (List[Tuple[str, Union[int, Tuple[int, int]]]], optional): Allow masking the history (past values) of
            specific columns. The complete specification is a list of individual column masking specifications. A column masking
            specification is a 2-tuple where the first index specifies a column name. The second index specifies and index/indices to
            mask in each of the the context windows (past_values) generated for that column. If a single index is provided, masking
            will begin at that index and continue to the end of the context window. If a tuple of two values is provded, they are
            treated as python list indices; the values given by these indices will be masked.
        enable_padding (bool, optional): If True, windows of context_length+prediction_length which are too short are padded with zeros. Defaults to True.
            If False, a warning is issued when the input data does not contain sufficient records to create a non-empty dataset.

    The resulting dataset returns records (dictionaries) containing:
        past_values: tensor of past values of the target columns of length equal to context length (context_length x number of features)
        past_observed_mask: tensor indicating which values are observed in the past values tensor (context_length x number of features)
        future_values: tensor of future values of the target columns of length equal to prediction length (prediction_length x number of features)
        future_observed_mask: tensor indicating which values are observed in the future values tensor (prediction_length x number of features)
        freq_token: tensor containing the frequency token (scalar)
        static_categorical_features: tensor of static categorical features (1 x len(static_categorical_columns))
        timestamp: the timestamp of the end of the context window
        id: a tuple of id values (taken from the id columns) containing the id information of the time series segment

        where number of features is the total number of columns specified in target_columns, observable_columns, control_columns,
            conditional_columns
    """

    def __init__(
        self,
        data: pd.DataFrame,
        id_columns: List[str] = [],
        timestamp_column: Optional[str] = None,
        target_columns: List[str] = [],
        observable_columns: List[str] = [],
        control_columns: List[str] = [],
        conditional_columns: List[str] = [],
        categorical_columns: List[str] = [],
        static_categorical_columns: List[str] = [],
        context_length: int = 1,
        prediction_length: int = 1,
        num_workers: int = 1,
        frequency_token: Optional[int] = None,
        autoregressive_modeling: bool = True,
        stride: int = 1,
        fill_value: Union[float, int] = 0.0,
        masking_specification: Optional[List[Tuple[str, Union[int, Tuple[int, int]]]]] = None,
        enable_padding: bool = True,
        metadata_columns: List[str] = [],
    ):
        # output_columns_tmp = input_columns if output_columns == [] else output_columns

        super().__init__(
            data_df=data,
            id_columns=id_columns,
            timestamp_column=timestamp_column,
            num_workers=num_workers,
            context_length=context_length,
            prediction_length=prediction_length,
            fill_value=fill_value,
            cls=self.BaseForecastDFDataset,
            stride=stride,
            enable_padding=enable_padding,
            # extra_args
            target_columns=target_columns,
            observable_columns=observable_columns,
            control_columns=control_columns,
            conditional_columns=conditional_columns,
            categorical_columns=categorical_columns,
            static_categorical_columns=static_categorical_columns,
            frequency_token=frequency_token,
            autoregressive_modeling=autoregressive_modeling,
            masking_specification=masking_specification,
            metadata_columns=metadata_columns,
        )
        self.n_inp = 2
        # for forecasting, the number of targets is the same as number of X variables
        self.n_targets = self.n_vars

    class BaseForecastDFDataset(BaseDFDataset):
        """
        X_{t+1,..., t+p} = f(X_{:t})
        """

        def __init__(
            self,
            data_df: pd.DataFrame,
            group_id: Optional[Union[List[int], List[str]]] = None,
            context_length: int = 1,
            prediction_length: int = 1,
            drop_cols: list = [],
            id_columns: List[str] = [],
            timestamp_column: Optional[str] = None,
            target_columns: List[str] = [],
            observable_columns: List[str] = [],
            control_columns: List[str] = [],
            conditional_columns: List[str] = [],
            categorical_columns: List[str] = [],
            static_categorical_columns: List[str] = [],
            frequency_token: Optional[int] = None,
            autoregressive_modeling: bool = True,
            stride: int = 1,
            fill_value: Union[float, int] = 0.0,
            masking_specification: Optional[List[Tuple[str, Union[int, Tuple[int, int]]]]] = None,
            enable_padding: bool = True,
            metadata_columns: List[str] = [],
        ):
            self.frequency_token = frequency_token
            self.target_columns = target_columns
            self.observable_columns = observable_columns
            self.control_columns = control_columns
            self.conditional_columns = conditional_columns
            self.categorical_columns = categorical_columns
            self.static_categorical_columns = static_categorical_columns
            self.autoregressive_modeling = autoregressive_modeling
            self.masking_specification = masking_specification
            self.metadata_columns = metadata_columns

            x_cols = join_list_without_repeat(
                target_columns,
                observable_columns,
                control_columns,
                conditional_columns,
                categorical_columns,
            )
            y_cols = copy.copy(x_cols)

            self.column_name_to_index_map = {k: v for v, k in enumerate(x_cols)}

            # check non-autoregressive case
            if len(target_columns) == len(x_cols) and not self.autoregressive_modeling:
                raise ValueError(
                    "Non-autoregressive modeling was chosen, but there are no input columns for prediction."
                )

            # masking for conditional values which are not observed during future period
            self.y_mask_conditional = np.array([(c in conditional_columns) for c in y_cols])

            # create a mask of x which masks targets
            self.x_mask_targets = np.array([(c in target_columns) for c in x_cols])

            super().__init__(
                data_df=data_df,
                id_columns=id_columns,
                timestamp_column=timestamp_column,
                x_cols=x_cols,
                y_cols=y_cols,
                context_length=context_length,
                prediction_length=prediction_length,
                group_id=group_id,
                drop_cols=drop_cols,
                stride=stride,
                fill_value=fill_value,
                zero_padding=enable_padding,
            )

        def apply_masking_specification(self, past_values_tensor: np.ndarray) -> np.ndarray:
            """Apply the desired mask defined by masking_specification.

            Args:
                past_values_tensor (np.ndarray): Tensor of past values, should have shape (context_length, num_channels)

            Returns:
                np.ndarry: Tensor with values masked
            """

            for col_name, spec in self.masking_specification:
                col_idx = self.column_name_to_index_map[col_name]
                if isinstance(spec, (tuple, list)) and len(spec) == 2:
                    past_values_tensor[spec[0] : spec[1], col_idx] = np.nan
                else:
                    past_values_tensor[spec:, col_idx] = np.nan
            return past_values_tensor

        def __getitem__(self, index):
            # seq_x: batch_size x seq_len x num_x_cols
            index = self._check_index(index)

            time_id = index * self.stride

            seq_x = self.X.iloc[time_id : time_id + self.context_length].values.astype(np.float32)
            if not self.autoregressive_modeling:
                seq_x[:, self.x_mask_targets] = 0

            if self.masking_specification is not None:
                seq_x = self.apply_masking_specification(seq_x)

            # seq_y: batch_size x pred_len x num_x_cols
            seq_y = self.y.iloc[
                time_id + self.context_length : time_id + self.context_length + self.prediction_length
            ].values
            seq_y[:, self.y_mask_conditional] = 0

            ret = {
                "past_values": np_to_torch(np.nan_to_num(seq_x, nan=self.fill_value)),
                "future_values": np_to_torch(np.nan_to_num(seq_y, nan=self.fill_value)),
                "past_observed_mask": np_to_torch(~np.isnan(seq_x)),
                "future_observed_mask": np_to_torch(~np.isnan(seq_y)),
            }

            if self.datetime_col:
                ret["timestamp"] = self.timestamps[time_id + self.context_length - 1]

            if self.group_id:
                ret["id"] = self.group_id

            if self.frequency_token is not None:
                ret["freq_token"] = torch.tensor(self.frequency_token, dtype=torch.int)

            if self.static_categorical_columns:
                categorical_values = self.data_df[self.static_categorical_columns].values[0, :]
                ret["static_categorical_values"] = np_to_torch(categorical_values)

            if self.metadata_columns:
                ret["metadata"] = self.data_df[self.metadata_columns].values[
                    time_id : time_id + self.context_length + self.prediction_length, :
                ]

            return ret

        def __len__(self):
            return max((len(self.X) - self.context_length - self.prediction_length) // self.stride + 1, 0)

