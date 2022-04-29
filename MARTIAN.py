import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from datetime import date
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Concatenate
from keras.callbacks import EarlyStopping
from dateutil.relativedelta import relativedelta
from myutils import data_preparation, data_normalization

census_tract_list = pd.read_csv('census_names.csv')['NAME'].tolist()
static_feature_list = ['S2502_C05_001E', 'S2502_C05_018E', 'S2502_C05_019E', 'S2502_C05_020E',
                       'S2502_C05_021E','S2503_C05_028E', 'S2503_C05_045E', 'S1702_C02_018E']


def data_denormalization(y_min, y_max, prediction):
    denominator = y_max - y_min
    prediction = (prediction * denominator) + y_min
    prediction = np.reshape(prediction, (prediction.shape[0], 1))
    return prediction


def get_common_reg_metrics(ground_truth, prediction):
    ground_truth, prediction = np.squeeze(ground_truth), np.squeeze(prediction)
    rmse = mean_squared_error(ground_truth, prediction, squared=False)
    spearman, _ = stats.spearmanr(ground_truth, prediction)
    return rmse, spearman


def MultiView_model(x_train1, x_train2, x_train3, y_train, x_val1, x_val2, x_val3, y_val, x_test1, x_test2, x_test3):
    y_train, y_val = np.reshape(y_train, (-1, 1)), np.reshape(y_val, (-1, 1))

    input_view3 = Input(shape=(x_train3.shape[1],))
    x3 = Dense(16, activation="relu")(input_view3)
    x4 = Dense(8, activation="relu")(x3)
    output_layer_3 = Dense(1, activation='sigmoid')(x4)

    input_view2 = Input(shape=(x_train2.shape[1], x_train2.shape[2]))
    x3_2 = LSTM(units=16)(input_view2)
    x4_2 = Dense(8, activation="relu")(x3_2)
    output_layer_2 = Dense(1, activation='sigmoid')(x4_2)

    input_view1 = Input(shape=(x_train1.shape[1], x_train1.shape[2]))
    x2_3 = LSTM(units=16)(input_view1)
    x4_3 = Dense(8, activation="relu")(x2_3)
    output_layer_1 = Dense(4, activation="relu")(x4_3)

    final_layer = Concatenate()([output_layer_1, output_layer_2, output_layer_3])
    output_layer = Dense(1, activation='sigmoid')(final_layer)

    model = Model(inputs=[input_view1, input_view2, input_view3], outputs=output_layer)
    my_opt = tf.keras.optimizers.Adam(learning_rate=0.0002)
    model.compile(optimizer=my_opt, loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
    model.fit(x=[x_train1, x_train2, x_train3], verbose=0, y=y_train, batch_size=32, epochs=200, callbacks=[es],
              validation_data=([x_val1, x_val2, x_val3], y_val))
    test_prediction = model.predict([x_test1, x_test2, x_test3])

    return test_prediction


def MARTIAN_algorithm(input_feature_len=6, step=1):
    train_st_date, train_end_date = date(2019, 1, 1), date(2021, 7, 30)
    ground_truth_data_test, predicted_data_test = None, None
    rmse, spearsman, counter = 0, 0, 0

    for valid_start_date in pd.date_range(start='10/1/2020', end=train_end_date, freq='3MS'):
        counter += 1
        test_st_date = (valid_start_date + relativedelta(months=3)).to_pydatetime().date()
        test_end_date = (test_st_date + relativedelta(months=3, days=-1))

        sw_train_x_static, sw_train_x_dynamic, sw_train_x_cases, sw_train_y, sw_train_y_gt, sw_val_x_static, sw_val_x_dynamic, sw_val_x_cases, sw_val_y, sw_val_y_gt, sw_test_x_static, sw_test_x_dynamic, sw_test_x_cases, sw_test_y, sw_test_y_gt, min_data, max_data = data_preparation(
            train_st_date, test_end_date, input_feature_len, step)

        predicted_test = MultiView_model(x_train1=sw_train_x_cases, x_train2=sw_train_x_dynamic,
                                         x_train3=sw_train_x_static, y_train=sw_train_y, x_val1=sw_val_x_cases,
                                         x_val2=sw_val_x_dynamic, x_val3=sw_val_x_static , y_val=sw_val_y,
                                         x_test1=sw_test_x_cases, x_test2=sw_test_x_dynamic, x_test3=sw_test_x_static)
        predicted_test_i, ground_truth_test_i = data_denormalization(min_data, max_data, predicted_test), sw_test_y_gt
        rmse_test, spearsman_test = get_common_reg_metrics(ground_truth_test_i, predicted_test_i)

        rmse += rmse_test
        spearsman += spearsman_test

        if ground_truth_data_test is None:
            ground_truth_data_test = ground_truth_test_i
            predicted_data_test = predicted_test_i
        else:
            ground_truth_data_test = np.concatenate((ground_truth_data_test, ground_truth_test_i))
            predicted_data_test = np.concatenate((predicted_data_test, predicted_test_i))

    print(rmse / counter, spearsman / counter)


MARTIAN_algorithm()
