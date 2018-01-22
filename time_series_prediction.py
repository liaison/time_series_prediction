"""

This module provides a framework to conduct the prediction with the time-series data. 

It includes the functions such as sequence slicing, data normalization and model evaluation etc.

"""


"""
#########################################################################

    Data Preprocessing

#########################################################################
"""
import numpy as np          
import pandas as pd              
import matplotlib.pyplot as plt

from datetime import date, datetime, timedelta


def get_date_range_list(start_date_str, end_date_str):
    """ Generate a list of date string in the format of Y-m-d. 
    example:
        start_date_str = '2017-02-01'
        end_date_str = '2018-01-12'
        df_date = get_date_range_list(start_date_str, end_date_str)
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    daterange = pd.date_range(start_date, end_date)

    return pd.DataFrame([single_date.strftime("%Y-%m-%d") for single_date in daterange],
                        columns=['date'])


def extract_time_series(df_hotel_stats, airport,
                        price_type='mean', nights=1,
                        interpolate='nearest'):
    """
        extract a time-serie on a specific type of price 
            around a particular airport.
        
        price_type:  'min', 'mean' and 'max'
        interpolate: the method to fill the missing values,
                     e.g. 'linear', 'nearest'
    """
    raw_data = df_hotel_stats[(df_hotel_stats['airport_code'] == airport) & 
                              (df_hotel_stats['nights'] == nights)]
    
    # get the date range for this specific time series
    checkin_dates = raw_data['checkin_date'].values
    start_date_str = checkin_dates[0]
    end_date_str = checkin_dates[-1]
    df_date = get_date_range_list(start_date_str, end_date_str)
    
    time_series = df_date.merge(raw_data, left_on='date', right_on='checkin_date', how='left')
    
    # the NaN in 'checkin_date' indicates the missing value
    # Note:  by default, limit_direction='forward', which could result in leading NaN.
    return time_series[['checkin_date', price_type]].interpolate(method=interpolate, limit_direction='backward')


from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error


def sequence_slicing(sequence, look_back):
    """
        slice the sequence into tuples of (feature_vector, target), i.e. (X, y)
        where X = [X_t... X_(t+look_back-1)]
              y = X_(t+look_back)
    """
    dataX, dataY = [], []
    for i in range(len(sequence)-look_back):
        a = sequence[i:(i+look_back)]
        dataX.append(a)
        dataY.append(sequence[i + look_back])
    return np.array(dataX), np.array(dataY)


class NoneScaler(MinMaxScaler):
    """  a scaler that does nothing """
    def fit_transform(self, data):
        return data
    
    def inverse_transform(self, data):
        return data


def normalize_input_split(raw_data, scaler='MinMaxScaler', look_back=1,
                     train_test_split = None, train_validate_split=1, # by default, there is NO validation dataset.
                     test_window = None, window_size = 7,             # fixed testing window
                     train_split_size = None, test_split_size = None, # flexible training and testing data set
                     verbose = False):
    '''
        There are 3 modes to build input data sets:
        
            1). train_test_split ratio
            2). fixed-size testing window, with two windows at the end
            3). specified size for training and testing
            
        build three data sets, i.e. train, validate, test
    '''
    # normalize the dataset
    # Note: one scaler for all features and also target
    if (scaler == 'MinMaxScaler'):
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif (scaler == 'RobustScaler'):
        # the RobustScaler is more resilient to outliers
        scaler = RobustScaler()
    elif (scaler == 'None'):
        scaler = NoneScaler()

    raw = np.reshape(raw_data, (-1,1))
    dataset = scaler.fit_transform(raw)

    # slicing the original sequence according to the look_back window
    totalX, totalY = sequence_slicing(dataset, look_back)
    
    # First, split the entire dataset into train and test sets
    if (train_test_split != None):
        train_size = int(len(totalX) * train_test_split)
        trainX, trainY = totalX[0:train_size], totalY[0:train_size]
        testX, testY = totalX[train_size:], totalY[train_size:]
        
    elif (test_window != None):
        train_size = int(len(totalX) - test_window * window_size)
        test_size = window_size
    
        trainX, trainY = totalX[0:train_size], totalY[0:train_size]
        testX = totalX[train_size:(train_size + test_size)] 
        testY = totalY[train_size:(train_size + test_size)]

    elif ((train_split_size != None) and (test_split_size != None)):
        # check if the data is splitable given the requirements
        if ((train_split_size + test_split_size) > len(totalX)):
            # return empty sets
            train_size = 0
            trainX, trainY = totalX[0:0], totalY[0:0]
            testX, testY = totalX[0:0], totalY[0:0]
        else:
            train_size = train_split_size
            trainX, trainY = totalX[0:train_size], totalY[0:train_size]
            testX = totalX[train_size:(train_size + test_split_size)]
            testY = totalY[train_size:(train_size + test_split_size)]
    else:
        train_size = 0
        trainX, trainY = totalX[0:0], totalY[0:0]
        testX, testY = totalX[0:0], totalY[0:0]
    
    # Second, further split the train data set into sub_train and validation
    sub_train_size = int(train_size * train_validate_split)
    sub_trainX, sub_trainY = trainX[0:sub_train_size], trainY[0:sub_train_size]
    validateX, validateY = trainX[sub_train_size:], trainY[sub_train_size:]
    
    # reshape feature vector to be [samples, features] for SVR
    sub_trainX = np.reshape(sub_trainX, (sub_trainX.shape[0], sub_trainX.shape[1]))
    validateX = np.reshape(validateX, (validateX.shape[0], validateX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1]))

    # reshape the target vector to 1-dimension
    sub_trainY = np.ravel(sub_trainY)
    validateY = np.ravel(validateY)
    testY = np.ravel(testY)
    
    if (verbose):
        print("total:", len(totalX),
              'train:', len(sub_trainX),
              'validate:', len(validateX),
              'test:', len(testX))

    return (sub_trainX, sub_trainY, validateX, validateY, testX, testY, scaler)


"""
#########################################################################

    Model Evaluation

#########################################################################
"""



def rolling_predict(model, head, steps, verbose = True):
    """
        predict the next N steps of values, 
        starting from a single element,
        in a recursive way, i.e. using the predicted value for the following prediction.
    """
    if (steps <= 0):
        return []
    else:
        # predict the target value for the "head" element
        head_target = model.predict(head)
        head_target = head_target.reshape(1)
        
        # otherwise, retrieve the feature vector for the head element
        feature = head[0]
        
        if (verbose):
            print(feature, '->',  head_target)
        
        # shift the feature vector to left and pad with newly-predicted value
        next_feature = np.roll(feature, -1)
        next_feature[-1] = head_target
        
        # construct a new head with the new feature vector
        next_head = np.expand_dims(next_feature, axis=0)

        # recursively predict the next N-1 values
        next_results = rolling_predict(model, next_head, steps-1, verbose = verbose)

        # concatenat the current and the next values
        ret = []
        ret.extend(head_target)
        ret.extend(next_results)
        return ret


def predict_and_inverse(model, testX, scaler, rolling = False, verbose = False):
    """
        predict the target value for the testing input 
         and inverse the value with the scaler.
    """
    
    if (rolling):
        head = testX[0:1]
        steps = len(testX)
        predicted_values = rolling_predict(model, head, steps, verbose)
    else:
        predicted_values = model.predict(testX)
    
    # inverse the prediction to its original scale
    inversed_values = np.array(predicted_values).reshape(-1, 1)
    inversed_values = scaler.inverse_transform(inversed_values)
    inversed_values = np.ravel(inversed_values)
    
    return (predicted_values, inversed_values)


def APE(origin, predict):
    ''' absolute percentage error
        origin as the denominator
    '''
    # convert the inputs into dataframes so that we could use the APIs such as replace()
    df_1 = pd.DataFrame(origin)
    df_2 = pd.DataFrame(predict)
    
    # drop the positive and negative infinitities, when divided by zero
    difference = ((df_1-df_2)/df_1).replace([np.inf, -np.inf], np.nan).dropna(inplace=False)
    
    ret = np.abs(difference) * 100
    return ret.iloc[:, 0].values


def MAPE(origin, predict):
    ''' MEAN absolute percentage error
        origin as the denominator
    '''
    return APE(origin, predict).mean()

### TODO: exclude the evaluation on the missing values 

def evaluate(model, trainX, trainY, testX, testY, scaler, verbose=False):
    """
        Validate the model with the testing data and
        calculate the KPIs, e.g. MAPE (Mean-Absolute-Percentage-Error)
    """
    
    # make predictions with the training data set.
    trainPredict, inversed_trainPredict = predict_and_inverse(
        model, trainX, scaler, rolling = False)  # Do NOT do the rolling prediction.
    
    # make predictions on the testing data.
    testPredict, inversed_testPredict = predict_and_inverse(
        model, testX, scaler, rolling = True)
    
    # inverse the scaled target values
    inversed_trainY = np.ravel(scaler.inverse_transform(trainY.reshape(-1, 1)))
    inversed_testY = np.ravel(scaler.inverse_transform(testY.reshape(-1, 1)))
    
    kpi = {
        'train_mape' : MAPE(inversed_trainY, inversed_trainPredict),
        'test_mape' : MAPE(inversed_testY, inversed_testPredict),
        'test_ape' : APE(inversed_testY, inversed_testPredict)
    }
    
    if (verbose):
        kpi['train_predict'] = inversed_trainPredict
        kpi['test_predict'] = inversed_testPredict
    
    return kpi


def plot_benchmark(baseline, inversed_trainPredict, inversed_testPredict, look_back,
                   title = 'prediction on mean hotel price'):
    """
        Plot 3 lines, 
        i.e. baseline, the fitting of the training data and the prediction of testing data
    """
    total_size = len(inversed_trainPredict) + len(inversed_testPredict) + look_back
    
    # shift train predictions for plotting
    trainPredictPlot = np.empty(total_size)
    trainPredictPlot[:] = np.nan
    #trainPredictPlot[look_back:len(inversed_trainPredict)+look_back] = inversed_trainPredict
    trainPredictPlot[look_back:len(inversed_trainPredict)+look_back] = inversed_trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty(total_size)
    testPredictPlot[:] = np.nan
    testPredictPlot[len(inversed_trainPredict)+look_back:] = inversed_testPredict
    
    # plot baseline and predictions
    # baseline = dataset.reshape(-1, 1)
    # baseline = np.reshape(dataset.values, (-1, 1))
    
    fig = plt.figure()
    fig.patch.set_facecolor('white')

    plt.plot(baseline, marker='.', label='baseline', linewidth=1, alpha=0.6, color='b')
    plt.plot(trainPredictPlot, marker='*', label='model fitting', linewidth=1, alpha=0.6, color='g')
    plt.plot(testPredictPlot, marker='o', label='model prediction', linewidth=1, alpha=0.6, color='r')
    plt.legend(loc='best')
    plt.title(title)
    #plt.title("SVR prediction on mean_price of hotels around " + selected_airport)
    plt.xlabel('time series (/day)')
    plt.ylabel('Mean Hotel Price (/$)')
    plt.show()


