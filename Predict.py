from BaselineBellmanFord import Task
import pandas as pd
from prophet import Prophet
import random
from sklearn.model_selection import train_test_split
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics



def read_tasks(fname):
    task_df = pd.read_csv(fname)
    tasks=[]
    task_list = []
    # making each task a task object
    for i, row in task_df.iterrows():
        # randomly generate a length of time the task is available for then add to appear_time
        time_avail = random.randint(1, 30)
        target_time = row['minute'] + time_avail
        reward = random.randint(1, 5)
        task = Task(location=row['VERTEX'], appear_time=row['minute'], target_time=target_time, reward=reward)
        tasks.append(task)
        task_list.append({'vertex': row['VERTEX'], 'appearTime': row['minute'], 'targetTime': target_time, 'reward': reward})

    return tasks, task_list


def tasks_to_dicts(tasks):
    return [{'vertex': task.location, 'minute': task.appear_time, 'reward': task.reward} for task in tasks]

'''
Input:
    task_list: a list of task objects
    target_vertex: int corresponding to the vertex id of the location to predict the next reward for
    curr_time: the current time of the simulator
Output: the next predicted reward to occur after the current time and the minute it will occur at
'''
def predict_reward_and_min(task_list, target_vertex, curr_time):
    if len(task_list) == 0:
        return None, None
    df = pd.DataFrame(tasks_to_dicts(task_list))
    # group by vertex (location) and count the number of occurrences
    counts = df.groupby(df['vertex']).size().reset_index(name="count")

    # Filter to only show locations with more than 1 occurrence
    multiples = (counts[counts['count'] > 1]).sort_values('count', ascending=False)
    # print(f"Found {len(multiples)} locations with multiple entries:")
    # target_vertex = multiples.iloc[0]['vertex']
    # print(target_vertex)
    # --> just need to make sure the target vertex exists in the multiples dataframe
    if target_vertex not in multiples['vertex'].values:
        return None, None # there have not been multiple tasks here in the past, so there is not likely to be multiple tasks here in the future
        # stop
    print(f"predicting for vertex {target_vertex} at time {curr_time}")
    # Filter the data
    filtered_data= df[df['vertex'] == target_vertex].copy()

    # print(f"Found {len(filtered_data)} records for location {target_vertex}.")
    filtered_data['minute_as_dt'] = pd.to_datetime(filtered_data['minute'], unit='m')
    filtered_data['_period'] = filtered_data['minute_as_dt'].dt.to_period('min').dt.to_timestamp()
    # grouped = filtered_data.groupby(["_period"]).size().rename("count").reset_index()
    # Initialize the model
    model = Prophet()

    # Fit the model
    model.fit(filtered_data.rename(columns={"_period": "ds", "reward": "y"}))


    # Create future dates dataframe 
    future = model.make_future_dataframe(periods=1) 

    # Make predictions 
    forecast = model.predict(future) 

    # View the forecast 
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    reward = forecast['yhat'].iloc[0]
    
    forecast['minute'] = (forecast['ds'].dt.hour * 100) + (forecast['ds'].dt.minute)
    # print(forecast)
    after_now_forecast = forecast[forecast['minute'] > curr_time]
    reward = after_now_forecast['yhat'].iloc[0]
    minute = after_now_forecast['minute'].iloc[0]
    print(len(forecast))
    print(forecast)
    return reward, minute


def test_performance(task_list):
    if len(task_list) == 0:
        return None, None
    df = pd.DataFrame(tasks_to_dicts(task_list))
    # group by vertex (location) and count the number of occurrences
    counts = df.groupby(df['vertex']).size().reset_index(name="count")

    # Filter to only show locations with more than 1 occurrence
    multiples = (counts[counts['count'] > 1]).sort_values('count', ascending=False)
    
    verts = multiples['vertex'].unique().tolist()
    all_mse_df = pd.DataFrame()
    # calculate the MSE of prophet on each vertex in verts
    for vert in verts:
        filtered_data= df[df['vertex'] == vert].copy()

        # print(f"Found {len(filtered_data)} records for location {target_vertex}.")
        filtered_data['minute_as_dt'] = pd.to_datetime(filtered_data['minute'], unit='m')
        filtered_data['_period'] = filtered_data['minute_as_dt'].dt.to_period('min').dt.to_timestamp()
        # grouped = filtered_data.groupby(["_period"]).size().rename("count").reset_index()
        # Initialize the model
        model = Prophet()

        # Fit the model
        try:
            model.fit(filtered_data.rename(columns={"_period": "ds", "reward": "y"}))
            cv_results_df = cross_validation(model, initial='100 minutes', period='30 minutes', horizon='10 minutes')
        except:
            print(f"not possible to predict future tasks for vertex {vert}")
            continue # not enough data 
        mse = performance_metrics(cv_results_df, metrics=['mse'])
        all_mse_df = pd.concat([all_mse_df, mse])
        print(type(mse))
        # print(f"MSE for vertex {vert}: {mse}")
        
    mse_avgs_by_horizon_df = all_mse_df.groupby('horizon')['mse'].mean()
    avg_mse = all_mse_df['mse'].mean()
    median_mse = all_mse_df['mse'].median()
    return all_mse_df, mse_avgs_by_horizon_df, avg_mse, median_mse
        

        


def main():
    tasks, task_list = read_tasks('input/christine-new-tasklog.csv')
    reward, minute = predict_reward_and_min(tasks,45,1810)
    print(f'reward of {reward} at time {minute}')
    
    all_mse, mse_avgs, avg_mse, median_mse = test_performance(tasks)
    mse_avgs.to_csv('output/predict-mse-avgs.csv')
    print(f"mse_avgs")
    print(f'avg mse: {avg_mse}, median mse: {median_mse}')
    
    
    
if __name__ == '__main__':
    main()
    
    