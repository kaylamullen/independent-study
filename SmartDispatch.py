from __future__ import annotations
import heapq
import random as rand
import itertools
from typing import Any, Optional, Tuple
from numpy import random
import math
# from PriorityQueue import PriorityQueue
from WeightedGraph import WeightedGraph
import csv
import pandas as pd
from scipy.sparse import csr_array
from scipy.sparse.csgraph import bellman_ford
import networkx as nx


class System:
    def __init__(self, graph):
        self.total_possible_reward = 0
        self.total_tasks = 0
        self.task_list = []
        self.task_number = 0
        self.task_map = {}
        self.num_to_task = {}
        self.dasher_list = []
        self.task_dasher_number = 0
        self.dasher_map = {}
        self.num_to_dasher = {}
        self.total_reward = 0
        self.graph = graph
        self.matrix_graph = csr_array(graph.getMatrix())
        self.dist_matrix = dist_matrix = bellman_ford(self.matrix_graph, directed=True, indices=range(109))
        self.next_node = {}
        for i in range(109):
            self.next_node[i] = {}
            for j in range(109):
                try:
                    path, dist = self.graph.dijkstra_shortest_path(i, j)
                except IndexError:
                    # if there is no path between the two nodes
                    path = []
                if len(path) == 0:
                    self.next_node[i][j]  = None
                elif len(path) == 1:
                    self.next_node[i][j] = path[0]
                else:
                    self.next_node[i][j] = path[1]
        
    # def to_list(self, type):
    #     if type == 'dasher':
    #         tolist = []
    #         for dasher in self.dasher_list:
    #             tolist.append(dasher.location)
    #     else: 
    #         tolist = []
    #         for task in self.task_list:
    #             tolist.append(task.location)
    #     return tolist

    def add_dasher(self, dasher):
        self.dasher_list.append(dasher)
        self.dasher_map[dasher] = self.task_dasher_number
        self.num_to_dasher[self.task_dasher_number] = dasher
        self.task_dasher_number +=1
    
    def add_task(self, task):
        self.task_list.append(task)
        self.task_map[task] = self.task_dasher_number
        self.num_to_task[self.task_dasher_number] = task
        self.task_dasher_number += 1

    def remove_task(self, task):
        self.task_list.remove(task)
        
    def remove_dasher(self, dasher):
        self.dasher_list.remove(dasher)
        
    def add_reward(self, task):
        self.total_reward += task.reward
        

class Task:
    def __init__(self, location : int, appear_time : float, target_time : float, reward : int):
        self.appear_time = appear_time
        self.location = location
        self.reward = reward
        self.target_time = target_time
        self.reward_ratio = 0
        self.chosen = False # true if a dasher has been assigned this task to complete; false otherwise
        

class Dasher:
    def __init__(self, location : int, available : bool, start_time : float, exit_time : float):
        self.start_time = start_time
        self.location = location
        self.available = available 
        self.exit_time = exit_time
        self.len_time_avail = exit_time - start_time # 0 if they are not available
        self.simulator_time = 0 # the current time in the simulator
        self.time_next_avail = start_time # the earliest time at which they can start going towards a new task
        self.distance_traveled = 0 #the current distance this dasher traveled 


class EventHandle:
    """Simple cancelable handle for a scheduled event."""
    __slots__ = ("_cancelled",)
    def __init__(self) -> None:
        self._cancelled = False
    def cancel(self) -> None:
        self._cancelled = True
    @property
    def cancelled(self) -> bool:
        return self._cancelled

class Simulator:

    def __init__(self, start_time: float = 0.0) -> None:
        self.now = float(start_time)
        self._queue: list[Tuple[float, int, Any, Any, EventHandle]] = []
        self._seq = itertools.count()
        self._stopped = False
        self.events_processed = 0
        self.recalc_events = {}

    def schedule_at(self, time: float, event_id: Any, payload: Any = None) -> EventHandle:
        if time < self.now:
            raise ValueError("Cannot schedule in the past")
        seq = next(self._seq)
        h = EventHandle()
        heapq.heappush(self._queue, (float(time), seq, event_id, payload, h))
        return h

    def _pop_next(self):
        while self._queue:
            time, seq, event_id, payload, h = heapq.heappop(self._queue)
            if not h.cancelled:
                return time, event_id, payload
            # skipped cancelled
        return None

    def step(self) -> bool:
        if self._stopped:
            return False
        item = self._pop_next()
        if item is None:
            return False
        time, event_id, payload = item
        self.now = time
        # dispatch to user-defined handler
        self.handle(event_id, payload)
        self.events_processed += 1
        return True

    def run(self, until: Optional[float] = None, max_events: Optional[int] = None) -> None:
        self._stopped = False
        processed = 0
        while not self._stopped:
            if not self._queue:
                break
            if until is not None and self._queue[0][0] > until:
                break
            if max_events is not None and processed >= max_events:
                break
            if not self.step():
                break
            processed += 1

    def stop(self) -> None:
        self._stopped = True

    def handle(self, event_id: Any, payload: Any) -> None:
        """Override in a subclass with a simple switch (if/elif) on event_id."""
        raise NotImplementedError("Override handle(event_id, payload)")

    # Example usage with a simple switch-case style handler



class SmartDispatch(Simulator):
    def handle(self, event_id: str, payload: Any) -> None:
        if event_id == 'Task Arrival':
            task = payload[1]
            system = payload[0]
            system.add_task(task)
            system.total_possible_reward = system.total_possible_reward+task.reward
            system.total_tasks += 1
            # recalc at t+1 to make sure all dashers ariving at self.now are considered in the recalc for task
            next_recalc_time = self.now+1
            if next_recalc_time not in self.recalc_events.keys():
                self.schedule_at(next_recalc_time, 'Recalc', system)
                self.recalc_events[next_recalc_time] = 'Recalc'
            
        elif event_id == 'Dasher Arrival':
            dasher = payload[1]
            system = payload[0]
            try:
                if dasher.exit_time <= self.now:
                    system.remove_dasher(dasher)
            except:
                pass 
            if dasher not in system.dasher_list:
                system.add_dasher(dasher)
                    
            next_recalc_time = self.now+1
            if next_recalc_time not in self.recalc_events.keys():

                self.schedule_at(next_recalc_time, 'Recalc', system)
                self.recalc_events[next_recalc_time] = 'Recalc'
            

        elif event_id == 'Recalc':
            system = payload
            graph = system.graph
            dist_matrix = system.dist_matrix
            

            # run bellman ford to get the shortest time from each node in the graph to each task
            # task_list = system.to_list('task')
            # map_task_list= {}
            # i = 0
            # for task in task_list:
            #     map_task_list[task] = i
            #     i+=1
            #dist_matrix = bellman_ford(matrix_graph, directed=True, indices=task_list)
            #for i in range(110):
            #print(dist_matrix)
            #print(type(dist_matrix))
            ratio_matrix = {}
            
            # for each task-dasher possible match, calculate the reward/time cost ratio 
            # that would be obtained by this dasher completing this task
            for task in system.task_list:
                reward = task.reward
                ratio_matrix[system.task_map[task]] = {}
                for dasher in system.dasher_list:
                    if dist_matrix[task.location][dasher.location] == 0:
                        ratio_matrix[system.task_map[task]][system.dasher_map[dasher]] = math.inf # if the task is at their current location make sure they collect that reward
                    else:
                        ratio_matrix[system.task_map[task]][system.dasher_map[dasher]] = reward/dist_matrix[task.location][dasher.location]

            # for each dasher, determine the task highest reward:time match for them
            # best_task_for_dasher = {}
            # for dasher in system.dasher_list:
            #     current_best = 0
            #     current_task = None
            #     for task in system.task_list:
            #         # check time constraint
            #         time_to_task = dist_matrix[task.location][dasher.location]
            #         if time_to_task + self.now > dasher.exit_time:
            #             # continue to check the next possible task for the dasher 
            #             # because it will not be possible for them to complete this one
            #             pass
                    
            #         # if the task can be completed before the dasher exits the system, 
            #         # check if this is the best task for the dasher to complete
            #         if ratio_matrix[system.task_map[task]][system.dasher_map[dasher]] > current_best:
            #             current_best = ratio_matrix[task][dasher]
            #             current_task = task
            #     if current_task != None:
            #         path = graph.dijkstra_shortest_path(dasher.location, current_task.location)
            #         best_task_for_dasher[dasher] = (current_task, path)
            
        
            G = nx.Graph()
            
            # IDEA FOR LATER: make task number/dasher number attribute of task and dasher instead of using map

            for task in system.task_list:
                for dasher in system.dasher_list:
                    if ratio_matrix[system.task_map[task]][system.dasher_map[dasher]] != 0:
                        G.add_node(system.task_map[task], bipartite=0)
                        G.add_node(system.dasher_map[dasher], bipartite=1)
                        G.add_edge(system.task_map[task], system.dasher_map[dasher], weight=ratio_matrix[system.task_map[task]][system.dasher_map[dasher]])
            
            if system.task_list == []:
                pass
            # get a list of pairs (task, dasher) that guarantees non-overlapping and maximizes sum of rewards/cost ratios 
            used = set()
            matching = set()

            edges = sorted(
                G.edges(data=True),
                key=lambda x: x[2]["weight"],
                reverse=True
            )

            for u, v, data in edges:
                if u not in used and v not in used:
                    matching.add((u, v))
                    used.add(u)
                    used.add(v)
            #move dasher towards their 
            assigned_dasher = []
            for pair in matching:
                # check order of pair
                if pair[0] in system.num_to_task.keys():
                    task = system.num_to_task[pair[0]]
                    dasher = system.num_to_dasher[pair[1]]
                else:
                    task = system.num_to_task[pair[1]]
                    dasher = system.num_to_dasher[pair[0]]
                assigned_dasher.append(dasher)
                if dist_matrix[dasher.location][task.location] == 0:
                    next_node = dasher.location
                else:
                    next_node = system.next_node[dasher.location][task.location]
                # if the dasher has reached the task, collect reward, remove the task
                # and have the dasher check back for a new task
                # print(f"next node: {next_node}")
                if next_node == task.location:
                    system.add_reward(task)
                    system.remove_task(task)
                
                dasher.location = next_node
                dasher.distance_traveled = dasher.distance_traveled+1 #  the dasher by 1 edge, can be changed to the edge's weight later
                self.schedule_at(self.now + dist_matrix[dasher.location][next_node], 'Dasher Arrival', (system, dasher))

            # have any dasher that was not assigned to a task check back for a new one at t+1
            for dasher in system.dasher_list:
                if dasher not in assigned_dasher:
                    self.schedule_at(self.now + 1, 'Dasher Arrival', (system, dasher))

            
            # find dasher with highest reward : time cost ratio --> assign to chosen dasher
            
            # move chosen dasher 1 step (edge) towards this task
            
            
            # if the next node is the task --> new collect reward event at t+edge cost
            
            # else --> new dasher arrival event at t+edge cost
        elif event_id == 'Collect Reward':
            system = payload[0]
            dasher = payload[1]
            reward = payload[2]
            system.add_reward(reward)
            self.schedule_at(self.now, 'Dasher Arrival', (system, dasher))
            
        
        elif event_id == 'Stop':
            # print(f"[{self.now:.3f}] stopping")
            self.stop()
            pass
        else:
            print(f"[{self.now:.3f}] unknown event {event_id!r} -> {payload}")



def read_graph(fname):
    # Open the file
    file = open(fname, "r")
    # Read the first line that contains the number of vertices
    # numVertices is the number of vertices in the graph (n)
    numVertices = file.readline()

    g = WeightedGraph()

    # Next, read the edges and build the graph
    for line in file:
        # edge is a list of 3 indices representing a pair of adjacent vertices and the weight
        # edge[0] contains the first vertex (index between 0 and numVertices-1)
        # edge[1] contains the second vertex (index between 0 and numVertices-1)
        # edge[2] contains the weight of the edge (a positive integer)
        edge = line.strip().split(",")

    # Use the edge information to populate your graph object
        g.addNode(int(edge[0]))
        g.addNode(int(edge[1]))
        g.addEdge(int(edge[0]), int(edge[1]), int(edge[2]))



    # Close the file safely after done reading
    file.close()
    return g
            
def read_dashers(fname):
    dasher_df = pd.read_csv(fname)
    dashers = []
    # making each dasher a dasher object
    for i, row in dasher_df.iterrows():
        dasher = Dasher(location=row['start-location'], available=False, start_time=row['start_time'], exit_time=row['exit_time'])
        dashers.append(dasher)
    return dashers

def read_tasks(fname):
    task_df = pd.read_csv(fname)
    tasks=[]
    # making each task a task object
    for i, row in task_df.iterrows():
        # randomly generate a length of time the task is available for then add to appear_time
        time_avail = random.randint(1, 10)
        target_time = row['minute'] + time_avail
        reward = random.randint(1, 5)
        task = Task(location=row['VERTEX'], appear_time=row['minute'], target_time=target_time, reward=reward)
        tasks.append(task)
    return tasks
  
def schedule_events(sim, dashers, tasks, system, stop_time):
        for task in tasks:
            sim.schedule_at(task.appear_time, 'Task Arrival', (system, task))
        for dasher in dashers:
            sim.schedule_at(dasher.start_time, 'Dasher Arrival', (system, dasher))
        sim.schedule_at(stop_time, 'Stop', None)
        
        
def get_results(dasher_fname, task_fname, stop_time):
    num_events_per_iter = []
    total_reward_per_iter = []
    pct_total_possible_reward_per_iter = []
    for i in range(10):
        sim = SmartDispatch()
        graph = read_graph("project_files/grid100.txt")
        system = System(graph)
        dashers = read_dashers(dasher_fname)
        tasks = read_tasks(task_fname)
        schedule_events(sim, dashers, tasks, system, stop_time)
        sim.run()
        num_events = sim.events_processed
        total_reward = system.total_reward
        pct_total_possible = (system.total_reward / system.total_possible_reward) *100
        num_events_per_iter.append(num_events)
        total_reward_per_iter.append(total_reward)
        pct_total_possible_reward_per_iter.append((pct_total_possible))
        print("events processed:", num_events)
        print(f"total reward: {total_reward}")
        print(f"total possible reward: {system.total_possible_reward}")
        print(f"max possible reward: {system.total_possible_reward}")
        print(f"pct total possible reward collected: {system.total_reward/system.total_possible_reward}")
    i = 0
    for reward, num_events, pct_total_possible in zip(total_reward_per_iter, num_events_per_iter, pct_total_possible_reward_per_iter):
        print(f"{i}: Total Reward: {reward}, Number of Events Processed: {num_events}, pct total possible reward: {pct_total_possible}")
        i += 1
    print("----------------------------------")
    print(f"Average total reward collected: {sum(total_reward_per_iter) / len(total_reward_per_iter)}")
    print(f"Average percent of total possible reward collected: {sum(pct_total_possible_reward_per_iter) / len(pct_total_possible_reward_per_iter)}")
     
def test_diff_dasher_amts(dasher_fname, task_fname, stop_time):
    num_events_per_iter = []
    total_reward_per_iter = []
    data = []
    pct_total_possible_reward_per_iter = []
    for num_tasks in [100, 200, 300, 400, 500, 699]:
        for num_dashers in [1, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]:
            sim = SmartDispatch()
            graph = read_graph("project_files/grid100.txt")
            system = System(graph)
            dashers = read_dashers(dasher_fname)
            # chosen_dashers = dashers[:num_dashers]
            chosen_dashers = rand.sample(dashers, num_dashers)
            chosen_tasks = rand.sample(read_tasks(task_fname), num_tasks)
            tasks = read_tasks(task_fname)
            schedule_events(sim, chosen_dashers, chosen_tasks, system, stop_time)
            sim.run()
            num_events = sim.events_processed
            total_reward = system.total_reward
            pct_total_possible = (system.total_reward / system.total_possible_reward) *100
            num_events_per_iter.append(num_events)
            total_reward_per_iter.append(total_reward)
            pct_total_possible_reward_per_iter.append((pct_total_possible))
            data.append((num_dashers, num_tasks, num_dashers/num_tasks, num_events, total_reward, pct_total_possible))
        i = 0
    with open('smartdispatch_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['num_dashers', 'num_tasks', 'ratio_dashers_tasks', 'num_events', 'total_reward', 'pct_total_possible_reward'])
        writer.writerows(data)
        for reward, num_events, pct_total_possible in zip(total_reward_per_iter, num_events_per_iter, pct_total_possible_reward_per_iter):
            # add below to csv
            print(f"{i}: Total Reward: {reward}, # Events Processed: {num_events}, # Dashers: {num_dashers}, # Tasks: {num_tasks}, ratio dashers:tasks: {num_dashers/num_tasks}, pct total possible reward: {pct_total_possible}")
            i += 1
    print("----------------------------------")
    print(f"Average total reward collected: {sum(total_reward_per_iter) / len(total_reward_per_iter)}")
    print(f"Average percent of total possible reward collected: {sum(pct_total_possible_reward_per_iter) / len(pct_total_possible_reward_per_iter)}")
     


if __name__ == "__main__":
    # class SimpleSim(Simulator):
        # def handle(self, event_id: str, payload: Any) -> None:
        #     # simple switch-case implemented with if/elif
        #     if event_id == "say":
        #         print(f"[{self.now:.3f}] {payload}")
        #     elif event_id == "heartbeat":
        #         print(f"[{self.now:.3f}] heartbeat")
        #         # reschedule recurring heartbeat
        #         self.schedule_at(self.now + 1.0, "heartbeat")
        #     elif event_id == "stop":
        #         print(f"[{self.now:.3f}] stopping")
        #         self.stop()
        #     else:
        #         print(f"[{self.now:.3f}] unknown event {event_id!r} -> {payload}")

    # sim = SimpleSim()
    # sim.schedule_at(1.0, "say", "first at t=1.0")
    # h = sim.schedule_at(2.0, "say", "second at t=2.0 (will be canceled)")
    # sim.schedule_at(3.0, "say", "third at t=3.0")
    # h.cancel()
    # sim.schedule_at(0.5, "heartbeat", 0.5)
    # # schedule a stop event at t=2.2
    # sim.schedule_at(10, "stop", None)

    # sim.run()
    # print("events processed:", sim.events_processed)
    
    
    
    # sim = SmartDispatch()
    # graph = read_graph("project_files/grid100.txt")
    # system = System(graph)
    # dashers_mini = read_dashers("input/christin-new-dashers-mini.csv")
    # # tasks_mini = read_tasks("project_files/our_tasklog.csv")
    # tasks_mini = read_tasks("input/christine-new-tasklog-mini.csv")
    # schedule_events(sim, dashers_mini, tasks_mini, system, 1090)
    # sim.run()
    # print("events processed:", sim.events_processed)
    # print(f"total reward: {system.total_reward}")
    # print(f"max possible reward: {system.total_possible_reward}")
    # print(f"pct total possible reward collected: {system.total_reward/system.total_possible_reward}")
    
    # sim = SmartDispatch()
    # graph = read_graph("project_files/grid100.txt")
    # system = System(graph)
    # dashers_mini = read_dashers("input/christine-new-dashers-10.csv")
    # # tasks_mini = read_tasks("project_files/our_tasklog.csv")
    # tasks_mini = read_tasks("input/christine-new-tasklog-10.csv")
    # schedule_events(sim, dashers_mini, tasks_mini, system, 1090)
    # sim.run()
    # print("events processed:", sim.events_processed)
    # print(f"total reward: {system.total_reward}")
    # print(f"max possible reward: {system.total_possible_reward}")
    # print(f"pct total possible reward collected: {system.total_reward/system.total_possible_reward}")
    
    
    # sim = SmartDispatch()
    # graph = read_graph("project_files/grid100.txt")
    # system = System(graph)
    # dashers_small = read_dashers("input/christine-new-dashers-small.csv")
    # tasks_small = read_tasks("input/christine-new-tasklog-small.csv")
    # schedule_events(sim, dashers_small, tasks_small, system, 1439)
    # sim.run()
    # print("events processed:", sim.events_processed)
    # print(f"total reward: {system.total_reward}")
    # print(f"max possible reward: {system.total_possible_reward}")
    # print(f"pct total possible reward collected: {system.total_reward/system.total_possible_reward}")
    
    
    # sim = SmartDispatch()
    # graph = read_graph("project_files/grid100.txt")
    # system = System(graph)
    # dashers_mini = read_dashers("input/christin-new-dashers-mini.csv")
    # # tasks_mini = read_tasks("project_files/our_tasklog.csv")
    # tasks_large = read_tasks("input/christine-new-tasklog.csv")
    # schedule_events(sim, dashers_mini, tasks_large, system, 1440)
    # sim.run()
    # print("testing few dashers, many tasks")
    # print("events processed:", sim.events_processed)
    # print(f"total reward: {system.total_reward}")
    # print(f"max possible reward: {system.total_possible_reward}")
    # print(f"pct total possible reward collected: {system.total_reward/system.total_possible_reward}")
    
    
    
    # sim = SmartDispatch()
    # graph = read_graph("project_files/grid100.txt")
    # system = System(graph)
    # dashers_large = read_dashers("input/christine-new-dashers.csv")
    # tasks_large = read_tasks("input/christine-new-tasklog.csv")
    # schedule_events(sim, dashers_large, tasks_large, system, 1439)
    # sim.run()
    # print("events processed:", sim.events_processed)
    # print(f"total reward: {system.total_reward}")
    # print(f"max possible reward: {system.total_possible_reward}")
    # print(f"pct total possible reward collected: {system.total_reward/system.total_possible_reward}")
    
    
    # get_results("input/christine-new-dashers-small.csv", "input/christine-new-tasklog-small.csv", 1440)
    # # get_results("project_files/dashers.csv", "project_files/tasklog.csv", 500)
    
    # get_results("input/christine-new-dashers.csv", "input/christine-new-tasklog.csv", 1440)
    # test_diff_dasher_amts("input/christine-new-dashers.csv", "input/christine-new-tasklog.csv", 1440)

    get_results("input/christine-inputs/christin-new-dashers-mini.csv", "input/christine-inputs/christine-new-tasklog-mini.csv", 1440)
    
    #agents = [(0, 17), (0, 17), (0, 17)]
    #agents = [(3, 9), (10, 19), (17, 2), (5, 18), (10, 15)]
    #agents = agents2 = [(0, 1), (11, 14), (5, 20)]
    # cars = []
    # id = 1
    # for agent in agents:
    #     x = int(random.exponential(scale=1))
    #     car = Car(id, x, agent[0], agent[1])
    #     cars.append(car)
    #     sim.schedule_at(x, 'first_time', (car, system))
    #     id = id+1
    
    # sim.run()
    # total = 0
    # for car in cars:
    #     print('Car ', car.id, '(', car.start, ', ', car.destination, '), arrived at t=', car.arrival_time, end="")
    #     car.print_path()
    #     print()
    #     total = total+car.path_cost
    # print('Average congestion is', total/(id-1))
    # print('Total congestion is', total)
    # # code for random time generation
    
    # # final tests
    # agents1 = read_agents('agents.txt')
    # graph1 = read_graph('graph1.txt')
    # print(agents)
    # print(graph1)
    
