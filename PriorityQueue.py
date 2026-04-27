class PriorityQueue:
    def __init__(self):
        self.pqueue = []

    def insert(self, item, priority):
        #push item at the end of the queue
        self.pqueue.append([priority, item])
        i = len(self.pqueue) - 1
        # heapify-up, to find the correct postion of the item based on the priority
        while i > 0:
            parent = (i - 1) // 2
            if self.pqueue[parent][0] <= self.pqueue[i][0]:
                break
            self.pqueue[parent], self.pqueue[i] = self.pqueue[i], self.pqueue[parent]
            i = parent

    def heapify_down(self, i):
        #this function just implements the heapify down based on what we saw in class
        n = len(self.pqueue)
        while True:
            l, r = 2*i + 1, 2*i + 2
            smallest = i
            if l < n and self.pqueue[l][0] < self.pqueue[smallest][0]:
                smallest = l
            if r < n and self.pqueue[r][0] < self.pqueue[smallest][0]:
                smallest = r
            if smallest == i:
                break
            self.pqueue[i], self.pqueue[smallest] = self.pqueue[smallest], self.pqueue[i]
            i = smallest

    def extractMin(self):
        #make sure queue is not emoty before we try to remove the elemet with the lowest priority
        if not self.pqueue:
            raise IndexError("extract from empty PriorityQueue")

        #get the first item
        pr, it = self.pqueue[0]
        #get the last item to place it where the firt item is (root of the heap)
        last = self.pqueue.pop()
        #fix the order of the heap/ priority queuue by runnin a heapify down
        if self.pqueue:
            self.pqueue[0] = last
            self.heapify_down(0)

        return it, pr

    def decreasePriority(self, item, new_priority):
        #find the item, get its index in idx
        idx = -1
        for i, (pr, it) in enumerate(self.pqueue):
            if it == item:
                idx = i
                break
        if idx == -1:
            raise KeyError("item not found")

        #change the priority of the item
        self.pqueue[idx][0] = new_priority
        # heapify-up to find its correct positon in the queue
        i = idx
        while i > 0:
            parent = (i - 1) // 2
            if self.pqueue[parent][0] <= self.pqueue[i][0]:
                break
            self.pqueue[parent], self.pqueue[i] = self.pqueue[i], self.pqueue[parent]
            i = parent

    def isEmpty(self):
        return len(self.pqueue) == 0