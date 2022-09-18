class CircularQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = [None] * max_size
        self.head = self.tail = -1

    # Insert an element into the circular queue
    def enQueue(self, data):

        # if this is the first entry
        if self.head == -1:
            self.head = 0
            self.tail = 0
            self.queue[self.tail] = data
        else:
            # if not the first, increment the tail and add the data
            self.tail = (self.tail + 1) % self.max_size
            self.queue[self.tail] = data

            # if the tail has caught up with the head, increment the head
            if self.tail == self.head:
                self.head = (self.head + 1) % self.max_size

    # Get the contents of the entire queue
    def getQueue(self):
        # create the output variable
        data = []

        if self.head == -1:
            print("Queue is empty")

        pointer = self.head
        while True:
            # print(f"Head is {self.head}, tail is {self.tail}, pointer is {pointer}")
            data.append(self.queue[pointer])

            if pointer == self.tail:
                break

            pointer = (pointer + 1) % self.max_size

        return data

    # print the contents of the queue
    def printQueue(self):
        if self.head == -1:
            print("Queue is empty")

        for i in range(self.max_size):

            if self.queue[i] is not None:

                print(self.queue[i], end=" ")
        # new line
        print()
