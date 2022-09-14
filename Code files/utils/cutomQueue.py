# Circular Queue implementation in Python


class CircularQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = [None] * max_size
        self.head = self.tail = -1

    # Insert an element into the circular queue
    def enQueue(self, data):

        if (self.tail + 1) % self.max_size == self.head:
            print("The circular queue is full\n")

        elif self.head == -1:
            self.head = 0
            self.tail = 0
            self.queue[self.tail] = data
        else:
            self.tail = (self.tail + 1) % self.max_size
            self.queue[self.tail] = data

    # Delete an element from the circular queue
    def deQueue(self):
        if self.head == -1:
            print("The circular queue is empty\n")

        elif self.head == self.tail:
            temp = self.queue[self.head]
            self.head = -1
            self.tail = -1
            return temp
        else:
            temp = self.queue[self.head]
            self.head = (self.head + 1) % self.max_size
            return temp

    def printQueue(self):
        if self.head == -1:
            print("No element in the circular queue")

        elif self.tail >= self.head:
            for i in range(self.head, self.tail + 1):
                print(self.queue[i], end=" ")
            print()
        else:
            for i in range(self.head, self.max_size):
                print(self.queue[i], end=" ")
            for i in range(0, self.tail + 1):
                print(self.queue[i], end=" ")
            print()

    def getQueue(self):
        data = []
        if self.head == -1:
            print("No element in the circular queue")

        elif self.tail >= self.head:
            for i in range(self.head, self.tail + 1):
                data.append(self.queue[i])

        else:
            for i in range(self.head, self.max_size):
                data.append(self.queue[i])
            for i in range(0, self.tail + 1):
                data.append(self.queue[i])
        return data


# # Your MyCircularQueue object will be instantiated and called as such:
# obj = MyCircularQueue(5)
# obj.enQueue(1)
# obj.enQueue(2)
# obj.enQueue(3)
# obj.enQueue(4)
# obj.enQueue(5)
# obj.deQueue()
# obj.enQueue(6)
# print("Initial queue")
# obj.printQueue()

# print(obj.getQueue())
# obj.deQueue()
# print("After removing an element from the queue")
# obj.printQueue()
# print(obj.getQueue())
