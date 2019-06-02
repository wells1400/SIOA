
class Vertex:
    # 初始化构造函数，name为字符串，connections字典为<Vertex(class), weight(fl)>
    # 前驱顶点pre，从起点距离distance，在Dijkstra执行赋值后有意义
    def __init__(self, name):
        self.id = name
        self.pre = None
        self.distance = float('inf')
        self.connections = dict()

    # 重载字符串化函数，返回字符串
    def __str__(self):
        return str(self.id) + " connected to: " + str([x.id for x in self.connections])

    # 增加相邻顶点，neighbour为Vertex类，weight为浮点型边权重
    def add_neighbour(self, neighbour, weight=0):
        self.connections[neighbour] = weight

    # 获取顶点id函数
    def get_id(self):
        return self.id

    # 获取顶点邻接点的函数，返回键值(Vertex)列表
    def get_connections(self):
        return self.connections.keys()

    # 获取顶点与邻接点边权重，传入Vertex类对象neighbour，返回weight(fl)
    def get_weight(self, neighbour):
        return self.connections[neighbour]

    # 获取顶点的距离（在BFS执行后使用）
    def get_distance(self):
        return self.distance

    # 获取前驱顶点（在Dijkstra执行后使用）
    def get_pre(self):
        return self.pre

    # 设定顶点的距离（在Dijkstra执行过程中调用）
    def set_distance(self, dist):
        self.distance = dist

    # 设定前驱顶点（在Dijkstra执行过程中调用）
    def set_pre(self, prev):
        self.pre = prev
class Graph(Vertex):
    # 无参数构造函数，vertex_dict为<id(str),Vertex>映射字典
    def __init__(self):
        self.vertex_dict = dict()
        self.vertex_num = 0

    # 增加顶点函数，传入新增顶点id
    def add_vertex(self, name):
        if name not in self.vertex_dict.keys():
            new_vertex = Vertex(name)
            self.vertex_dict[name] = new_vertex
            self.vertex_num = self.vertex_num + 1

    # 增加边函数， 传入顶点1名称、顶点2名称、权重
    def add_edge(self, vertex1, vertex2, weight):
        if vertex1 not in self.vertex_dict:
            self.add_vertex(vertex1)
        if vertex2 not in self.vertex_dict:
            self.add_vertex(vertex2)
        self.vertex_dict[vertex1].add_neighbour(self.vertex_dict[vertex2], weight)
        self.vertex_dict[vertex2].add_neighbour(self.vertex_dict[vertex1], weight)

    # 按照id检索顶点函数，传入id，返回Vertex类
    def get_vertex(self, name):
        if name in self.vertex_dict.keys():
            return self.vertex_dict[name]
        else:
            return None

    # 重载contains方法，传入id，返回bool值
    def __contains__(self, item):
        return item in self.vertex_dict

    # 重载迭代器，返回对应迭代器
    def __iter__(self):
        return iter(self.vertex_dict.values())

    # 重载字符串化方法，返回字符串
    def __str__(self):
        o_str = str()
        for item in self:
            o_str = o_str + str(item) + '\n'
        return o_str

    # Dijkstra算法，传入起点
    def dijkstra_search(self, start):
        # 优先队列priority
        priority = list(self.vertex_dict.values())
        # 起点距离置零
        start.set_distance(0)
        # 优先队列重排
        priority.sort(key=lambda x: x.get_distance(), reverse=True)
        while priority:
            # 重排标记changed，若存在顶点发生distance变化则标记为True
            changed = False
            # 弹出最高优先顶点current
            current = priority.pop()
            # 遍历current邻接顶点
            for vertex_tmp in current.get_connections():
                dist_tmp = current.get_distance() + current.get_weight(vertex_tmp)
                # 若发现优势路径则更改邻接顶点的distance和前驱顶点pre
                if dist_tmp < vertex_tmp.get_distance():
                    vertex_tmp.set_distance(dist_tmp)
                    vertex_tmp.set_pre(current)
                    changed = True
            # 若有更改则重排优先队列
            if changed:
                priority.sort(key=lambda x: x.get_distance(), reverse=True)
    # 读取顶点文件（适用于地铁路线例子的方法），传入文件路径path和线路编号subway
    def read_in(self, path):
        with open(path, 'r') as file:
            # 按行读取文件
            line = file.readline()
            station = None
            while line:
                info_list = line.split()
                # 当行中仅含有1个元素时，仅创建
                if len(info_list) == 1:
                    station = info_list[0]
                    self.add_vertex(station)
                # 当行中含有2个元素时，第1个元素为车站名称，第2个元素为距上个车站距离
                elif len(info_list) == 2:
                    pre_station = station
                    [station, distance] = info_list
                    distance = int(distance)
                    self.add_vertex(station)
                    self.add_edge(pre_station, station, distance)
                # 否则当行中含有3个元素，前2个元素为车站，第3个元素为前二者间距
                else:
                    [station1, station2, distance] = info_list
                    distance = int(distance)
                    self.add_edge(station1, station2, distance)
                line = file.readline()
# 回溯路径函数，传入参数终点destination，返回路径列表list(Vertex)
def reverse_trace(destination):
    current =Vertex(destination)
    trace = [destination]
    while current.get_pre():
        current = current.get_pre()
        trace.append(current)
    trace.reverse()
    return trace
# 初始化函数（适用于地铁路线例子的方法）
# 读入在file_path所示文件夹下的北京地铁线数据，返回生成的图
def initialize():
    graph = Graph()
    file_path = 'F:\Personal Documents\Project\BeijingSubway\line'
    for i in range(1, 3):
        path = file_path + str(i) + '.txt'
        graph.read_in(path)
    return graph
# 交互式菜单函数（适用于地铁路线例子的方法），传入图
def route_find_menu(graph):
    print('>> 寻找乘地铁最短路线，输入0退出')
    start = input('>> 输入起始站名：')
    while start != '0':
        graph.dijkstra_search(graph.get_vertex(start))
        destination = input('>> 输入终点站名：')
        if destination == '0':
            break
        trace_list = reverse_trace(graph.get_vertex(destination))
        print([x for x in trace_list])
        start = input('>> 输入起始站名：')
graph1=initialize()
route_find_menu(graph1)