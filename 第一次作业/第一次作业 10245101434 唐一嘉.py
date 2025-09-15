from prettytable import PrettyTable  # 外部库


class Student:
    """
    学生信息类，用于存储单个学生的基本信息。

    Attributes:
        stunum (int): 学号
        name (str): 姓名
        sex (str): 性别
        dorm (str): 宿舍
        phonenum (int): 电话号码
    """

    def __init__(self, stunum=None, name=None, sex=None, dorm=None, phonenum=None):
        """
        初始化学生对象。

        Args:
            stunum (int): 学号
            name (str): 姓名
            sex (str): 性别
            dorm (str): 宿舍
            phonenum (int): 电话号码
        """
        self.stunum = stunum
        self.name = name
        self.sex = sex
        self.dorm = dorm
        self.phonenum = phonenum


class ManageList:
    """
    学生信息管理类，用于管理学生信息列表。

    Attributes:
        list (list): 存储学生对象的列表
    """

    def __init__(self):
        """初始化空的学生信息列表。"""
        self.list = []

    def add(self, student):
        """
        添加学生信息到列表中。

        Args:
            student (Student): 学生对象
        """
        self.list.append(student)

    def delete(self, stunum):
        """
        根据学号删除学生信息。

        Args:
            stunum (int): 要删除学生的学号

        Returns:
            bool: 删除成功返回True，未找到学生返回False
        """
        for i in self.list:
            if i.stunum == stunum:
                self.list.remove(i)
                return True
        return False

    def search(self, stunum):
        """
        根据学号查找学生信息。

        Args:
            stunum (int): 要查找学生的学号

        Returns:
            Student or bool: 找到返回学生对象，未找到返回False
        """
        for i in self.list:
            if i.stunum == stunum:
                return i
        return False


# 创建学生信息管理实例
managelist = ManageList()

# 主程序循环，提供用户交互界面
while True:
    # 显示功能菜单
    print("1. 按学号查找某一位学生的信息")
    print("2. 录入新的学生信息")
    print("3. 显示现有的所有学生信息")
    print("4. 删除现有的某一学生的信息")
    print("5. 退出程序")
    print("请输入功能对应数字：")
    opnum = int(input())
    # 功能1：按学号查找学生信息
    if opnum == 1:
        # 检查是否有学生信息
        if len(managelist.list) == 0:
            print("目前没有学生信息！")
            print("\n\n------------------------------------------")
            continue

        print("请输入学号：")
        # 输入验证，确保输入为整数
        while True:
            try:
                stunum = int(input())
                break
            except:
                print("请重新输入正确的学号！")

        # 查找学生信息并显示结果
        if managelist.search(stunum) == False:
            print("未找到该学生信息！")
        else:
            stu = managelist.search(stunum)
            searchtable = PrettyTable()
            searchtable.field_names = ["学号", "姓名", "性别", "宿舍", "电话号码"]
            searchtable.add_row([stu.stunum, stu.name, stu.sex, stu.dorm, stu.phonenum])
            print("查询成功！")
            print(searchtable)
        print("\n\n------------------------------------------")

    # 功能2：录入新的学生信息
    elif opnum == 2:
        newstudent = Student()
        print("请输入学号：")
        # 输入验证，确保学号为整数
        while True:
            try:
                newstudent.stunum = int(input())
                break
            except:
                print("请重新输入正确的学号！")
        print("请输入姓名：")
        newstudent.name = input()
        print("请输入性别：（男/女）")
        # 输入验证，确保性别为"男"或"女"
        while True:
            newstudent.sex = input()
            if newstudent.sex != "男" and newstudent.sex != "女":
                print("请重新输入正确的性别！（男/女）")
            else:
                break
        print("请输入宿舍：")
        newstudent.dorm = input()
        print("请输入电话号码：")
        # 输入验证，确保电话号码为整数
        while True:
            try:
                newstudent.phonenum = int(input())
                break
            except:
                print("请重新输入正确的电话号码！")
                continue
        managelist.add(newstudent)
        print("添加成功！")
        print("\n\n------------------------------------------")

    # 功能3：显示所有学生信息
    elif opnum == 3:
        # 检查是否有学生信息
        if len(managelist.list) == 0:
            print("目前没有学生信息！")
            print("\n\n------------------------------------------")
            continue

        # 使用表格格式显示所有学生信息
        table = PrettyTable()
        table.field_names = ["学号", "姓名", "性别", "宿舍", "电话号码"]
        for i in managelist.list:
            table.add_row([i.stunum, i.name, i.sex, i.dorm, i.phonenum])
        print(table)
        print("\n\n------------------------------------------")

    # 功能4：删除某一学生信息
    if opnum == 4:
        # 检查是否有学生信息
        if len(managelist.list) == 0:
            print("目前没有学生信息！")
            print("\n\n------------------------------------------")
            continue

        print("请输入学号：")
        # 输入验证，确保输入为整数
        while True:
            try:
                stunum = int(input())
                break
            except:
                print("请重新输入正确的学号！")

        # 查找学生信息并显示结果
        if managelist.search(stunum) == False:
            print("未找到该学生！")
        else:
            managelist.delete(stunum)
            print("删除成功！")
        print("\n\n------------------------------------------")

    # 功能5：退出程序
    elif opnum == 5:
        print("程序退出")
        break
