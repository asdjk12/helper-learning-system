class SessionMemory():
    # 只记录当前会话中的一小部分内容
    # 滑动窗口 + 总结
    def __init__(self) -> None:
        self.N = 5      # editable by developer rather than users
        self.messages  = []

    def add(self, role, content):
        self.messages.append({"role":role, "content":content})
        self.messages = self.messages[-self.N]

    def recentMess(self):
        return self.messages