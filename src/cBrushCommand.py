import maya.api.OpenMaya as om


class CallbackCmd(om.MPxCommand):
    COMMAND_NAME = "cCallbackCmd"

    # 万能快递柜：专门收 partial 函数
    _staging_data = None

    def __init__(self):
        super(CallbackCmd, self).__init__()
        # 你的 partial 函数会存在这里
        self.undo_func = None
        self.redo_func = None

    def doIt(self, args):
        if CallbackCmd._staging_data is None:
            return

        # 接收你传进来的 partial 函数
        data = CallbackCmd._staging_data
        CallbackCmd._staging_data = None

        self.undo_func = data.get("undo")
        self.redo_func = data.get("redo")

        # 第一次执行直接调 redo
        self.redoIt()

    def undoIt(self):
        if self.undo_func:
            self.undo_func()  # 💥 直接无脑执行你的 partial！

    def redoIt(self):
        if self.redo_func:
            self.redo_func()  # 💥 直接无脑执行你的 partial！

    def isUndoable(self):
        return True

    @classmethod
    def creator(cls):
        return CallbackCmd()
