import sys
from shutil import get_terminal_size
import time


class ProgressBar(object):
    '''A progress bar which can print the progress
    这段代码定义了一个进度条类 ProgressBar，用于在命令行界面中显示任务的进度。
    在控制台应用中使用，能够动态显示任务的进度和预计完成时间，使得用户在执行长时间运行的任务时可以清楚地看到进度。通过合理的终端宽度管理，确保了在不同终端下都能良好显示。
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    """
        初始化进度条，接受以下参数：
        task_num：总任务数量（默认为0）。
        bar_width：进度条的宽度（默认为50）。
        start：是否立即开始（默认为True）。
        self.task_num：记录总任务数。
        self.bar_width：计算并设置进度条宽度，确保不超过终端的最大宽度。
        self.completed：初始化已完成的任务数。
        如果 start 为 True，调用 start() 方法开始进度条。
    """

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width
        """
        获取终端的宽度，计算最大进度条宽度，确保它小于终端宽度的60%并且留出50个字符的空间。
        如果终端宽度过小，给出提示并将最大宽度设为10。
        """
    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()
        """
        输出进度条的初始状态，显示已完成任务数和预计剩余时间（ETA）。
        记录开始时间以计算任务耗时。
        """
    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '█' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()
        """
        每次调用 update 方法时，增加已完成任务计数，并计算经过的时间和完成速率（FPS）。
        如果设置了总任务数，计算进度百分比和预计剩余时间（ETA），并构造进度条的字符表示。
        使用 ANSI 转义序列移动光标和清除行，更新进度条的显示。
        """
