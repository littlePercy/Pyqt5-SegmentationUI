使用Pyqt5搭建一个简易图像分割界面
====================================================================
PyQt5安装教程https://zhuanlan.zhihu.com/p/162866700

![Pyqt5](https://github.com/littlePercy/Pyqt5-SegmentationUI/assets/52816016/3cec0dc5-64e1-44f7-b68d-bd31c6ea30e5)


将 PyQt5 工程文件打包成可执行文件（.exe），以下是使用 PyInstaller 的步骤：

step1=>安装 PyInstaller: 打开终端或命令提示符，并运行以下命令：

pip install pyinstaller
====================================================================
step2=>在项目目录中运行 PyInstaller:打开命令提示符或终端，进入你的 PyQt5 项目目录。然后运行以下命令：

pyinstaller your_script.py
====================================================================
请将 your_script.py 替换为你实际的 PyQt5 脚本文件名。PyInstaller 将会在 dist 文件夹中生成一个包含你的应用程序可执行文件的文件夹。

step3=>查看生成的 .exe 文件：
在 dist 文件夹中，你会找到一个与你的脚本同名的文件夹，里面包含了可执行文件。这个可执行文件可以在没有 Python 解释器的情况下运行。
请注意，打包 PyQt5 应用程序可能会遇到一些问题，尤其是如果你的应用程序依赖于其他外部资源、文件或者特殊的 PyQt5 组件。在这种情况下，你可能需要在 PyInstaller 命令中添加一些选项，以确保所有依赖项都被正确地包含在最终的可执行文件中。
例如，如果你的应用程序使用了 Qt 的样式表文件（.qss 文件），你可能需要将这些文件复制到打包后的文件夹中，并在 PyInstaller 命令中添加相应的选项。

