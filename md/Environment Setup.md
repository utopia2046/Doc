# Environment Setup

## Clone github repro in SSH

### 1. Generate SSH public/private key pairs

Open git command console, under home path ~

``` console
ssh-keygen -t rsa -C "utopia2046@hotmail.com"
```

The pass phrase and filename input could be ignored. That would use default file names and with empty pass phrase

Check the generated id_rsa (private key) and id_rsa.pub (public key) files under .ssh folder. Copy content from id_rsa.pub.

Test the generated key pair using:

``` console
ssh -T git@github.com
```

When seeing below warning, input yes to continue.

``` console
This key is not known by any other names
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
```

### 2. Add SSH key on Github website

Sign in Github website and go to <https://github.com/settings/keys> page. Click New SSH Key, give it a name and paste the public key content.

### 3. Clone repo to local

Go to your repro, copy clone string from Code menue SSH tab like below, and run in Git console:

``` console
git clone git@github.com:utopia2046/Doc.git
```

Reference:

<https://blog.csdn.net/felicity294250051/article/details/53606158>

### Enable Long Path

``` console
git config --system core.longpaths true
```

In registry, set
Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
LongPathsEnabled=1

## Install Common Packages

Install Node.js, then install common packages like:

``` console
npm install --global gulp
npm install --global yarn
npm install -g eslint
npm install -g typescript
npm install -g ts-node
yarn install
```

## Boost PIP Install using Domestic Servers

- [阿里云](http://mirrors.aliyun.com/pypi/simple/)
- [豆瓣](http://pypi.douban.com/simple/)
- [清华大学](https://pypi.tuna.tsinghua.edu.cn/simple/)
- [中国科技大学](https://pypi.mirrors.ustc.edu.cn/simple/)
- [中国科学技术大学](http://pypi.mirrors.ustc.edu.cn/simple/)

``` console
pip install -i https://pypi.douban.com/simple/ <package-name>
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package-name>
```

Or update local PIP ini

- %APPDATA%\pip\pip.ini
- %HOMEPATH%\pip\pip.ini

``` pip.ini
[global]
timeout = 60
index-url = https://pypi.douban.com/simple
trusted-host = pypi.douban.com
```

Ref: <https://blog.csdn.net/QLeelq/article/details/121197098>

## Install PyTorch using Conda

<https://pytorch.org/>

``` console
conda update -n base -c defaults conda
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Or download package from <https://download.pytorch.org/whl/torch_stable.html>
and use `pip install <package_name>.whl`

Check package version after install

``` python
import torch
print(torch.__version__)
# 2.0.0
import torchvision
print(torchvision.__version__)
# 0.15.0
```

Import PyTorch in Jupyter Notebook

``` console
conda install ipykernel
python -m ipykernel install --name pytorch --display-name "PyTorch for Deep Learning"
jupyter notebook --generate-config
```

Ref:

- <https://blog.csdn.net/m0_52571323/article/details/110222966>
- <https://blog.csdn.net/qq_38140292/article/details/114157146>

## Create Python Virtual Environment

``` console
pip install virtualenv
pip install virtualenvwrapper
virtualenv <env-name>
mkvirtualenv <env-name>
virtualend --python C:\Python311 <env-name>   # assign python version
virtualend --system-site-packages <env-name>  # depends on system packages
# active virtual env
source <env-name>/bin/activate
workon <env-name>
# deactivate
deactivate
# list virtual envs
lsvirtualenv
# check installed packages on current virtual env
lssitepackages
# remove virtual env
rmvirtualenv
```

Ref: <https://zhuanlan.zhihu.com/p/338424040>

## Install TensorFlow in Virtual Environment

using pip

``` console
virtualenv –system-site-packages -p python3 tensorflow
source tensorflow/bin/activate
pip install –upgrade tensorflow
```

using conda

``` console
conda create -n TensorFlow python=3.9
conda info --envs
conda activate TensorFlow
conda install tensorflow
```

validate installation

``` python
import tensorflow as tf
tf.__version__
hello = tf.constant('Hello tensorfolw')
sess = tf.Session()
print(sess.run(hello))
```

add virtualenv in jupyter kernel

``` console
ipython kernel install --user --name=TensorFlow
```

Ref:

- <https://blog.csdn.net/weixin_42555080/article/details/100704078>
- <https://blog.csdn.net/kevindree/article/details/88627830>
- <https://queirozf.com/entries/jupyter-kernels-how-to-add-change-remove>

## Run Py file in Jupyter & IPython

``` ipython
%run -i list_file.py [args]
```

## Customize Jupyter Themes

1. Install `Jupyter` package from Douban server inside GFW, the download speed is much faster than from origin.
2. Notice that the notebook is not using local fonts, instead it has a pre-defined font list at <https://openbase.com/python/jupyterthemes/documentation>.
3. More style could be adjusted in css file `C:\Users\utopi\.jupyter\custom\custom.css`;

``` console
pip install -i https://pypi.doubanio.com/simple/ jupyterthemes
# list all themes
jt -l
# set theme and font
jt -t oceans16 -f source -nf opensans -tf source -N -T -cellw 60%
# revert to default
jt -r
```

``` python
# In Jupyter notebook, set plot style
from jupyterthemes import jtplot
jtplot.style()
```

References:

- <https://blog.csdn.net/Apple_xiaoli/article/details/104533008>
- <https://cloud.tencent.com/developer/article/1440422>
- <https://openbase.com/python/jupyterthemes/documentation>
- <https://github.com/dunovank/jupyter-themes/blob/master/README.md#monospace-fonts-code-cells>

## Trouble shooting

``` console
kex_exchange_identification: Connection closed by remote host
Connection closed by 20.205.243.166 port 22
fatal: Could not read from remote repository.

Please make sure you have the correct access rights
and the repository exists.
```

Solution:

1. Regenerate a new key.
2. Change SSH port.

Create {user}/.ssh/config file with following content

``` config
Host github.com
Hostname ssh.github.com
Port 443
```

Reference: <https://blog.csdn.net/Eric_q8666/article/details/127179501>

### Clean up unused packages and cache

Reference: <https://docs.conda.io/projects/conda/en/latest/commands/clean.html>

``` console
conda clean -a
```

``` console
pip list --not-required
pip uninstall --yes <package_name>
...
pip cache purge
```

### Re-install OS

Fix the "We Couldn't Create a New Partition" Error

1. Remove (physical or disable in BIOS) all external and internal storage devices except the target SSD to install OS and the bootable USB;
2. Try plug bootable USB in USB 2.0 rather than 3.0 slot;
3. In windows installation, press Shift+F11 to open console, or open it in Trouble shooting -> Advanced options -> Command;
4. In console, use `diskpart` util to create partition, convert it to GPT format, activate it, and format it;

    ``` console
    diskpart
    list disk
    select disk #
    clean                       // optional, clean all partitions on this disk
    convert gpt                 // optional, change disk format from MBR to GPT
    create partition primary    // optional if not re-partitioning
    list partition
    select partition #
    active                      // set selected partition as active
    format fs=ntfs quick        // optional, fast format as NTFS
    assign                      // assign a disk symbol
    exit
    ```

5. Retry installation.

### Clean up Honor Built-In Apps

荣耀手机更新多了一堆应用，智慧XX，荣耀XX，全是广告乌烟瘴气且不能卸载，深厌之。

1. Download abd from:

- https://developer.android.google.cn/tools/releases/platform-tools, or
- https://adbshell.com

2. Enable USB debugging on phone

- 打开开发者模式
- 打开 USB 调试
- 仅充电模式下允许 ADB 调试
- 打开 USB 共享网络
- 选择 USB 配置为 RNDIS (USB 以太网)

3. Install HDB dirver on windows

- Connect phone to PC, open Device manager
- 其他设备下，找到带黄色惊叹号的 HDB interface，更新驱动程序，浏览我的电脑，从可用驱动程序列表中选择，Android Phone -> Huawei HDB Interface，无视警告，安装
- 安装完成后，Android Phone -> Huawei HDB Interface，右键，属性，详细信息，硬件ID，VID_XXXX，后面这个四位的就是 VID
- 在 C:\Users\{current_user}\.android\ 目录下新建 adb_usb.ini，写入一行内容 0xXXXX (刚才那个四位的VID)

4. Start abd tool

- Extract adb package
- In command line, run

```
adb start-server
adb devices
```

没找到 device 的话就试试插拔 USB 线，删除 USB debugging 授权再插，重启 PC，重启 ADB server，查找是否有 5037 端口冲突等等

```
adb kill-server
adb start-server
adb devices

adb nodaemon server
netstat -ano | findstr "5037"
```

找到设备后，列出所有 app 清单

```
adb shell pm list packages
```

卸载流氓软件，以 智能助手(负一屏) 为例

```
adb shell pm uninstall --user 0 com.huawei.intelligent
```

Examples:

adb shell pm uninstall --user 0 cn.HONOR.qinxuan            # 荣耀亲选
adb shell pm uninstall --user 0 com.huawei.gamebox          # 游戏中心
adb shell pm uninstall --user 0 com.huawei.gameassistant    # 应用助手
adb shell pm uninstall --user 0 com.huawei.hifolder         # 精品推荐
adb shell pm uninstall --user 0 com.huawei.fastapp          # 快应用中心
adb shell pm uninstall --user 0 com.huawei.tips             # 智能提醒
adb shell pm uninstall --user 0 com.huawei.scanner          # 智慧视觉
adb shell pm uninstall --user 0 com.huawei.hwvoipservice    # 智能电话
adb shell pm uninstall --user 0 com.huawei.search           # 智慧搜索
adb shell pm uninstall --user 0 com.huawei.hwdetectrepair   # 智能检测
adb shell pm uninstall --user 0 com.huawei.vassistant       # 语音助手
adb shell pm uninstall --user 0 com.huawei.skytone          # 天际通数据服务
adb shell pm uninstall --user 0 com.huawei.hiai             # 华为智慧引擎
adb shell pm uninstall --user 0 com.huawei.hitouch          # 智慧识屏
adb shell pm uninstall --user 0 com.huawei.bd               # 用户改进体验计划
adb shell pm uninstall --user 0 com.huawei.intelligent      # 负一屏

Reference:

https://www.bilibili.com/read/cv21078097/
