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
yarn install
```

## Boost PIP Install using Domestic Servers

``` console
pip install -i https://pypi.douban.com/simple/ <package-name>
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package-name>
```

Or update local PIP ini

- %APPDATA%\pip\pip.ini
- %HOME%\pip\pip.ini

``` pip.ini
[global]
timeout = 60
index-url = http://pypi.douban.com\simple
trusted-host = pypi.douban.com
```

Ref: <https://blog.csdn.net/QLeelq/article/details/121197098>

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

* <https://blog.csdn.net/Apple_xiaoli/article/details/104533008>
* <https://cloud.tencent.com/developer/article/1440422>
* <https://openbase.com/python/jupyterthemes/documentation>
* <https://github.com/dunovank/jupyter-themes/blob/master/README.md#monospace-fonts-code-cells>

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
