---
title: Install Pentaho with GUI on EC2 Linux Instance
date: 2017-03-09 16:52:09
tags: 
- Linux
- Data Warehousing
category: 
- 时习之
- Miscellaneous
description: 记录一下Pentaho及VNCServer安装过程
---

In this post I will introduce how to install and visit the GUI interface from VNC client. I have been sitting here for 4 hours and were so frustrated for all the BUGS I saw.. Hopefully this post will help you get this done within one hour. 

Now let's get started!!

#### Security Group<br>
When you run an instance, please edit the inbound traffic rules of security group:
Protocol: Customed TCP/IP, Port **5901**. <br>
This step is critical for setting up VNC server which we will introduce later. See <a href = 'https://survivalguides.wordpress.com/2010/08/16/vnc-server-configuration/'>ref</a>.


#### Install Java
Pentaho runs on Java. Installation instruction goes <a href = 'https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-get-on-ubuntu-16-04'>here</a>.<br>Please, please check you've installed **the SAME version of JDK and JRE**!  If you get error message <a href = 'http://stackoverflow.com/questions/37612054/what-does-unsupported-major-minor-version-52-0-mean-and-how-do-i-fix-it'>"Unsupported major.minor version 52.0"</a> then it's very likely the version doesn't match. At least it was the problem for me. Well it's also possible that there is something wrong with yout /etc/environment : either you didn't set up JAVA_HOME and Path; or you forgot to run ``source /etc/environment`` to confirm your setup.

#### Download the bin
File is available at <a href = 'http://www.pentaho.com/download'> here </a>. If you can't download with ``wget``; try transfer downloads to your instance using filezilla. It took only half an hour to download and transfer to instance. 
After downloading (& transfering), run 
`chmod a+x pentaho-business-analytics-7.0.0-x64.bin`<br>
`./pentaho-business-analytics-7.0.0-x64.bin`

#### Install VNC server on server side
Steps as in <a href = 'http://stackoverflow.com/questions/25657596/how-to-set-up-gui-on-amazon-ec2-ubuntu-server'>Reference</a>:
``` 
sudo useradd -m pentaho
sudo passwd pentaho
sudo usermod -aG admin pentaho
sudo apt-get update
sudo apt-get install ubuntu-desktop
sudo apt-get install vnc4server
su - pentaho
vncserver
vncserver -kill :1
vim /home/pentaho/.vnc/xstartup
```
and then change the document xstartup tp:
```
#!/bin/sh

export XKL_XMODMAP_DISABLE=1
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS

[ -x /etc/vnc/xstartup ] && exec /etc/vnc/xstartup
[ -r $HOME/.Xresources ] && xrdb $HOME/.Xresources
xsetroot -solid grey
vncconfig -iconic &

gnome-panel &
gnome-settings-daemon &
metacity &
nautilus &
gnome-terminal &
```
start VNC server again. 

**Very Important: user name to start VNC server should be the same as username to run ./spoon.sh**. Else will encounter <a href = 'http://unix.stackexchange.com/questions/140113/org-eclipse-swt-swterror-no-more-handles-gtk-init-check-failed-while-runnin?rq=1'>org.eclipse.swt.SWTError</a>.

#### Download VNC client
I found <a href = 'https://www.realvnc.com/download/viewer/'> RealVNC</a> simple and good to use. After installation, open:
${your public DNS}**:1**

#### Open Pentaho
In VNC client, open terminal. 
Navigate using 
`cd /computer/home/pentaho/Pentaho/design-tools/data-integration/`
Final step: run 
`./spoon.sh`

#### DONE!!!!