---
layout: post
title: 远程访问Jupyter notebook/lab计算资源
subtitle: Enable remote accsss for Jupyter notebook/lab server
tags: [Jupyter]
comments: false
mathjax: false
---

Jupyter Notebook是免费开源、基于网页、支持多种编程语言的交互式计算工具，可以在里面自由添加动态代码、描述性文字、公式、图片、动画等内容，让数据处理和可视化变得生动而有条理。而Jupyter Lab是以Jupyter Notebook继任者定位开发的新一代产品，在保留Notebook主要功能的基础上加入了一些新特性。

![Jupyter notebook](../img/postimg_jupyterpreview.png "Jupyter notebook")

> 软件安装：以常用的编程语言Python为例，除了使用`pip`安装Jupyter的方法之外，直接安装面向数据科学、集成了Jupyter等模块的Python发行版[Anaconda](https://www.anaconda.com)也是比较方便的做法。

启动Jupyter Notebook/Lab程序之后，Windows系统下默认会启动浏览器并定位到导航菜单页面，也可以自行根据后台提示粘贴网址到浏览器打开。用户可以通过该页面创建.ipynb格式的notebook文件，并在上面进行编写文案或处理数据。

关于Jupyter Notebook/Lab的具体使用方法，可以参看官方文档。

这里要介绍的是如何让Jupyter Notebook/Lab启用远程接入，以便在不同终端不同场合下都能够使用某一台高配置主机的计算能力，即搭建为一个自用的云计算平台。

默认情况下Jupyter Notebook/Lab只能本地（localhost）访问，即运行了Jupyter软件的电脑可以通过访问地址`http://127.0.0.1:8888`与Python（或R等其它语言）内核交互。假设同一内网路由下，Jupyter主机IP地址为`192.168.1.10`，另外一台电脑（假设IP为`192.168.1.100`）默认是无法通过浏览器访问`http://192.168.1.10:8888`来使用Jupyter主机的计算能力的。

开启远程访问并设置密码需要以下设置流程：

1. Anaconda Prompt终端中使用命令`jupyter notebook --generate-config`生成默认配制文件。
2. 根据显示的配置文件位置，打开配置文件并进行编辑，将`#c.NotebookApp.ip = 'localhost'`改为`c.NotebookApp.ip = '0.0.0.0'`，然后保存退出。
3. Anaconda Prompt终端中使用命令`jupyter notebook password`为访问加入密码验证。
4. Anaconda Prompt终端中使用命令`jupyter notebook`启动服务。
5. 其它设备即可通过访问服务主机的IP地址加端口号(如`http://192.168.1.10:8888`)使用它的计算资源。

如果服务器具备公网可以访问的IP或域名，那么可访问对象将不受限于内网设备。
