# xconnector

请补充说明内容

## 常用git命令

1.git global setup

git config --global user.name 'test'

git config --global user.email 'xxx@email.com'

2.克隆代码至本地

git clone <仓库克隆地址>

git clone http://10.64.46.151:18080/aipaas/InferenceEngine/xconnector.git

git clone ssh://git@10.64.46.151:54322/aipaas/InferenceEngine/xconnector.git

3.克隆指定分支

git clone -b <指定分支名> <远程仓库地址>，如: git clone -b dev http://10.64.46.151:18080/aipaas/InferenceEngine/xconnector.git

4.查看分支

git branch  // 查看所有本地分支

git branch -a //查看本地和远程所有分支

git branch  -r //查看所有远程分支

5.切换分支

git checkout <指定分支名>，如：git checkout dev //切换到指定分支

git checkout -b <指定分支名>  //新建分支，并切换到该分支

6.拉取代码

git pull

7.将本地修改的文件xx添加到暂存区

git add <文件名称>， 如：git add test01

git add -A  提交所有变化

git add -u  提交被修改(modified)和被删除(deleted)文件，不包括新文件(new)

git add .  提交新文件(new)和被修改(modified)文件，不包括被删除(deleted)文件

8.提交暂存区的内容

git commit -m "注释"，如： git commit -m "feat: devops-xxxx, 新增文件xxx，完成xx功能"

9.推送代码

git push //将提交的文件推送到远端仓库

git push --set-upstream origin <分支名称>  // 若该分支远端gitlab中不存在，则使用该命令进行推送

 