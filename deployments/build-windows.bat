@echo off
REM Windows XConnector 构建脚本
REM 需要安装 Docker Desktop

echo ====================================
echo XConnector Windows 构建工具
echo ====================================

REM 检查 Docker 是否安装
docker --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未检测到 Docker Desktop
    echo 请先安装 Docker Desktop: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

echo Docker 检测成功

REM 设置变量
set XCONNECTOR_VERSION=latest
set BUILD_DIR=%~dp0
set PROJECT_ROOT=%BUILD_DIR%\..\..

echo 项目根目录: %PROJECT_ROOT%
echo 构建目录: %BUILD_DIR%

REM 进入项目根目录
cd /d "%PROJECT_ROOT%"

REM 检查必要文件
if not exist "xconnector\" (
    echo 错误: 未找到 xconnector 目录
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo 错误: 未找到 requirements.txt
    pause
    exit /b 1
)

echo 开始构建 XConnector 服务镜像...

REM 构建 XConnector 服务镜像
docker build -f deployments\docker\Dockerfile.xconnector-service -t xconnector-service:%XCONNECTOR_VERSION% .

if errorlevel 1 (
    echo 构建失败!
    pause
    exit /b 1
)

echo 构建成功!

REM 创建导出目录
if not exist "docker-images" mkdir docker-images

echo 导出镜像到文件...

REM 导出 XConnector 镜像
docker save xconnector-service:%XCONNECTOR_VERSION% | gzip > docker-images\xconnector-service_%XCONNECTOR_VERSION%.tar.gz

REM 拉取并导出依赖镜像
echo 拉取依赖镜像...

docker pull quay.io/coreos/etcd:v3.5.9
docker save quay.io/coreos/etcd:v3.5.9 | gzip > docker-images\etcd_v3.5.9.tar.gz

docker pull nats:2.10-alpine
docker save nats:2.10-alpine | gzip > docker-images\nats_2.10-alpine.tar.gz

echo 镜像导出完成!
echo 文件位置: docker-images\

dir docker-images\

REM 创建部署包
echo 创建部署包...

set TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

set PACKAGE_NAME=xconnector-deployment-%TIMESTAMP%

if not exist "%PACKAGE_NAME%" mkdir "%PACKAGE_NAME%"

REM 复制文件到部署包
xcopy /E /I deployments\docker "%PACKAGE_NAME%\docker\"
xcopy /E /I docker-images "%PACKAGE_NAME%\docker-images\"
copy deployments\scripts\* "%PACKAGE_NAME%\" 2>nul

REM 创建 Windows 部署脚本
echo @echo off > "%PACKAGE_NAME%\deploy-windows.bat"
echo echo 正在加载 Docker 镜像... >> "%PACKAGE_NAME%\deploy-windows.bat"
echo. >> "%PACKAGE_NAME%\deploy-windows.bat"
echo cd docker-images >> "%PACKAGE_NAME%\deploy-windows.bat"
echo for %%%%f in (*.tar.gz) do ( >> "%PACKAGE_NAME%\deploy-windows.bat"
echo     echo 加载 %%%%f >> "%PACKAGE_NAME%\deploy-windows.bat"
echo     docker load ^< %%%%f >> "%PACKAGE_NAME%\deploy-windows.bat"
echo ) >> "%PACKAGE_NAME%\deploy-windows.bat"
echo. >> "%PACKAGE_NAME%\deploy-windows.bat"
echo cd ..\docker >> "%PACKAGE_NAME%\deploy-windows.bat"
echo echo 启动服务... >> "%PACKAGE_NAME%\deploy-windows.bat"
echo docker-compose up -d >> "%PACKAGE_NAME%\deploy-windows.bat"
echo echo 部署完成! >> "%PACKAGE_NAME%\deploy-windows.bat"
echo pause >> "%PACKAGE_NAME%\deploy-windows.bat"

REM 创建 README
echo # XConnector 部署包 > "%PACKAGE_NAME%\README.md"
echo. >> "%PACKAGE_NAME%\README.md"
echo 构建时间: %date% %time% >> "%PACKAGE_NAME%\README.md"
echo. >> "%PACKAGE_NAME%\README.md"
echo ## 部署步骤: >> "%PACKAGE_NAME%\README.md"
echo 1. 将整个文件夹复制到目标服务器 >> "%PACKAGE_NAME%\README.md"
echo 2. 在目标服务器上运行 deploy-windows.bat (Windows) 或 deploy-linux.sh (Linux) >> "%PACKAGE_NAME%\README.md"

REM 创建 Linux 部署脚本
echo #!/bin/bash > "%PACKAGE_NAME%\deploy-linux.sh"
echo echo "正在加载 Docker 镜像..." >> "%PACKAGE_NAME%\deploy-linux.sh"
echo cd docker-images >> "%PACKAGE_NAME%\deploy-linux.sh"
echo for f in *.tar.gz; do >> "%PACKAGE_NAME%\deploy-linux.sh"
echo   echo "加载 $f" >> "%PACKAGE_NAME%\deploy-linux.sh"
echo   gunzip -c "$f" ^| docker load >> "%PACKAGE_NAME%\deploy-linux.sh"
echo done >> "%PACKAGE_NAME%\deploy-linux.sh"
echo cd ../docker >> "%PACKAGE_NAME%\deploy-linux.sh"
echo echo "启动服务..." >> "%PACKAGE_NAME%\deploy-linux.sh"
echo docker-compose up -d >> "%PACKAGE_NAME%\deploy-linux.sh"
echo echo "部署完成!" >> "%PACKAGE_NAME%\deploy-linux.sh"

echo 部署包创建完成: %PACKAGE_NAME%

echo.
echo ====================================
echo 构建完成!
echo ====================================
echo.
echo 生成的文件:
echo - Docker 镜像: docker-images\
echo - 部署包: %PACKAGE_NAME%\
echo.
echo 下一步:
echo 1. 将 %PACKAGE_NAME% 文件夹复制到服务器
echo 2. 在服务器上运行相应的部署脚本
echo.

pause