# python 环境配置
- **虚拟环境的作用**：为每个项目隔离 Python 解释器与第三方依赖，避免不同项目的包版本互相影响。

## 创建虚拟环境

- 在项目根目录执行：`python -m venv .venv`
- `.venv` 是常用的环境目录名，也可以替换成 `venv`、`env` 等名称。

## 激活虚拟环境

- macOS / Linux：`source .venv/bin/activate`

## 在虚拟环境中安装依赖

- 安装包：`python -m pip install 包名`
- 查看已安装的包：`python -m pip list`
- 根据依赖文件安装：`python -m pip install -r requirements.txt`
- 建议使用 `python -m pip`，以确保调用的是当前虚拟环境中的 pip。

## 导出项目依赖

安装或更新第三方包后，可以把当前虚拟环境中的包及其版本保存到 `requirements.txt`：

```bash
python -m pip freeze > requirements.txt
```

- `pip freeze` 会输出当前虚拟环境中已安装的包及其具体版本。
- `>` 会把输出写入项目根目录下的 `requirements.txt`；如果文件已经存在，其原有内容会被覆盖。
- 将 `requirements.txt` 提交到代码仓库后，其他开发者可以安装相同版本的依赖：

```bash
python -m pip install -r requirements.txt
```

执行命令前应先激活项目的虚拟环境，避免把全局环境或其他项目的依赖写入文件。

### 使用 pipreqs 导出项目依赖

`pip freeze` 会导出虚拟环境中的所有包，而 `pipreqs` 会扫描项目代码中的 `import` 语句，只导出项目实际使用的第三方依赖，生成的 `requirements.txt` 通常更加精简。

先在已激活的虚拟环境中安装 `pipreqs`：

```bash
python -m pip install pipreqs
```

然后在项目根目录执行：

```bash
pipreqs . --ignore .venv --force
```

- `.`：扫描当前项目目录。
- `--ignore .venv`：忽略虚拟环境目录，避免扫描其中的第三方包源码。
- `--force`：如果 `requirements.txt` 已经存在，直接覆盖原文件。

生成文件后，可以使用以下命令安装依赖：

```bash
python -m pip install -r requirements.txt
```

`pipreqs` 根据代码中的导入语句推断依赖，可能无法识别动态导入或某些“导入名与安装包名不同”的依赖。生成后应检查 `requirements.txt`，并通过重新创建虚拟环境或运行项目来验证依赖是否完整。

## 退出与重新进入

- 退出虚拟环境：`deactivate`
- 下次进入项目后，重新执行：`source .venv/bin/activate`
