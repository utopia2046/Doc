# Visual Studio Code 指南阅读笔记

Visual Studio Code 是基于 Electron 开发框架开发的。Electron 是一个使用 JavaScript、HTML 和 CSS 构建桌面应用程序的框架。通过嵌入 Chromium 和 Node.js，允许开发者使用 JavaScript 创建跨平台应用（包括 Windows，macOS 和 Linux）。

[Electron Documents](https://www.electronjs.org/docs/latest/)

核心组件

- Electron: 跨平台
- Monaco Editor: Web-based editor
- TypeScript: strong type for JavaScript
- Language Server Protocol (LSP) + Debug Adapter Protocol (DAP): 解耦前端界面与后端语言功能
- Xterm.js: integrated terminal

## Settings: (Ctrl+,)

1. User Settings (global, %APPDATA%\Code\User\settings.json, 可以用 Settings 右上角 Open Settings (JSON) 按钮打开或者 Ctrl+Shift+P 搜索 Settings (JSON))
2. Workspace Settings (override user settings, 保存在当前 workspace 的 .code-workspace 文件里)
3. Folder Settings (保存在 .\.vscode\settings.json 里)
4. Language Specific Settings (Ctrl+Shift+P 或 F1, Language Specific Settings, @lang:python)

### Common settings

``` json
"editor.fontFamily": "'Cascadia Mono', Consolas, 'Courier New', monospace",
"editor.fontSize": 14,
"editor.insertSpaces": true,
"editor.tabSize": 4,
"editor.tabCompletion": "on",
"editor.renderWhitespace": "all",
"editor.renderControlCharacters": true,
"editor.trimAutoWhitespace": true,
"editor.formatOnPaste": true,
"editor.formatOnSave": true,
"gulp.autoDetect": "on",
```

## 编辑功能

- 多光标功能：Alt+Click，增加新的光标，同时编辑多个光标处
- 列编辑：Shift+Alt+MouseDrag，或者鼠标中键拖拽
- 格式化：Shift+Alt+F，格式化当前文件；Ctrl+K,Ctrl+F，格式化选中部分
- 当前文件语言和编码：可以用右下角状态栏按钮改变 Language Mode 和 Encoding
- 快捷键：Ctrl+K,Ctrl+S
- IntelliSense: Ctrl+Space
- Go to symbol: Ctrl+P,@: 或者 Ctrl+Shift+O,:

![Visual Studio Code Keyboard Shortcuts](../images/VisualStudioCode%20Shortcuts.gif)

## 集成终端 Ctrl+`

[设置 Anaconda Prompt 为默认 terminal](https://blog.csdn.net/god_wen/article/details/99450356)

``` json
"terminal.integrated.defaultProfile.windows": "Windows PowerShell",
"terminal.integrated.cwd": "D:\\Projects",

"[python]": {
    "editor.formatOnType": true
},
"python.pythonPath": "C:\\Users\\utopi\\anaconda3\\python.exe",
"python.languageServer": "Jedi",
```

## 命令行

用 Visual Studio Code 打开当前目录: `code .`
参数：
-n, --new-window
-r, --reuse-window
-g, `-goto file[:line[:character]]`, example: `code --goto package.json:10:5`
-d, `--diff <file1> <file2>`

## Run Tasks

Reference: <https://code.visualstudio.com/Docs/editor/tasks>
Run Build Task: `Ctrl+Shift+B`
Configuration: .vscode\tasks.json

Example

``` json
{
  "version": "2.0.0",
  "tasks": [
    {
      "type": "markdownlint",
      "problemMatcher": [
        "$markdownlint"
      ],
      "label": "markdownlint: Lint all Markdown files in the workspace with markdownlint",
      "group": {
        "kind": "build",
        "isDefault": true
      }
    }
  ]
}
```

Next build error: F8
Previous build error: Shift+F8

Turn `"gulp.autoDetect": "on"` in settings.json, then `Ctrl+Shift+P, Task: Run Tasks, gulp:default` to run gulp
Other tasks, like npm, eslint, are similar.
For frequently used task, add Keyboard Shortcuts in `keybindings.json` to save time.

## 常用插件

### REST Client

- 类似 Postman，可以发 HTTP request，多个 request 可以写在一个 .rest 或 .http 文件里，用 ### 分隔
- 在 Visual Studio Code 内，可以用 Send Request link，右键菜单，或者 Ctrl+Alt+R 发 request
- 右键菜单或 Ctrl+Alt+C 可以生成常用语言的 HTTP request 代码
Ref: <https://marketplace.visualstudio.com/items?itemName=humao.rest-client>

``` rest
@baseUrl = https://example.com/api

# @name login
POST {{baseUrl}}/api/login HTTP/1.1
Content-Type: application/x-www-form-urlencoded

name=foo&password=bar

###

@authToken = {{login.response.headers.X-AuthToken}}

# @name createComment
POST {{baseUrl}}/comments HTTP/1.1
Authorization: {{authToken}}
Content-Type: application/json

{
    "content": "fake content"
}

###

@commentId = {{createComment.response.body.$.id}}

# @name getCreatedComment
GET {{baseUrl}}/comments/{{commentId}} HTTP/1.1
Authorization: {{authToken}}

###

# @name getReplies
GET {{baseUrl}}/comments/{{commentId}}/replies HTTP/1.1
Accept: application/xml

###

# @name getFirstReply
GET {{baseUrl}}/comments/{{commentId}}/replies/{{getReplies.response.body.//reply[1]/@id}}
```

## 常用语言

### Python

- Ctrl+Shift+P, Python: Select Interpreter, black (or autopep8), Python: Select Linter, pylint (or flake8)
- 选择一行或多行代码，Shift+Enter, 等于右键 Run Python -> Run Selection/Line in Python Terminal
- 右键，Sort Imports
- autoDocstring, 快速生成文档字符串

### JavaScript

- JSDoc: 在函数上方输入 /**，触发代码片段提示
- Enable Imports Organize & CodeLens:

  ``` json
  "editor.codeActionsOnSave": {
      "source.organizeImports": true
  },
  "javascript.updateImportsOnFileMove.enabled": "prompt",
  "javascript.referencesCodeLens.enabled": true,
  "javascript.referencesCodeLens.showOnAllFunctions": true
  ```

### TypeScript

- Run `tsc file.ts` to compile the file to .js
- In launch.json or user settings.json, set code runner to be Node.js, then F5 to start debugging

``` json
    // in launch.json
    "configurations": [
        {
            "name": "Launch Program",
            "program": "${workspaceFolder}/app.js",
            "request": "launch",
            "skipFiles": [
                "<node_internals>/**"
            ],
            "type": "node"
        }
    ]
    // in settings.json
    "launch": {
        "version": "0.2.0",
        "configurations": [
            {
                "type": "node",
                "request": "launch",
                "name": "Launch Program",
                "program": "${file}"
            }
        ]
    }
```

### C Sharp

- Create a new console project: `dotnet new console`
- `code .` to start a new workspace for this project
- Generate `tasks.json` and `launch.json` in VSCode
- Recommended extensions:
  - C#
  - XUnit or NUnit for unit tests
  - C# FixFormat
  - NuGet Package Manager
  - MSBuild project tools

### C/C++

- Install Mingw-w64 to get gcc (latest x86_64-posix-seh build) on Windows from <https://sourceforge.net/projects/mingw-w64/files/>
- Extract .7z package and add bin folder to path
- Add C/C++ Extension Pack
- Generate config files: `tasks.json`, `launch.json`, `c_cpp_properties.json`
- Recommended extensions:
  - CMake
  - vscode-clangd
  - C/C++ Project Generator
  - Native Debug

### HTML

- 推荐插件：HTML CSS Support
- 通过定义区域标记实现代码折叠

  ``` html
  <!-- #region -->
  <!-- #endregion -->
  ```

- 使用 Emmet 缩写扩展功能，按 Tab 或 Enter 键自动扩展成 HTML (css or pug flavor markups)
  - `#` for id
  - `.` for class
  - `>` for child
  - `+` for sibling
  - `*n` for repeating n times
  - `$` for continuous numbers, `$@n*m`, starting from n, creating m items
  - `()` for group
  - `[]` for customized attributes, e.g. `td[rowspan=2 colspan=3 title='test']`
  - `{}` for inner text, e.g. `p>{Click }+a{here}+{ to continue}`

Setting

``` json
"emmet.triggerExpansionOnTab": true
```

Example

``` emmet
#page>div.logo+ul#navigation>li*5>a(Item $)
```

References:

- <http://docs.emmet.io/abbreviations/>
- <https://code.visualstudio.com/docs/editor/emmet>

### CSS, SCSS, Less

- Ctrl+Space 触发 IntelliSense
- 颜色预览和颜色选择器（鼠标悬停）
- 代码折叠

``` css
/* #region */
/* #endregion */
```

## Create Visual Studio Extension

### Setup env

1. Install Visual Studio Code & Node.js;
2. Install Yeoman and VS Code Extension Generator;

``` console
npm install -g yo generator-code
yo code
```

插件清单文件：package.json
入口文件：src\extension.ts

package.json 中 activationEvents 定义插件在何种情况下被激活，contributes 定义插件的 commands
当插件被激活时，extensions.ts 里的 activate 函数被调用

Tutorial: <https://code.visualstudio.com/api/get-started/your-first-extension>

Important APIs: <https://code.visualstudio.com/api/references/vscode-api>

- Web View Panel
  - window.createWebviewPanel
  - window.registerWebviewPanelSerializer
- Status Bar
  - window.createStatusBarItem
  - StatusBarItem
- Tree View
  - window.createTreeView
  - window.registerTreeDataProvider
  - TreeView
  - TreeDataProvider
  - contributes.views
  - contributes.viewsContainers
- Tasks
  - tasks.registerTaskProvider
  - Task
  - ShellExecution
  - contributes.taskDefinitions
- Workspace
  - workspace.getWorkspaceFolder
  - workspace.onDidChangeWorkspaceFolders
