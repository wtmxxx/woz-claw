# Skill Zip Upload Design

## 1. Goal
在配置中心的 Skills 区增加 skill zip 上传能力：上传 `.zip` 后自动解压、自动写入当前用户的 skills 配置，并让新 skill 默认启用。

## 2. Hard Constraints
- 只支持 zip 上传，不改变现有的技能开关保存接口语义。
- 上传后的 skill 必须落在当前用户的 `.sandbox/skills/<user_id>/` 下。
- skill 目录名优先使用 `SKILL.md` frontmatter 中的 `name`。
- 如果 `SKILL.md` 没有 `name`，回退到 zip 文件名的归一化结果，且忽略版本号尾缀，例如 `multi-search-engine-2.0.1.zip` 回退为 `multi-search-engine`。
- 同名 skill 上传时覆盖旧目录，并更新用户技能配置为启用状态。
- 必须防止 zip 解压路径穿越。

## 3. Scope
### In Scope
- 新增 skill zip 上传 API。
- 前端配置中心 Skills 区新增上传入口。
- 自动解压、命名归一、写入 `skills.yaml`。
- 新增回归测试覆盖上传、覆盖、前端入口展示。

### Out of Scope
- 不修改聊天、记忆和会话历史的数据结构。
- 不新增技能编辑器或在线预览。
- 不改变现有 skill 开关列表的交互逻辑。

## 4. Architecture
采用“独立上传接口 + 共享技能配置落盘”的方式实现。后端新增一个专用上传端点，接收 zip 后先解压到临时目录，定位包含 `SKILL.md` 的 skill 根目录，再读取 frontmatter 的 `name` 字段决定最终目录名。最终目录写入 `.sandbox/skills/<user_id>/<skill_name>`，并同步更新该用户的 `skills.yaml`，将该 skill 设为 `enabled: true`。

前端只负责提供文件选择、上传和成功后刷新配置。技能列表依然由现有 `GET /api/settings` 返回，确保配置中心没有新的数据源分叉。

## 5. Data Flow
1. 用户在配置中心的 Skills 区选择一个 `.zip` 文件并提交。
2. 前端调用上传接口，携带 `user_id` 和文件内容。
3. 后端校验后缀与压缩包内容，解压到临时目录。
4. 后端查找 `SKILL.md`，读取 frontmatter `name`，决定 skill 目录名；如果缺失，则使用 zip 文件名去掉版本尾缀后的结果。
5. 后端删除同名旧目录，复制/移动 skill 目录到最终位置。
6. 后端写入或覆盖该用户的 `skills.yaml`，保证该 skill 已启用。
7. 前端刷新技能列表，用户立即看到新 skill。

## 6. Error Handling
- 非 zip 文件：返回 400，并提示仅支持 zip。
- 压缩包内找不到 `SKILL.md`：返回 400。
- `SKILL.md` 无法解析或没有可用 name：回退到 zip 文件名；如果回退结果也为空，返回 400。
- 解压遇到路径穿越：跳过非法成员并在最终校验失败时返回 400。
- 同名目标目录已存在：先安全删除再覆盖，避免旧内容残留。
- 写配置失败：返回 500，不对外暴露内部堆栈。

## 7. Testing Strategy
- API 测试覆盖 zip 上传成功后目录解压、`SKILL.md` name 取值、`skills.yaml` 自动启用。
- API 测试覆盖 `SKILL.md` 缺少 name 时的 zip 文件名回退。
- API 测试覆盖同名 skill 上传时的覆盖行为。
- 前端快照测试覆盖 Skills 区存在上传控件和上传按钮文案。
- 验证命令：`pytest -q tests/test_app_api.py tests/test_frontend_rendering.py tests/test_agent_prompt.py`

## 8. Acceptance Criteria
- 上传一个合法 zip 后，当前用户的技能目录中出现对应 skill 目录和 `SKILL.md`。
- `skills.yaml` 自动包含该 skill，且 `enabled` 为 `true`。
- 现有技能开关仍可正常保存和读取。
- 前端配置中心可以完成上传并刷新列表。